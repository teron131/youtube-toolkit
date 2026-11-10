"""Ask a decision model typed questions through TypeSafe's SDK in its System One format, which Vercel AI Gateway and OpenRouter both accept with images, so DECISION_BASE_URL alone picks the host.

Vercel's native /v1/evaluate endpoint is not used because it accepts image assets but silently ignores them.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import AliasPath, BaseModel, Field
from typesafe_sdk import (
    Answer,
    Question,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
    Usage,
)

from .config import DecisionConfig
from .execution import collect_concurrently

REQUEST_TIMEOUT_SECONDS = 60
MAX_WORKERS = 32
# Overloaded hosts answer 503 "try again shortly".
MAX_RETRIES = 4
# Overload comes in spells, partly from the fan-out itself, so requests still failing get one more round once the first has ended, a few at a time.
RETRY_WORKERS = 4
# TypeSafe's opt-in for image assets; Vercel AI Gateway rejects image requests without it, and the SDK never sends it.
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


class DecisionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DecisionInput:
    state: str | list[str]
    assets: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Token counts, cost, and provider time are None when the host does not report them."""

    answers: dict[str, Answer]
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    # The gateway's calls to model providers, summed over its attempts; the trip to the gateway and its own work are left out.
    provider_seconds: float | None


def image_asset(jpeg: bytes) -> dict[str, str]:
    """A JPEG for the `assets` map; models read it where its key appears as a whole string in `state`."""
    return {
        "type": "image",
        "format": "jpg",
        "data_base64": base64.b64encode(jpeg).decode("ascii"),
    }


def decide_each(
    config: DecisionConfig,
    *,
    questions: Mapping[str, Question],
    inputs: list[DecisionInput],
) -> list[DecisionResult | DecisionError]:
    """Rows align with inputs, holding a DecisionError for each request that failed both rounds.

    When every request fails, there is no second round: a bad key, a bad model, or a host that is down will not recover in seconds.
    """
    with TypeSafeClient(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=IMAGES_HEADER,
        # The SDK's default 30-second retry budget would never retry a request that timed out.
        retry=RetryPolicy(max_retries=MAX_RETRIES, timeout=None),
        # The SDK compresses bodies with zstd on Python 3.14+, which OpenRouter does not document accepting.
        compression=False,
    ) as client:

        def ask(item: DecisionInput) -> DecisionResult:
            try:
                reply = client.system_one(
                    item.state,
                    questions,
                    # Base64 assets in the JSON body replace the SDK's multipart image upload, which OpenRouter rejects.
                    extra_body={"assets": item.assets} if item.assets else None,
                    response_model=_Reply,
                )
            except TypeSafeError as error:
                raise DecisionError(
                    f"{config.host} decision request failed: {error}"
                ) from error
            return reply.result()

        jobs = dict(enumerate(inputs))
        completed, failures = collect_concurrently(jobs, ask, max_workers=MAX_WORKERS)
        if completed and failures:
            retried, failures = collect_concurrently(
                {index: jobs[index] for index in failures},
                ask,
                max_workers=RETRY_WORKERS,
            )
            completed |= retried
    rows: list[DecisionResult | DecisionError] = []
    for index in range(len(inputs)):
        if index in completed:
            rows.append(completed[index])
            continue
        error = failures[index]
        if not isinstance(error, DecisionError):
            raise error
        rows.append(error)
    return rows


class _VercelProviderAttempt(BaseModel):
    startTime: float
    endTime: float


class _VercelModelAttempt(BaseModel):
    providerAttempts: list[_VercelProviderAttempt]


class _VercelRouting(BaseModel):
    modelAttempts: list[_VercelModelAttempt]


class _VercelGateway(BaseModel):
    """Unlike the SDK's strict models, this accepts the decimal string Vercel sends as cost."""

    cost: float
    routing: _VercelRouting

    @property
    def provider_seconds(self) -> float:
        return (
            sum(
                attempt.endTime - attempt.startTime
                for model in self.routing.modelAttempts
                for attempt in model.providerAttempts
            )
            / 1000
        )


class _Usage(Usage):
    cost: float | None = None


class _Reply(SystemOneResponse):
    """The SDK's reply carries answers and token counts only, so cost and provider time come from what each host adds: Vercel's gateway block, or OpenRouter's cost in usage."""

    usage: _Usage
    vercel: _VercelGateway | None = Field(
        None, validation_alias=AliasPath("provider_metadata", "gateway")
    )

    def result(self) -> DecisionResult:
        vercel = self.vercel
        return DecisionResult(
            self.answers,
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.usage.cost if vercel is None else vercel.cost,
            None if vercel is None else vercel.provider_seconds,
        )
