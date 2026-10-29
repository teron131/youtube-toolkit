"""Ask a decision model typed questions in TypeSafe's System One format, which Vercel AI Gateway and OpenRouter both accept with images, so DECISION_URL alone picks the host.

Vercel's native /v1/evaluate endpoint is not used because it accepts image assets but silently ignores them.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import http.client
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from .config import DecisionConfig
from .execution import collect_concurrently, retry_transient

REQUEST_TIMEOUT_SECONDS = 60
# For 690 vision frames on Decider Flash, 8 in flight took 102 s, 16 took 50 s, and 32 took 27 s at the same 1.2 s per request; 64 and 128 drew 503 overload errors under that sustained load.
MAX_WORKERS = 32
# Overloaded hosts answer 503 "try again shortly".
ATTEMPTS = 5
# Overload comes in spells, partly from the fan-out itself, so requests still failing get one more round once the first has ended, a few at a time.
RETRY_WORKERS = 4
# TypeSafe's opt-in for image assets; Vercel AI Gateway rejects image requests without it.
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


class DecisionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DecisionInput:
    state: str | list[Any] | dict[str, Any]
    assets: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Answers are keyed by question name in TypeSafe's shape; token counts and cost are None when the host does not report them."""

    answers: dict[str, dict[str, Any]]
    input_tokens: int | None
    cost_usd: float | None


def image_asset(path: Path) -> dict[str, str]:
    """A JPEG for the `assets` map; models read it where its key appears as a whole string in `state`."""
    return {
        "type": "image",
        "format": "jpg",
        "data_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def decide_each(
    config: DecisionConfig,
    *,
    questions: dict[str, dict[str, Any]],
    inputs: list[DecisionInput],
) -> list[DecisionResult | DecisionError]:
    """Ask the same questions about each input, MAX_WORKERS requests at a time; rows align with inputs and hold the error of a request that failed both rounds.

    When every request fails, there is no second round: a bad key, a bad model, or a host that is down will not recover in seconds.
    """

    def ask(item: DecisionInput) -> DecisionResult:
        return decide(
            config,
            state=item.state,
            questions=questions,
            assets=item.assets,
        )

    jobs = dict(enumerate(inputs))
    completed, failures = collect_concurrently(jobs, ask, max_workers=MAX_WORKERS)
    if completed and failures:
        retried, failures = collect_concurrently(
            {index: jobs[index] for index in failures}, ask, max_workers=RETRY_WORKERS
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


def decide(
    config: DecisionConfig,
    *,
    state: str | list[Any] | dict[str, Any],
    questions: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, str]],
) -> DecisionResult:
    body = {"model": config.model, "state": state, "questions": questions}
    if assets:
        body["assets"] = assets
    request = urllib.request.Request(  # noqa: S310 -- DecisionConfig allows only http and https URLs.
        config.url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **IMAGES_HEADER,
        },
        method="POST",
    )

    def send() -> object:
        with urllib.request.urlopen(  # noqa: S310 -- DecisionConfig allows only http and https URLs.
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return json.load(response)

    try:
        payload = retry_transient(
            send,
            attempts=ATTEMPTS,
            backoff_seconds=1,
            retryable_errors=(
                urllib.error.URLError,
                TimeoutError,
                http.client.IncompleteRead,
            ),
        )
    except urllib.error.HTTPError as error:
        raise DecisionError(
            f"{config.host} decision request failed (HTTP {error.code}): {_error_message(error)}"
        ) from error
    except (OSError, http.client.IncompleteRead) as error:
        raise DecisionError(f"{config.host} decision connection failed") from error
    except ValueError as error:
        raise DecisionError(
            f"{config.host} decision response is not valid JSON"
        ) from error
    return _parse(config, payload)


def _parse(config: DecisionConfig, payload: object) -> DecisionResult:
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(answers, dict):
        raise DecisionError(f"{config.host} decision response has no answers")
    usage = payload.get("usage") or {}
    gateway = (payload.get("provider_metadata") or {}).get("gateway") or {}
    cost = gateway.get("cost", usage.get("cost"))
    tokens = usage.get("input_tokens")
    return DecisionResult(
        answers,
        tokens if isinstance(tokens, int) else None,
        None if cost is None else float(cost),
    )


def _error_message(error: urllib.error.HTTPError) -> str:
    """TypeSafe-style hosts put the reason in `message`, OpenRouter in `error.message`."""
    try:
        payload = json.loads(error.read())
    except (OSError, ValueError):
        return "no readable error body"
    if not isinstance(payload, dict):
        return "no readable error body"
    nested = payload.get("error")
    message = payload.get("message") or (
        nested.get("message") if isinstance(nested, dict) else None
    )
    return str(message) if message else "no error message"
