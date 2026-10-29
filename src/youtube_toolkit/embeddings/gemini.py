"""Embed JPEG images and text separately through Gemini, preserving input alignment and reported usage."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING
import urllib.error
import urllib.request

from ..execution import collect_concurrently, retry_transient
from .results import EmbeddingError, EmbeddingResult, EmbeddingUsage, sum_reported

if TYPE_CHECKING:
    import numpy as np

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents"
# On 384 frames at 960x540, 64 per request x 4 concurrent uploaded about twice as fast as 32 x 2 with no retries; 100 per request (the API maximum) risks Gemini's inline request size limit with dense frames.
BATCH_SIZE = 64
MAX_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = 90
REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclass(frozen=True, slots=True)
class _BatchResult:
    vectors: np.ndarray
    usage: EmbeddingUsage


class GeminiEmbeddingClient:
    def __init__(self, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("Gemini embedding API key must not be empty")
        self._api_key = api_key.strip()
        self.model = MODEL

    def embed(self, image_paths: list[Path], texts: list[str]) -> EmbeddingResult:
        """Duplicate inputs share requests within this call, but output rows preserve every supplied input in order.

        A failed batch fails the call; partial vectors are never returned as a complete ranking input.
        """
        import numpy as np

        inputs: dict[bytes | str, int] = {}
        image_rows = []
        for path in image_paths:
            try:
                content = path.read_bytes()
            except OSError as error:
                raise EmbeddingError("Could not read an embedding image") from error
            image_rows.append(inputs.setdefault(content, len(inputs)))
        text_rows = [inputs.setdefault(text, len(inputs)) for text in texts]
        unique = list(inputs)
        if not unique:
            return EmbeddingResult(
                np.empty((0, DIMENSIONS), dtype=np.float32),
                np.empty((0, DIMENSIONS), dtype=np.float32),
                EmbeddingUsage(0, 0, 0),
            )
        jobs = {
            offset: unique[offset : offset + BATCH_SIZE]
            for offset in range(0, len(unique), BATCH_SIZE)
        }
        completed, failures = collect_concurrently(
            jobs, self._embed_batch, max_workers=MAX_WORKERS
        )
        if failures:
            error = failures[min(failures)]
            if isinstance(error, EmbeddingError):
                raise error
            raise EmbeddingError("Gemini embedding batch failed") from error
        batches = [completed[offset] for offset in jobs]
        vectors = np.concatenate([batch.vectors for batch in batches])
        return EmbeddingResult(
            vectors[image_rows],
            vectors[text_rows],
            EmbeddingUsage(
                sum_reported([batch.usage.image_tokens for batch in batches]),
                sum_reported([batch.usage.text_tokens for batch in batches]),
                sum_reported([batch.usage.total_tokens for batch in batches]),
            ),
        )

    def _embed_batch(self, inputs: list[bytes | str]) -> _BatchResult:
        requests = []
        for content in inputs:
            part = (
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(content).decode("ascii"),
                    }
                }
                if isinstance(content, bytes)
                else {"text": content}
            )
            requests.append(
                {
                    "model": f"models/{MODEL}",
                    "content": {"parts": [part]},
                    "outputDimensionality": DIMENSIONS,
                }
            )
        request = urllib.request.Request(  # noqa: S310 -- Fixed HTTPS Gemini endpoint.
            ENDPOINT,
            data=json.dumps({"requests": requests}).encode("utf-8"),
            headers={
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        def send() -> object:
            with urllib.request.urlopen(  # noqa: S310 -- Fixed HTTPS Gemini endpoint.
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                return json.load(response)

        try:
            payload = retry_transient(
                send,
                attempts=3,
                backoff_seconds=1,
                retryable_errors=(
                    urllib.error.URLError,
                    TimeoutError,
                    http.client.IncompleteRead,
                ),
            )
        except urllib.error.HTTPError as error:
            reason = _error_reason(error)
            raise EmbeddingError(
                f"Gemini embedding request failed (HTTP {error.code}"
                + (f", {reason})" if reason else ")")
            ) from error
        except OSError as error:
            raise EmbeddingError("Gemini embedding connection failed") from error
        except http.client.IncompleteRead as error:
            raise EmbeddingError(
                "Gemini embedding response was cut off before it finished"
            ) from error
        except ValueError as error:
            raise EmbeddingError(
                "Gemini embedding response is not valid JSON"
            ) from error
        return _parse_response(payload, inputs)


def _error_reason(error: urllib.error.HTTPError) -> str | None:
    """Return Google's reason code, such as API_KEY_INVALID, never the message text, which can quote request content."""
    try:
        payload = json.loads(error.read())
    except (OSError, ValueError):
        return None
    detail = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(detail, dict):
        return None
    codes = [
        item.get("reason")
        for item in detail.get("details") or []
        if isinstance(item, dict)
    ] + [detail.get("status")]
    return next(
        (
            code
            for code in codes
            if isinstance(code, str) and REASON_CODE.fullmatch(code)
        ),
        None,
    )


def _parse_response(payload: object, inputs: list[bytes | str]) -> _BatchResult:
    import numpy as np

    if not isinstance(payload, dict):
        raise EmbeddingError("Gemini embedding response must be an object")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
        raise EmbeddingError("Gemini embedding response has the wrong vector count")
    vectors = []
    for embedding in embeddings:
        values = embedding.get("values") if isinstance(embedding, dict) else None
        if (
            not isinstance(values, list)
            or len(values) != DIMENSIONS
            or any(
                isinstance(value, bool) or not isinstance(value, int | float)
                for value in values
            )
        ):
            raise EmbeddingError(
                "Gemini embedding vector has invalid dimensions or values"
            )
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (ValueError, OverflowError) as error:
            raise EmbeddingError(
                "Gemini embedding vector has invalid numeric values"
            ) from error
        if not np.isfinite(vector).all() or not np.any(vector):
            raise EmbeddingError("Gemini embedding vector must be finite and nonzero")
        vector /= np.max(np.abs(vector))
        vector /= np.linalg.norm(vector)
        vectors.append(vector)
    return _BatchResult(
        np.asarray(vectors, dtype=np.float32),
        _parse_usage(payload.get("usageMetadata"), inputs),
    )


def _parse_usage(payload: object, inputs: list[bytes | str]) -> EmbeddingUsage:
    if payload is None:
        return EmbeddingUsage(None, None, None)
    if not isinstance(payload, dict):
        raise EmbeddingError("Gemini embedding usage must be an object")
    total = _token_count(payload.get("promptTokenCount"))
    details = payload.get("promptTokenDetails")
    if details is None:
        return EmbeddingUsage(None, None, total)
    if not isinstance(details, list):
        raise EmbeddingError("Gemini embedding modality usage must be a list")
    counts: dict[str, list[int | None]] = {"IMAGE": [], "TEXT": []}
    for detail in details:
        if not isinstance(detail, dict) or not isinstance(detail.get("modality"), str):
            raise EmbeddingError("Gemini embedding modality usage is invalid")
        modality = detail["modality"]
        if modality in counts:
            counts[modality].append(_token_count(detail.get("tokenCount")))
    return EmbeddingUsage(
        sum_reported(counts["IMAGE"])
        if counts["IMAGE"] or not any(isinstance(item, bytes) for item in inputs)
        else None,
        sum_reported(counts["TEXT"])
        if counts["TEXT"] or not any(isinstance(item, str) for item in inputs)
        else None,
        total,
    )


def _token_count(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EmbeddingError(
            "Gemini embedding token count must be a nonnegative integer"
        )
    return value
