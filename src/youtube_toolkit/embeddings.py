"""Configured embedding API client shared by stateless search and storage indexing."""

from __future__ import annotations

from typing import Any
import json
import time
import urllib.error
import urllib.request

from .execution import retry_transient
from .model_config import EmbeddingConfig, embedding_config_from_env

EMBEDDING_REQUEST_ATTEMPTS = 3
EMBEDDING_RETRY_BACKOFF_SECONDS = 0.25


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding API cannot return a usable vector."""


def embed_text(
    text: str,
    *,
    config: EmbeddingConfig | None = None,
) -> list[float]:
    """Call the configured embedding endpoint and return the numeric vector."""
    try:
        resolved_config = config or embedding_config_from_env()
    except ValueError as error:
        raise EmbeddingError(str(error)) from error
    payload = {
        "model": resolved_config.model,
        "input": text,
        "dimensions": resolved_config.output_dimensionality,
    }
    response = post_embedding_json(
        resolved_config.endpoint("embeddings"),
        resolved_config,
        payload,
    )
    data = response.get("data")
    first_embedding = data[0] if isinstance(data, list) and data else {}
    values = first_embedding.get("embedding") if isinstance(first_embedding, dict) else None
    if values is None:
        values = response.get("embedding")
    if not isinstance(values, list):
        raise EmbeddingError("Embedding response did not include values")
    return [float(value) for value in values]


def post_embedding_json(
    url: str,
    config: EmbeddingConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST one embedding request with bounded retries for transient provider failures."""
    if not url.startswith(f"{config.base_url}/"):
        raise EmbeddingError("Unsupported embedding URL")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    def request_payload() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            response_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise json.JSONDecodeError("Embedding response was not a JSON object", "", 0)
        return response_payload

    try:
        return retry_transient(
            request_payload,
            attempts=EMBEDDING_REQUEST_ATTEMPTS,
            backoff_seconds=EMBEDDING_RETRY_BACKOFF_SECONDS,
            retryable_errors=(OSError, json.JSONDecodeError),
            sleep=time.sleep,
        )
    except urllib.error.HTTPError as error:
        message = compact_embedding_error(error.read().decode("utf-8", errors="replace"))
        raise EmbeddingError(message or f"Embedding request HTTP {error.code}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingError(str(error)) from error


def compact_embedding_error(text: str) -> str:
    """Extract a short useful message from an embedding API error payload."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return " ".join(text.split())[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return " ".join(str(error.get("message", "")).split())[:500]
    return ""
