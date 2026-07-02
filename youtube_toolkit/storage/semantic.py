"""Embedding-backed semantic indexing and ranking for stored videos."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Any
import urllib.error
import urllib.request

import turso

from ..model_config import EmbeddingConfig, embedding_config_from_env

EmbedText = Callable[[str], list[float]]


class EmbeddingSearchError(RuntimeError):
    """Raised when semantic search cannot call or parse the embedding response."""


def index_video_embedding(
    connection: turso.Connection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT
            youtube_video_id,
            title,
            upload_date,
            transcript,
            embedding_content_hash,
            CASE
                WHEN embedding IS NULL THEN NULL
                ELSE vector_extract(embedding)
            END AS embedding_text
        FROM videos
        WHERE youtube_video_id = ?
        LIMIT 1
        """,
        (youtube_video_id,),
    ).fetchone()
    if row is None:
        return None

    source_text = "\n".join(
        str(part)
        for part in (
            row["youtube_video_id"],
            row["title"],
            row["upload_date"],
            row["transcript"],
        )
        if part not in (None, "")
    )
    content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if row["embedding_content_hash"] == content_hash and row["embedding_text"]:
        return embedding_index_payload(
            row,
            content_hash,
            decode_extracted_vector(row["embedding_text"]),
        )

    embedding = embed_text(source_text) if embed_text is not None else embed_text_with_gemini(source_text, config=embedding_config)
    if not embedding:
        raise EmbeddingSearchError("Embedding request returned an empty vector")
    connection.execute(
        """
        UPDATE videos
        SET embedding_content_hash = ?,
            embedding = vector32(?)
        WHERE youtube_video_id = ?
        """,
        (
            content_hash,
            encode_vector(embedding),
            row["youtube_video_id"],
        ),
    )
    return embedding_index_payload(row, content_hash, embedding)


def embedding_index_payload(
    row: turso.Row,
    content_hash: str,
    embedding: list[float],
) -> dict[str, Any]:
    return {
        "video_id": row["youtube_video_id"],
        "content_hash": content_hash,
        "dimensions": len(embedding),
    }


def embed_text_with_gemini(
    text: str,
    *,
    config: EmbeddingConfig | None = None,
) -> list[float]:
    resolved_config = config or gemini_embedding_config()
    payload = {
        "model": f"models/{resolved_config.model}",
        "content": {"parts": [{"text": text}]},
        "output_dimensionality": resolved_config.output_dimensionality,
    }
    response = post_embedding_json(
        resolved_config.endpoint(f"models/{resolved_config.model}:embedContent"),
        resolved_config,
        payload,
    )
    embedding = response.get("embedding")
    values = embedding.get("values") if isinstance(embedding, dict) else response.get("values")
    if not isinstance(values, list):
        raise EmbeddingSearchError("Embedding response did not include values")
    return [float(value) for value in values]


def post_embedding_json(
    url: str,
    config: EmbeddingConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not url.startswith(f"{config.base_url}/"):
        raise EmbeddingSearchError("Unsupported embedding URL")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": config.api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = compact_embedding_error(error.read().decode("utf-8", errors="replace"))
        raise EmbeddingSearchError(message or f"Embedding request HTTP {error.code}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingSearchError(str(error)) from error


def gemini_embedding_config() -> EmbeddingConfig:
    try:
        return embedding_config_from_env()
    except ValueError as error:
        raise EmbeddingSearchError(str(error)) from error


def encode_vector(embedding: list[float]) -> str:
    """Encode an embedding vector for libSQL vector conversion."""
    return json.dumps(embedding, separators=(",", ":"))


def decode_extracted_vector(value: str) -> list[float]:
    """Decode a vector_extract result into floats."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise EmbeddingSearchError("Stored embedding is not valid vector text") from error
    if not isinstance(parsed, list):
        raise EmbeddingSearchError("Stored embedding is not a vector")
    return [float(item) for item in parsed]


def compact_embedding_error(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return " ".join(text.split())[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return " ".join(str(error.get("message", "")).split())[:500]
    return ""
