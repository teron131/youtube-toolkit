"""Embedding-backed semantic indexing and ranking for stored videos."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import struct
from typing import Any
import urllib.error
import urllib.request

import turso

from ..model_config import EmbeddingConfig, embedding_auto_index_enabled, embedding_config_from_env
from ..schemas import EmbeddingIndexResult

EmbedText = Callable[[str], list[float]]


class EmbeddingSearchError(RuntimeError):
    """Raised when semantic search cannot call or parse the embedding response."""


def auto_index_video_embedding(
    connection: turso.Connection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
) -> EmbeddingIndexResult:
    try:
        if not embedding_auto_index_enabled():
            return {
                "status": "skipped",
                "video_id": youtube_video_id,
                "detail": "Auto embedding is disabled",
            }
        payload = index_video_embedding(
            connection,
            youtube_video_id,
            embed_text=embed_text,
        )
    except Exception as error:
        return {
            "status": "failed",
            "video_id": youtube_video_id,
            "error": str(error),
        }
    if payload is None:
        return {
            "status": "skipped",
            "video_id": youtube_video_id,
            "detail": "Video is not stored",
        }
    return {
        "status": "indexed",
        "video_id": payload["video_id"],
        "dimensions": payload["dimensions"],
    }


def list_video_ids_for_embedding_index(
    connection: turso.Connection,
    *,
    require_transcript: bool = True,
) -> list[str]:
    rows = connection.execute(
        """
        SELECT youtube_video_id
        FROM videos
        WHERE (
            ? = 0
            OR (transcript IS NOT NULL AND transcript != '')
        )
        ORDER BY upload_date DESC, youtube_video_id DESC
        """,
        (int(require_transcript),),
    ).fetchall()
    return [row["youtube_video_id"] for row in rows]


def index_video_embedding(
    connection: turso.Connection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any] | None:
    connection.commit()
    row = connection.execute(
        """
        SELECT
            videos.youtube_video_id,
            videos.title,
            channels.channel_handle,
            videos.transcript,
            videos.embedding_content_hash,
            videos.embedding
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.youtube_channel_id = videos.channel_id
        WHERE videos.youtube_video_id = ?
        LIMIT 1
        """,
        (youtube_video_id,),
    ).fetchone()
    if row is None:
        return None

    source_text = embedding_source_text(row)
    content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if row["embedding_content_hash"] == content_hash and row["embedding"] is not None:
        return {
            "video_id": row["youtube_video_id"],
            "content_hash": content_hash,
            "dimensions": len(row["embedding"]) // 4,
        }

    embedding = embed_text(source_text) if embed_text is not None else embed_text_with_gemini(source_text, config=embedding_config)
    if not embedding:
        raise EmbeddingSearchError("Embedding request returned an empty vector")
    connection.execute(
        """
        UPDATE videos
        SET embedding_content_hash = ?,
            embedding = ?
        WHERE youtube_video_id = ?
        """,
        (
            content_hash,
            struct.pack(f"<{len(embedding)}f", *embedding),
            row["youtube_video_id"],
        ),
    )
    connection.commit()
    return {
        "video_id": row["youtube_video_id"],
        "content_hash": content_hash,
        "dimensions": len(embedding),
    }


def embedding_source_text(row: turso.Row) -> str:
    return "\n".join(
        str(part)
        for part in (
            row["title"],
            row["channel_handle"],
            row["transcript"],
        )
        if part not in (None, "")
    )


def embed_text_with_gemini(
    text: str,
    *,
    config: EmbeddingConfig | None = None,
) -> list[float]:
    try:
        resolved_config = config or embedding_config_from_env()
    except ValueError as error:
        raise EmbeddingSearchError(str(error)) from error
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


def encode_vector(embedding: list[float]) -> str:
    """Encode an embedding vector for libSQL vector conversion."""
    return json.dumps(embedding, separators=(",", ":"))


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
