"""Embedding-backed semantic indexing and ranking for stored videos."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from typing import Any
import urllib.error
import urllib.request

from ..model_config import EmbeddingConfig, embedding_auto_index_enabled, embedding_config_from_env
from ..schemas import EmbeddingIndexResult
from .connection import MAX_TABLE_SCAN_ROWS, LanceConnection, filter_in_values, filter_literal
from .videos import find_video_row, video_payload_row

EmbedText = Callable[[str], list[float]]
TRANSCRIPT_CHUNK_CHARS = 1200
TRANSCRIPT_CHUNK_OVERLAP_CHARS = 200


class EmbeddingSearchError(RuntimeError):
    """Raised when semantic search cannot call or parse the embedding response."""


def auto_index_video_embedding(
    connection: LanceConnection,
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


def auto_index_transcript_chunks(
    connection: LanceConnection,
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
        payload = index_transcript_chunks(
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
            "detail": "Video has no stored transcript",
        }
    return {
        "status": "indexed",
        "video_id": payload["video_id"],
        "dimensions": payload["dimensions"],
    }


def index_transcript_chunks(
    connection: LanceConnection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any] | None:
    row = find_video_row(connection, youtube_video_id)
    if row is None or not row["transcript"]:
        return None

    chunks = split_transcript_chunks(row["transcript"])
    if not chunks:
        return None

    existing_rows = {row["chunk_index"]: row for row in connection.rows("video_transcript_chunks") if row["video_id"] == youtube_video_id}
    indexed_rows: list[dict[str, Any]] = []
    dimensions: int | None = None
    for chunk_index, chunk_text in enumerate(chunks):
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        existing_row = existing_rows.get(chunk_index)
        if existing_row is not None and existing_row["embedding_content_hash"] == content_hash and existing_row["embedding"] is not None:
            dimensions = len(existing_row["embedding"])
            continue

        embedding = embed_text(chunk_text) if embed_text is not None else embed_text_with_gemini(chunk_text, config=embedding_config)
        if not embedding:
            raise EmbeddingSearchError("Embedding request returned an empty vector")
        dimensions = len(embedding)
        connection.ensure_transcript_chunk_table(dimensions)
        indexed_rows.append(
            {
                "video_id": youtube_video_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "embedding_content_hash": content_hash,
                "embedding": [float(value) for value in embedding],
            }
        )

    connection.upsert_rows(
        "video_transcript_chunks",
        ["video_id", "chunk_index"],
        indexed_rows,
    )
    connection.delete_where(
        "video_transcript_chunks",
        f"video_id = {filter_literal(youtube_video_id)} AND chunk_index >= {len(chunks)}",
    )
    return {
        "video_id": youtube_video_id,
        "dimensions": dimensions or 0,
    }


def split_transcript_chunks(
    transcript: str,
    *,
    chunk_chars: int = TRANSCRIPT_CHUNK_CHARS,
    overlap_chars: int = TRANSCRIPT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    clean_transcript = " ".join(transcript.split())
    if not clean_transcript:
        return []
    if chunk_chars <= 0:
        raise ValueError("transcript chunk size must be greater than 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("transcript chunk overlap must be non-negative and smaller than chunk size")

    chunks: list[str] = []
    start = 0
    while start < len(clean_transcript):
        end = min(len(clean_transcript), start + chunk_chars)
        chunk = clean_transcript[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(clean_transcript):
            break
        start = end - overlap_chars
    return chunks


def search_transcript_chunk_scores(
    connection: LanceConnection,
    query: str,
    video_ids: list[str],
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, float]:
    if not video_ids or connection.optional_table("video_transcript_chunks") is None:
        return {}

    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    rows = (
        connection.table("video_transcript_chunks")
        .search(query_embedding, vector_column_name="embedding")
        .metric("cosine")
        .where(f"video_id IN ({filter_in_values(video_ids)})")
        .limit(MAX_TABLE_SCAN_ROWS)
        .to_list()
    )
    scores: dict[str, float] = {}
    for row in rows:
        score = 1.0 - float(row["_distance"])
        video_id = row["video_id"]
        scores[video_id] = max(score, scores.get(video_id, float("-inf")))
    return scores


def list_video_ids_for_embedding_index(
    connection: LanceConnection,
    *,
    require_transcript: bool = True,
) -> list[str]:
    rows = [row for row in connection.rows("videos") if not require_transcript or row["transcript"]]
    return [
        row["youtube_video_id"]
        for row in sorted(
            rows,
            key=lambda row: (
                row["upload_date"] or "",
                row["youtube_video_id"],
            ),
            reverse=True,
        )
    ]


def index_video_embedding(
    connection: LanceConnection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any] | None:
    video = find_video_row(connection, youtube_video_id)
    if video is None:
        return None

    row = video_payload_row(connection, video)
    source_text = embedding_source_text(row)
    content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    existing_row = next(
        (row for row in connection.rows("video_embeddings") if row["video_id"] == youtube_video_id),
        None,
    )
    if existing_row is not None and existing_row["embedding_content_hash"] == content_hash and existing_row["embedding"] is not None:
        return {
            "video_id": youtube_video_id,
            "content_hash": content_hash,
            "dimensions": len(existing_row["embedding"]),
        }

    embedding = embed_text(source_text) if embed_text is not None else embed_text_with_gemini(source_text, config=embedding_config)
    if not embedding:
        raise EmbeddingSearchError("Embedding request returned an empty vector")
    connection.ensure_video_embedding_table(len(embedding))
    connection.upsert_rows(
        "video_embeddings",
        "video_id",
        [
            {
                "video_id": youtube_video_id,
                "embedding_content_hash": content_hash,
                "embedding": [float(value) for value in embedding],
            }
        ],
    )
    return {
        "video_id": youtube_video_id,
        "content_hash": content_hash,
        "dimensions": len(embedding),
    }


def embedding_source_text(row: Mapping[str, Any]) -> str:
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
