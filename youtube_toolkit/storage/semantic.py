"""Embedding-backed semantic indexing and ranking for stored videos."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from typing import Any
import urllib.error
import urllib.request

from lancedb.rerankers import RRFReranker

from ..model_config import EmbeddingConfig, embedding_auto_index_enabled, embedding_concurrency, embedding_config_from_env
from ..schemas import EmbeddingIndexResult
from .connection import MAX_TABLE_SCAN_ROWS, LanceConnection, filter_in_values, filter_literal, transcript_chunk_metadata
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
    """Best-effort wrapper for maintaining video embeddings during write workflows."""
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


def auto_index_video_title_embedding(
    connection: LanceConnection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
) -> EmbeddingIndexResult:
    """Best-effort wrapper for maintaining title embeddings during search workflows."""
    try:
        if not embedding_auto_index_enabled():
            return {
                "status": "skipped",
                "video_id": youtube_video_id,
                "detail": "Auto embedding is disabled",
            }
        payload = index_video_title_embedding(
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
            "detail": "Video has no stored title",
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
    """Best-effort wrapper for maintaining transcript chunk embeddings during semantic search."""
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
    """Index stored transcript text into reusable overlapping chunks with content-hash reuse."""
    row = find_video_row(connection, youtube_video_id)
    if row is None or not row["transcript"]:
        return None

    video = video_payload_row(connection, row)
    video_metadata = transcript_chunk_metadata(video)
    chunks = split_transcript_chunks(row["transcript"])
    if not chunks:
        return None

    existing_rows = {
        row["chunk_index"]: row
        for row in connection.rows(
            "video_transcript_chunks",
            where_sql=f"video_id = {filter_literal(youtube_video_id)}",
        )
    }
    indexed_rows: list[dict[str, Any]] = []
    dimensions: int | None = None
    for chunk_index, chunk_text in enumerate(chunks):
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        existing_row = existing_rows.get(chunk_index)
        if existing_row is not None and existing_row["embedding_content_hash"] == content_hash and existing_row["embedding"] is not None:
            dimensions = len(existing_row["embedding"])
            if any(existing_row.get(column_name) != value for column_name, value in video_metadata.items()):
                indexed_rows.append(
                    {
                        **existing_row,
                        **video_metadata,
                    }
                )
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
                **video_metadata,
                "embedding_content_hash": content_hash,
                "embedding": [float(value) for value in embedding],
            }
        )

    connection.upsert_rows(
        "video_transcript_chunks",
        ["video_id", "chunk_index"],
        indexed_rows,
    )
    connection.ensure_transcript_chunk_text_index()
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
    """Split normalized transcript text into overlapping chunks for embedding search."""
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
    video_ids: list[str] | None,
    *,
    where_sql: str | None = None,
    min_similarity: float | None = None,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, float]:
    """Return each video's best semantic transcript-chunk score for a query."""
    rows = search_transcript_chunks(
        connection,
        query,
        video_ids,
        where_sql=where_sql,
        min_similarity=min_similarity,
        embed_text=embed_text,
        embedding_config=embedding_config,
    )
    scores: dict[str, float] = {}
    for row in rows:
        score = float(row["similarity"])
        video_id = row["video_id"]
        scores[video_id] = max(score, scores.get(video_id, float("-inf")))
    return scores


def search_transcript_chunks(
    connection: LanceConnection,
    query: str,
    video_ids: list[str] | None,
    *,
    limit: int = MAX_TABLE_SCAN_ROWS,
    where_sql: str | None = None,
    columns: list[str] | None = None,
    min_similarity: float | None = None,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[dict[str, Any]]:
    """Search transcript chunk embeddings with optional video and SQL filters."""
    if video_ids == [] or connection.optional_table("video_transcript_chunks") is None:
        return []
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    rows = transcript_chunk_search_query(
        connection,
        video_ids,
        query_embedding,
        limit=limit,
        where_sql=where_sql,
    ).to_list()
    rows = rows_with_similarity(rows, query_embedding)
    if min_similarity is not None:
        rows = [row for row in rows if row["similarity"] >= min_similarity]
    rows = rows[:limit]
    if columns is None:
        return rows
    return [
        {
            **{column_name: row[column_name] for column_name in columns if column_name in row},
            "similarity": row["similarity"],
        }
        for row in rows
    ]


def search_video_title_scores(
    connection: LanceConnection,
    query: str,
    video_ids: list[str],
    *,
    limit: int = MAX_TABLE_SCAN_ROWS,
    min_similarity: float | None = None,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, float]:
    """Return semantic title scores keyed by video ID for requested stored videos."""
    rows = search_video_titles(
        connection,
        query,
        video_ids,
        limit=limit,
        min_similarity=min_similarity,
        embed_text=embed_text,
        embedding_config=embedding_config,
    )
    return {row["video_id"]: float(row["similarity"]) for row in rows}


def search_video_titles(
    connection: LanceConnection,
    query: str,
    video_ids: list[str],
    *,
    limit: int = MAX_TABLE_SCAN_ROWS,
    min_similarity: float | None = None,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[dict[str, Any]]:
    """Search stored title embeddings with hybrid text/vector ranking and cosine scoring."""
    if not video_ids or connection.optional_table("video_title_embeddings") is None:
        return []
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    connection.ensure_video_title_text_index()
    rows = (
        connection.table("video_title_embeddings")
        .search(query_type="hybrid")
        .vector(query_embedding)
        .text(query)
        .where(f"video_id IN ({filter_in_values(video_ids)})")
        .rerank(RRFReranker())
        .limit(limit)
        .to_list()
    )
    rows = rows_with_similarity(rows, query_embedding)
    if min_similarity is not None:
        rows = [row for row in rows if row["similarity"] >= min_similarity]
    return rows[:limit]


def rows_with_similarity(
    rows: list[dict[str, Any]],
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    """Attach cosine similarity and sort vector-search rows by that score."""
    return sorted(
        [
            {
                **row,
                "similarity": cosine_similarity(query_embedding, row.get("embedding")),
            }
            for row in rows
            if row.get("embedding") is not None
        ],
        key=lambda row: row["similarity"],
        reverse=True,
    )


def cosine_similarity(
    left: list[float],
    right: Any,
) -> float:
    """Compute cosine similarity for same-length numeric vectors, treating invalid rows as zero."""
    if not isinstance(right, list) or len(left) != len(right):
        return 0.0
    if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in right):
        return 0.0
    right_values = [float(value) for value in right]
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right_values, strict=True))
    left_norm = math.sqrt(sum(left_value * left_value for left_value in left))
    right_norm = math.sqrt(sum(right_value * right_value for right_value in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def explain_transcript_chunk_search(
    connection: LanceConnection,
    query: str,
    video_ids: list[str],
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> str:
    """Return LanceDB's plan explanation for a transcript chunk search."""
    if not video_ids or connection.optional_table("video_transcript_chunks") is None:
        return ""
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    return str(transcript_chunk_search_query(connection, video_ids, query_embedding).explain_plan())


def analyze_transcript_chunk_search(
    connection: LanceConnection,
    query: str,
    video_ids: list[str],
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> str:
    """Return LanceDB's analyzed execution plan for a transcript chunk search."""
    if not video_ids or connection.optional_table("video_transcript_chunks") is None:
        return ""
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    return str(transcript_chunk_search_query(connection, video_ids, query_embedding).analyze_plan())


def transcript_chunk_search_query(
    connection: LanceConnection,
    video_ids: list[str] | None,
    query_embedding: list[float],
    *,
    limit: int = MAX_TABLE_SCAN_ROWS,
    where_sql: str | None = None,
) -> Any:
    """Build the filtered LanceDB transcript chunk query before scoring or column projection."""
    query = connection.table("video_transcript_chunks").search(query_embedding)
    filters: list[str] = []
    if video_ids is not None:
        filters.append(f"video_id IN ({filter_in_values(video_ids)})")
    if where_sql:
        filters.append(where_sql)
    if filters:
        query = query.where(" AND ".join(f"({filter_sql})" for filter_sql in filters))
    return query.limit(limit)


def list_video_ids_for_embedding_index(
    connection: LanceConnection,
    *,
    require_transcript: bool = True,
) -> list[str]:
    """List stored video IDs eligible for embedding jobs, newest first."""
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
    """Index the combined video title, channel, and transcript text for stored-video semantic search."""
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


def index_video_title_embedding(
    connection: LanceConnection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any] | None:
    """Index one stored title for title-prefilter semantic search."""
    video = find_video_row(connection, youtube_video_id)
    if video is None:
        return None

    row = video_payload_row(connection, video)
    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        return None

    source_text = title.strip()
    content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    existing_row = next(
        (row for row in connection.rows("video_title_embeddings") if row["video_id"] == youtube_video_id),
        None,
    )
    if existing_row is not None and existing_row["embedding_content_hash"] == content_hash and existing_row["embedding"] is not None:
        if existing_row.get("title") != source_text:
            connection.upsert_rows(
                "video_title_embeddings",
                "video_id",
                [{**existing_row, "title": source_text}],
            )
            connection.ensure_video_title_text_index(refresh=True)
        return {
            "video_id": youtube_video_id,
            "content_hash": content_hash,
            "dimensions": len(existing_row["embedding"]),
        }

    embedding = embed_text(source_text) if embed_text is not None else embed_text_with_gemini(source_text, config=embedding_config)
    if not embedding:
        raise EmbeddingSearchError("Embedding request returned an empty vector")
    connection.ensure_video_title_embedding_table(len(embedding))
    connection.upsert_rows(
        "video_title_embeddings",
        "video_id",
        [
            {
                "video_id": youtube_video_id,
                "title": source_text,
                "embedding_content_hash": content_hash,
                "embedding": [float(value) for value in embedding],
            }
        ],
    )
    connection.ensure_video_title_text_index(refresh=True)
    return {
        "video_id": youtube_video_id,
        "content_hash": content_hash,
        "dimensions": len(embedding),
    }


def index_video_title_embeddings(
    connection: LanceConnection,
    youtube_video_ids: list[str],
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[EmbeddingIndexResult]:
    """Batch-index stored titles while reusing unchanged embeddings and reporting per-video status."""
    existing_rows = {row["video_id"]: row for row in connection.rows("video_title_embeddings") if row.get("video_id") is not None}
    jobs: list[tuple[str, str, str]] = []
    results: list[EmbeddingIndexResult] = []
    for youtube_video_id in youtube_video_ids:
        video = find_video_row(connection, youtube_video_id)
        if video is None:
            results.append(
                {
                    "status": "skipped",
                    "video_id": youtube_video_id,
                    "detail": "Video is not stored",
                }
            )
            continue
        row = video_payload_row(connection, video)
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            results.append(
                {
                    "status": "skipped",
                    "video_id": youtube_video_id,
                    "detail": "Video has no stored title",
                }
            )
            continue
        source_text = title.strip()
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        existing_row = existing_rows.get(youtube_video_id)
        if existing_row is not None and existing_row["embedding_content_hash"] == content_hash and existing_row["embedding"] is not None:
            results.append(
                {
                    "status": "indexed",
                    "video_id": youtube_video_id,
                    "dimensions": len(existing_row["embedding"]),
                }
            )
            continue
        jobs.append(
            (
                youtube_video_id,
                source_text,
                content_hash,
            )
        )

    if not jobs:
        return results

    def embed_job(job: tuple[str, str, str]) -> dict[str, Any]:
        """Embed one stored title and carry enough metadata for idempotent upsert."""
        video_id, source_text, content_hash = job
        embedding = embed_text(source_text) if embed_text is not None else embed_text_with_gemini(source_text, config=embedding_config)
        if not embedding:
            raise EmbeddingSearchError("Embedding request returned an empty vector")
        return {
            "video_id": video_id,
            "title": source_text,
            "embedding_content_hash": content_hash,
            "embedding": [float(value) for value in embedding],
            "dimensions": len(embedding),
        }

    indexed_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=embedding_concurrency(len(jobs))) as executor:
        futures = {executor.submit(embed_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                results.append(
                    {
                        "status": "failed",
                        "video_id": job[0],
                        "error": str(error),
                    }
                )
                continue
            indexed_rows.append({key: value for key, value in result.items() if key != "dimensions"})
            results.append(
                {
                    "status": "indexed",
                    "video_id": job[0],
                    "dimensions": result["dimensions"],
                }
            )

    if indexed_rows:
        connection.ensure_video_title_embedding_table(len(indexed_rows[0]["embedding"]))
        connection.upsert_rows(
            "video_title_embeddings",
            "video_id",
            indexed_rows,
        )
        connection.ensure_video_title_text_index(refresh=True)
    return results


def embedding_source_text(row: Mapping[str, Any]) -> str:
    """Build the video-level embedding source text from stable searchable fields."""
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
    """Call the configured Gemini embedding endpoint and return the numeric vector."""
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
    """POST one embedding request after enforcing the configured base URL boundary."""
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
