"""Short-lived LanceDB staging for fresh external search candidates."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
from typing import Any

from lancedb.rerankers import RRFReranker

from ..model_config import EmbeddingConfig, embedding_concurrency
from ..schemas import EmbeddingIndexResult
from .connection import LanceConnection, filter_literal
from .semantic import EmbeddingSearchError, EmbedText, embed_text_with_gemini, rows_with_similarity


def store_search_candidates(
    connection: LanceConnection,
    session_id: str,
    query: str,
    candidates: list[dict[str, Any]],
) -> None:
    rows = [
        search_candidate_row(
            session_id,
            query,
            candidate,
            source_rank,
        )
        for source_rank, candidate in enumerate(candidates)
        if candidate.get("source") != "db" and candidate_video_id(candidate) is not None
    ]
    connection.upsert_rows(
        "search_candidates",
        [
            "session_id",
            "youtube_video_id",
        ],
        rows,
    )


def index_search_candidate_titles(
    connection: LanceConnection,
    session_id: str,
    *,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> list[EmbeddingIndexResult]:
    candidates = connection.rows(
        "search_candidates",
        where_sql=f"session_id = {filter_literal(session_id)}",
    )
    existing_rows = {
        row["video_id"]: row
        for row in connection.rows(
            "search_candidate_title_embeddings",
            where_sql=f"session_id = {filter_literal(session_id)}",
        )
        if row.get("video_id") is not None
    }
    jobs: list[tuple[str, int, str, str]] = []
    results: list[EmbeddingIndexResult] = []
    for candidate in candidates:
        video_id = candidate.get("youtube_video_id")
        title = candidate.get("title")
        if not isinstance(video_id, str) or not isinstance(title, str) or not title.strip():
            continue
        source_text = title.strip()
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        existing_row = existing_rows.get(video_id)
        if existing_row is not None and existing_row["embedding_content_hash"] == content_hash and existing_row["embedding"] is not None:
            results.append(
                {
                    "status": "indexed",
                    "video_id": video_id,
                    "dimensions": len(existing_row["embedding"]),
                }
            )
            continue
        jobs.append(
            (
                video_id,
                int(candidate.get("source_rank") or 0),
                source_text,
                content_hash,
            )
        )

    if not jobs:
        return results

    def embed_job(job: tuple[str, int, str, str]) -> dict[str, Any]:
        video_id, source_rank, source_text, content_hash = job
        embedding = embed_text(source_text) if embed_text is not None else embed_text_with_gemini(source_text, config=embedding_config)
        if not embedding:
            raise EmbeddingSearchError("Embedding request returned an empty vector")
        return {
            "session_id": session_id,
            "video_id": video_id,
            "source_rank": source_rank,
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
        connection.ensure_search_candidate_title_embedding_table(len(indexed_rows[0]["embedding"]))
        connection.upsert_rows(
            "search_candidate_title_embeddings",
            [
                "session_id",
                "video_id",
            ],
            indexed_rows,
        )
        connection.ensure_search_candidate_title_text_index(refresh=True)
    return results


def search_candidate_title_scores(
    connection: LanceConnection,
    session_id: str,
    query: str,
    *,
    limit: int,
    min_similarity: float | None = None,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, float]:
    if connection.optional_table("search_candidate_title_embeddings") is None:
        return {}
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    connection.ensure_search_candidate_title_text_index()
    rows = (
        connection.table("search_candidate_title_embeddings")
        .search(query_type="hybrid")
        .vector(query_embedding)
        .text(query)
        .where(f"session_id = {filter_literal(session_id)}")
        .rerank(RRFReranker())
        .limit(limit)
        .to_list()
    )
    rows = rows_with_similarity(rows, query_embedding)
    if min_similarity is not None:
        rows = [row for row in rows if row["similarity"] >= min_similarity]
    return {row["video_id"]: float(row["similarity"]) for row in rows[:limit]}


def search_candidate_row(
    session_id: str,
    query: str,
    candidate: Mapping[str, Any],
    source_rank: int,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "query": query,
        "source": str(candidate.get("source") or ""),
        "source_rank": source_rank,
        "youtube_video_id": candidate_video_id(candidate),
        "channel_id": text_value(candidate.get("channel_id")),
        "channel_handle": text_value(candidate.get("channel_handle")),
        "title": text_value(candidate.get("title")),
        "upload_date": text_value(candidate.get("upload_date")),
        "duration": float_value(candidate.get("duration")),
        "view_count": int_value(candidate.get("view_count")),
        "like_count": int_value(candidate.get("like_count")),
        "metadata_fetched_at": text_value(candidate.get("metadata_fetched_at")),
        "created_at": datetime.now(UTC).isoformat(),
    }


def candidate_video_id(candidate: Mapping[str, Any]) -> str | None:
    for key in (
        "id",
        "youtube_video_id",
        "video_id",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def text_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, int | float) else None
