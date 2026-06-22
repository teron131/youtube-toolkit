"""Stored-video lookup across IDs, URLs, keywords, and optional semantic ranking."""

from __future__ import annotations

from collections.abc import Callable
import sqlite3
from typing import Any, Literal

from ..url import extract_video_id
from .semantic import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    cosine_similarity,
    embed_text_with_gemini,
    embedding_from_json,
)

EmbedText = Callable[[str], list[float]]
SearchSort = Literal["views", "upload_date", "relevance"]
SEARCH_SORTS = {"views", "upload_date", "relevance"}


def search_stored_videos(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    require_transcript: bool = False,
    semantic: bool = False,
    embed_text: EmbedText | None = None,
    sort: SearchSort = "views",
) -> list[dict[str, Any]]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("search query must be non-empty")
    if limit <= 0:
        raise ValueError("search limit must be greater than 0")
    if sort not in SEARCH_SORTS:
        raise ValueError(f"unsupported search sort: {sort}")
    if sort == "relevance" and not semantic:
        raise ValueError("relevance sort requires semantic=True")

    video_id = extract_video_id(normalized_query)
    if video_id is not None:
        rows = search_video_rows_by_id(
            connection,
            video_id,
            limit=limit,
            require_transcript=require_transcript,
            sort=sort,
        )
        return [stored_video_payload(row) for row in rows]

    keyword_payloads = [
        stored_video_payload(row)
        for row in search_video_rows_by_keyword(
            connection,
            normalized_query,
            limit=limit,
            require_transcript=require_transcript,
            sort=sort,
        )
    ]
    if not semantic:
        return keyword_payloads

    semantic_payloads = search_video_payloads_by_semantic(
        connection,
        normalized_query,
        require_transcript=require_transcript,
        embed_text=embed_text,
    )
    return rank_search_payloads(
        merge_search_payloads(keyword_payloads, semantic_payloads),
        sort=sort,
        limit=limit,
    )


def search_video_rows_by_id(
    connection: sqlite3.Connection,
    youtube_video_id: str,
    *,
    limit: int,
    require_transcript: bool,
    sort: SearchSort,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            videos.id,
            videos.youtube_video_id,
            videos.title,
            videos.upload_date,
            videos.view_count,
            videos.transcript,
            channels.youtube_channel_id,
            channels.channel_handle
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.id = videos.channel_id
        WHERE videos.youtube_video_id = ?
            AND (
                ? = 0
                OR (videos.transcript IS NOT NULL AND videos.transcript != '')
            )
        ORDER BY
            CASE WHEN ? = 'views' THEN videos.view_count END DESC,
            videos.upload_date DESC,
            videos.id DESC
        LIMIT ?
        """,
        (youtube_video_id, int(require_transcript), sort, limit),
    ).fetchall()


def search_video_rows_by_keyword(
    connection: sqlite3.Connection,
    keyword: str,
    *,
    limit: int,
    require_transcript: bool,
    sort: SearchSort,
) -> list[sqlite3.Row]:
    like_pattern = f"%{escape_like(keyword)}%"
    return connection.execute(
        """
        SELECT
            videos.id,
            videos.youtube_video_id,
            videos.title,
            videos.upload_date,
            videos.view_count,
            videos.transcript,
            channels.youtube_channel_id,
            channels.channel_handle
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.id = videos.channel_id
        WHERE (
            videos.youtube_video_id LIKE ? ESCAPE '\\'
            OR COALESCE(videos.title, '') LIKE ? ESCAPE '\\'
            OR COALESCE(videos.upload_date, '') LIKE ? ESCAPE '\\'
            OR CAST(COALESCE(videos.view_count, '') AS TEXT) LIKE ? ESCAPE '\\'
            OR COALESCE(videos.transcript, '') LIKE ? ESCAPE '\\'
            OR COALESCE(channels.youtube_channel_id, '') LIKE ? ESCAPE '\\'
            OR COALESCE(channels.channel_handle, '') LIKE ? ESCAPE '\\'
        )
            AND (
                ? = 0
                OR (videos.transcript IS NOT NULL AND videos.transcript != '')
            )
        ORDER BY
            CASE WHEN ? = 'views' THEN videos.view_count END DESC,
            videos.upload_date DESC,
            videos.id DESC
        LIMIT ?
        """,
        (
            like_pattern,
            like_pattern,
            like_pattern,
            like_pattern,
            like_pattern,
            like_pattern,
            like_pattern,
            int(require_transcript),
            sort,
            limit,
        ),
    ).fetchall()


def search_video_payloads_by_semantic(
    connection: sqlite3.Connection,
    query: str,
    *,
    require_transcript: bool,
    embed_text: EmbedText | None,
) -> list[dict[str, Any]]:
    indexed_rows = list_semantic_video_rows(
        connection,
        require_transcript=require_transcript,
    )
    if not indexed_rows:
        return []

    query_embedding = (embed_text or embed_text_with_gemini)(query)
    scored_payloads: list[tuple[float, dict[str, Any]]] = []
    for row in indexed_rows:
        score = cosine_similarity(
            query_embedding,
            embedding_from_json(row["embedding_json"]),
        )
        payload = stored_video_payload(row)
        payload["semantic_score"] = score
        scored_payloads.append((score, payload))

    scored_payloads.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in scored_payloads]


def list_semantic_video_rows(
    connection: sqlite3.Connection,
    *,
    require_transcript: bool,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            videos.id,
            videos.youtube_video_id,
            videos.title,
            videos.upload_date,
            videos.view_count,
            videos.transcript,
            channels.youtube_channel_id,
            channels.channel_handle,
            video_embeddings.embedding_json
        FROM video_embeddings AS video_embeddings
        JOIN videos AS videos ON videos.id = video_embeddings.video_id
        LEFT JOIN channels AS channels ON channels.id = videos.channel_id
        WHERE video_embeddings.provider = ?
            AND video_embeddings.model = ?
            AND (
                ? = 0
                OR (videos.transcript IS NOT NULL AND videos.transcript != '')
            )
        """,
        (
            EMBEDDING_PROVIDER,
            DEFAULT_EMBEDDING_MODEL,
            int(require_transcript),
        ),
    ).fetchall()


def stored_video_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "video_id": row["youtube_video_id"],
        "database_row_id": int(row["id"]),
        "title": row["title"],
        "upload_date": row["upload_date"],
        "view_count": row["view_count"],
        "channel_id": row["youtube_channel_id"],
        "channel_handle": row["channel_handle"],
        "transcript": row["transcript"],
    }


def merge_search_payloads(
    keyword_payloads: list[dict[str, Any]],
    semantic_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    payload_by_video_id: dict[str, dict[str, Any]] = {}
    for payload in keyword_payloads:
        payload_by_video_id[payload["video_id"]] = payload
        merged.append(payload)

    for payload in semantic_payloads:
        existing_payload = payload_by_video_id.get(payload["video_id"])
        if existing_payload is not None:
            existing_payload["semantic_score"] = payload["semantic_score"]
            continue
        payload_by_video_id[payload["video_id"]] = payload
        merged.append(payload)
    return merged


def rank_search_payloads(
    payloads: list[dict[str, Any]],
    *,
    sort: SearchSort,
    limit: int,
) -> list[dict[str, Any]]:
    if sort == "views":
        sorted_payloads = sorted(
            payloads,
            key=lambda payload: (
                payload["view_count"] if isinstance(payload["view_count"], int) else -1,
                payload["upload_date"] or "",
                payload["database_row_id"],
            ),
            reverse=True,
        )
    elif sort == "upload_date":
        sorted_payloads = sorted(
            payloads,
            key=lambda payload: (
                payload["upload_date"] or "",
                payload["database_row_id"],
            ),
            reverse=True,
        )
    elif sort == "relevance":
        sorted_payloads = sorted(
            payloads,
            key=lambda payload: (
                payload.get("semantic_score", float("-inf")),
                payload["view_count"] if isinstance(payload["view_count"], int) else -1,
                payload["database_row_id"],
            ),
            reverse=True,
        )
    else:
        sorted_payloads = payloads
    return sorted_payloads[:limit]


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
