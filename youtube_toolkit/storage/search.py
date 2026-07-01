"""Stored-video lookup across IDs, URLs, keywords, and optional semantic ranking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import turso

from ..identity import VideoRef, extract_video_id
from ..model_config import EmbeddingConfig
from .payloads import stored_video_payload
from .semantic import embed_text_with_gemini, encode_vector

EmbedText = Callable[[str], list[float]]
SearchSort = Literal["views", "upload_date", "relevance"]
SEARCH_SORTS = {"views", "upload_date", "relevance"}
STORED_VIDEO_RESULT_COLUMNS = "videos.*, channels.youtube_channel_id, channels.channel_handle"


def search_stored_videos(
    connection: turso.Connection,
    query: str,
    *,
    limit: int = 10,
    require_transcript: bool = False,
    semantic: bool = False,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
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
        embedding_config=embedding_config,
    )
    return rank_search_payloads(
        merge_search_payloads(keyword_payloads, semantic_payloads),
        sort=sort,
        limit=limit,
    )


def get_stored_video(
    connection: turso.Connection,
    video_ref: VideoRef,
    *,
    require_transcript: bool = False,
) -> dict[str, Any] | None:
    """Return one stored video payload by ID or URL."""
    video_id = extract_video_id(video_ref)
    if video_id is None:
        return None
    row = connection.execute(
        f"""
        SELECT {STORED_VIDEO_RESULT_COLUMNS}
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.id = videos.channel_id
        WHERE videos.youtube_video_id = ?
            AND (
                ? = 0
                OR (videos.transcript IS NOT NULL AND videos.transcript != '')
            )
        LIMIT 1
        """,
        (video_id, int(require_transcript)),
    ).fetchone()
    if row is None:
        return None
    return stored_video_payload(row)


def search_video_rows_by_id(
    connection: turso.Connection,
    youtube_video_id: str,
    *,
    limit: int,
    require_transcript: bool,
    sort: SearchSort,
) -> list[turso.Row]:
    return connection.execute(
        f"""
        SELECT {STORED_VIDEO_RESULT_COLUMNS}
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
    connection: turso.Connection,
    keyword: str,
    *,
    limit: int,
    require_transcript: bool,
    sort: SearchSort,
) -> list[turso.Row]:
    like_pattern = f"%{escape_like(keyword)}%"
    return connection.execute(
        f"""
        SELECT {STORED_VIDEO_RESULT_COLUMNS}
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
    connection: turso.Connection,
    query: str,
    *,
    require_transcript: bool,
    embed_text: EmbedText | None,
    embedding_config: EmbeddingConfig | None,
) -> list[dict[str, Any]]:
    if not has_semantic_video_rows(
        connection,
        require_transcript=require_transcript,
    ):
        return []
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    indexed_rows = list_semantic_video_rows(
        connection,
        query_embedding=query_embedding,
        require_transcript=require_transcript,
    )
    if not indexed_rows:
        return []

    semantic_payloads: list[dict[str, Any]] = []
    for row in indexed_rows:
        payload = stored_video_payload(row)
        payload["semantic_score"] = row["semantic_score"]
        semantic_payloads.append(payload)
    return semantic_payloads


def has_semantic_video_rows(
    connection: turso.Connection,
    *,
    require_transcript: bool,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM videos
        WHERE embedding IS NOT NULL
            AND (
                ? = 0
                OR (transcript IS NOT NULL AND transcript != '')
            )
        LIMIT 1
        """,
        (int(require_transcript),),
    ).fetchone()
    return row is not None


def list_semantic_video_rows(
    connection: turso.Connection,
    *,
    query_embedding: list[float],
    require_transcript: bool,
) -> list[turso.Row]:
    return connection.execute(
        f"""
        SELECT
            {STORED_VIDEO_RESULT_COLUMNS},
            1.0 - vector_distance_cos(videos.embedding, vector32(?)) AS semantic_score
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.id = videos.channel_id
        WHERE videos.embedding IS NOT NULL
            AND (
                ? = 0
                OR (videos.transcript IS NOT NULL AND videos.transcript != '')
            )
        ORDER BY semantic_score DESC
        """,
        (
            encode_vector(query_embedding),
            int(require_transcript),
        ),
    ).fetchall()


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
