"""Stored-video lookup owns direct ID hits, keyword scans, optional semantic ranking, and public payload projection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from ..identity import VideoRef, extract_video_id
from ..model_config import EmbeddingConfig
from . import embeddings
from .connection import MAX_TABLE_SCAN_ROWS, LanceConnection, filter_contains_case_insensitive, filter_equal, filter_in_values
from .payloads import stored_video_payload

EmbedText = Callable[[str], list[float]]
SearchSort = Literal["upload_date", "popularity", "relevance"]
SEARCH_SORTS = {"upload_date", "popularity", "relevance"}
TRANSCRIPT_PRESENT_SQL = "transcript IS NOT NULL AND transcript != ''"
UPLOAD_DATE_ORDER = [
    ("upload_date", False),
    ("youtube_video_id", False),
]


def search_stored_videos(
    connection: LanceConnection,
    query: str,
    *,
    limit: int = 10,
    require_transcript: bool = False,
    semantic: bool = False,
    embed_text: EmbedText | None = None,
    embedding_config: EmbeddingConfig | None = None,
    sort: SearchSort | None = None,
    where_sql: str | None = None,
) -> list[dict[str, Any]]:
    """Search stored videos through native ID, substring, ordering, and vector queries."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("search query must be non-empty")
    if limit <= 0:
        raise ValueError("search limit must be greater than 0")
    resolved_sort = sort or ("relevance" if semantic else "upload_date")
    if resolved_sort not in SEARCH_SORTS:
        raise ValueError(f"unsupported search sort: {resolved_sort}")
    if resolved_sort == "relevance" and not semantic:
        raise ValueError("relevance sort requires semantic=True")

    video_id = extract_video_id(normalized_query)
    if video_id is not None:
        rows = stored_video_rows(
            connection,
            where_sql=combine_filters(
                video_lookup_filter(video_id, require_transcript=require_transcript),
                where_sql,
            ),
            limit=limit,
            order_by=UPLOAD_DATE_ORDER,
        )
        return [stored_video_payload(row) for row in rows]

    normalized_keyword = normalized_query.casefold()
    keyword_where_sql = combine_filters(
        stored_video_keyword_filter(
            connection,
            normalized_keyword,
        ),
        TRANSCRIPT_PRESENT_SQL if require_transcript else None,
        where_sql,
    )
    rows = stored_video_rows(
        connection,
        where_sql=keyword_where_sql,
        limit=MAX_TABLE_SCAN_ROWS if resolved_sort == "popularity" else limit,
        order_by=None if resolved_sort == "popularity" else UPLOAD_DATE_ORDER,
    )
    keyword_payloads = [stored_video_payload(row) for row in rows]
    if not semantic:
        if resolved_sort == "popularity":
            return rank_search_payloads(
                keyword_payloads,
                sort=resolved_sort,
                limit=limit,
            )
        return keyword_payloads

    semantic_payloads = search_video_payloads_by_semantic(
        connection,
        normalized_query,
        require_transcript=require_transcript,
        embed_text=embed_text,
        embedding_config=embedding_config,
        where_sql=where_sql,
        limit=(limit if resolved_sort == "relevance" and not require_transcript and where_sql is None else MAX_TABLE_SCAN_ROWS),
    )
    return rank_search_payloads(
        merge_search_payloads(keyword_payloads, semantic_payloads),
        sort=resolved_sort,
        limit=limit,
    )


def get_stored_video(
    connection: LanceConnection,
    video_ref: VideoRef,
    *,
    require_transcript: bool = False,
) -> dict[str, Any] | None:
    """Direct stored-video lookups accept the same ID and URL forms as public video refs."""
    video_id = extract_video_id(video_ref)
    if video_id is None:
        return None
    rows = stored_video_rows(
        connection,
        where_sql=video_lookup_filter(video_id, require_transcript=require_transcript),
        limit=1,
    )
    if not rows:
        return None
    return stored_video_payload(rows[0])


def stored_video_rows(
    connection: LanceConnection,
    *,
    where_sql: str | None = None,
    limit: int = MAX_TABLE_SCAN_ROWS,
    order_by: list[tuple[str, bool]] | None = None,
) -> list[dict[str, Any]]:
    """Return filtered video rows with channel handles attached through one batched lookup."""
    rows = connection.rows(
        "videos",
        where_sql=where_sql,
        limit=limit,
        order_by=order_by,
    )
    channel_ids = sorted({row["channel_id"] for row in rows if row.get("channel_id") is not None})
    if not channel_ids:
        return [{**row, "channel_handle": None} for row in rows]
    channels = connection.rows(
        "channels",
        where_sql=f"youtube_channel_id IN ({filter_in_values(channel_ids)})",
        columns=[
            "youtube_channel_id",
            "channel_handle",
        ],
    )
    handle_by_channel_id = {row["youtube_channel_id"]: row["channel_handle"] for row in channels}
    return [
        {
            **row,
            "channel_handle": handle_by_channel_id.get(row.get("channel_id")),
        }
        for row in rows
    ]


def video_lookup_filter(
    video_id: str,
    *,
    require_transcript: bool,
) -> str:
    """Exact-ID filters can require transcript text without turning lookup into a broader search."""
    clauses = [filter_equal("youtube_video_id", video_id)]
    if require_transcript:
        clauses.append(TRANSCRIPT_PRESENT_SQL)
    return " AND ".join(clauses)


def stored_video_keyword_filter(
    connection: LanceConnection,
    normalized_keyword: str,
) -> str:
    """Build a native substring predicate across stored video fields and channel handles."""
    clauses = [
        filter_contains_case_insensitive(column_name, normalized_keyword)
        for column_name in (
            "youtube_video_id",
            "title",
            "upload_date",
            "transcript",
            "channel_id",
        )
    ]
    matching_channels = connection.rows(
        "channels",
        where_sql=filter_contains_case_insensitive(
            "channel_handle",
            normalized_keyword,
        ),
        columns=["youtube_channel_id"],
    )
    matching_channel_ids = [row["youtube_channel_id"] for row in matching_channels if row.get("youtube_channel_id") is not None]
    if matching_channel_ids:
        clauses.append(f"channel_id IN ({filter_in_values(matching_channel_ids)})")
    return " OR ".join(f"({clause})" for clause in clauses)


def combine_filters(*filters: str | None) -> str | None:
    """Combine optional SQL filters without leaking grouping decisions to callers."""
    active_filters = [filter_sql for filter_sql in filters if filter_sql]
    if not active_filters:
        return None
    return " AND ".join(f"({filter_sql})" for filter_sql in active_filters)


def search_video_payloads_by_semantic(
    connection: LanceConnection,
    query: str,
    *,
    require_transcript: bool,
    embed_text: EmbedText | None,
    embedding_config: EmbeddingConfig | None,
    where_sql: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Search video-level embeddings and project matching videos into stored-video payloads."""
    if connection.optional_table("video_embeddings") is None:
        return []
    query_embedding = (
        embed_text(query)
        if embed_text is not None
        else embeddings.embed_text(
            query,
            config=embedding_config,
        )
    )
    vector_query: Any = connection.table("video_embeddings").search(
        query_embedding,
        vector_column_name="embedding",
    )
    rows = (
        vector_query.distance_type("cosine")
        .select(
            [
                "video_id",
                "_distance",
            ]
        )
        .limit(limit)
        .to_list()
    )
    video_ids = [video_id for row in rows if isinstance(video_id := row.get("video_id"), str)]
    if not video_ids:
        return []
    clauses = [f"youtube_video_id IN ({filter_in_values(video_ids)})"]
    if require_transcript:
        clauses.append(TRANSCRIPT_PRESENT_SQL)
    if where_sql:
        clauses.append(where_sql)
    videos_by_id = {
        row["youtube_video_id"]: row
        for row in stored_video_rows(
            connection,
            where_sql=" AND ".join(clauses),
        )
    }
    semantic_payloads: list[dict[str, Any]] = []
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str):
            continue
        video = videos_by_id.get(video_id)
        if video is None:
            continue
        distance = row.get("_distance")
        if not isinstance(distance, int | float):
            continue
        payload = stored_video_payload(video)
        payload["semantic_score"] = 1.0 - float(distance)
        semantic_payloads.append(payload)
    return semantic_payloads


def merge_search_payloads(
    keyword_payloads: list[dict[str, Any]],
    semantic_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge keyword and semantic stored-video hits while preserving keyword order."""
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
    """Apply the final stored-video sort policy after keyword and semantic merge."""
    if sort == "upload_date":
        sorted_payloads = sorted(
            payloads,
            key=lambda payload: (
                payload["upload_date"] or "",
                payload["video_id"],
            ),
            reverse=True,
        )
    elif sort == "relevance":
        sorted_payloads = sorted(
            payloads,
            key=lambda payload: (
                payload.get("semantic_score", float("-inf")),
                payload["video_id"],
            ),
            reverse=True,
        )
    elif sort == "popularity":
        sorted_payloads = sorted(
            payloads,
            key=lambda payload: (
                payload.get("view_count") or 0,
                payload.get("like_count") or 0,
            ),
            reverse=True,
        )
    else:
        sorted_payloads = payloads
    return sorted_payloads[:limit]
