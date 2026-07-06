"""Stored-video lookup across IDs, URLs, keywords, and optional semantic ranking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from ..identity import VideoRef, extract_video_id
from ..model_config import EmbeddingConfig
from .connection import MAX_TABLE_SCAN_ROWS, LanceConnection
from .payloads import stored_video_payload
from .semantic import embed_text_with_gemini
from .videos import video_payload_row

EmbedText = Callable[[str], list[float]]
SearchSort = Literal["upload_date", "relevance"]
SEARCH_SORTS = {"upload_date", "relevance"}


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
) -> list[dict[str, Any]]:
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
    rows = stored_video_rows(connection)
    if video_id is not None:
        matching_rows = [row for row in rows if row["youtube_video_id"] == video_id and has_required_transcript(row, require_transcript)]
        return [stored_video_payload(row) for row in sort_video_rows_by_upload_date(matching_rows)[:limit]]

    normalized_keyword = normalized_query.casefold()
    matching_rows = [row for row in rows if has_required_transcript(row, require_transcript) and video_row_matches_keyword(row, normalized_keyword)]
    keyword_payloads = [stored_video_payload(row) for row in sort_video_rows_by_upload_date(matching_rows)[:limit]]
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
        sort=resolved_sort,
        limit=limit,
    )


def get_stored_video(
    connection: LanceConnection,
    video_ref: VideoRef,
    *,
    require_transcript: bool = False,
) -> dict[str, Any] | None:
    """Return one stored video payload by ID or URL."""
    video_id = extract_video_id(video_ref)
    if video_id is None:
        return None
    row = next(
        (row for row in stored_video_rows(connection) if row["youtube_video_id"] == video_id and has_required_transcript(row, require_transcript)),
        None,
    )
    if row is None:
        return None
    return stored_video_payload(row)


def stored_video_rows(connection: LanceConnection) -> list[dict[str, Any]]:
    return [video_payload_row(connection, row) for row in connection.rows("videos")]


def has_required_transcript(
    row: dict[str, Any],
    require_transcript: bool,
) -> bool:
    return not require_transcript or bool(row["transcript"])


def video_row_matches_keyword(
    row: dict[str, Any],
    normalized_keyword: str,
) -> bool:
    return any(
        normalized_keyword in str(value).casefold()
        for value in (
            row["youtube_video_id"],
            row["title"],
            row["upload_date"],
            row["transcript"],
            row["channel_id"],
            row["channel_handle"],
        )
        if value not in (None, "")
    )


def search_video_payloads_by_semantic(
    connection: LanceConnection,
    query: str,
    *,
    require_transcript: bool,
    embed_text: EmbedText | None,
    embedding_config: EmbeddingConfig | None,
) -> list[dict[str, Any]]:
    if connection.optional_table("video_embeddings") is None:
        return []
    query_embedding = embed_text(query) if embed_text is not None else embed_text_with_gemini(query, config=embedding_config)
    rows = connection.table("video_embeddings").search(query_embedding, vector_column_name="embedding").metric("cosine").limit(MAX_TABLE_SCAN_ROWS).to_list()
    videos_by_id = {row["youtube_video_id"]: row for row in stored_video_rows(connection)}
    semantic_payloads: list[dict[str, Any]] = []
    for row in rows:
        video = videos_by_id.get(row["video_id"])
        if video is None or (require_transcript and not video["transcript"]):
            continue
        payload = stored_video_payload(video)
        payload["semantic_score"] = 1.0 - float(row["_distance"])
        semantic_payloads.append(payload)
    return semantic_payloads


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
    else:
        sorted_payloads = payloads
    return sorted_payloads[:limit]


def sort_video_rows_by_upload_date(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["upload_date"] or "",
            row["youtube_video_id"],
        ),
        reverse=True,
    )
