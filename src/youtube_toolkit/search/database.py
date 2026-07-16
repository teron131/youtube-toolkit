"""DB-essential search sources enforce stored filters, ranking, and result projection."""

from __future__ import annotations

from typing import Any

from ..storage import get_connection, list_blacklist_channel_identities, search_stored_videos, search_transcript_chunk_scores, search_transcript_chunks
from ..storage.connection import LanceConnection, filter_in_values
from ..storage.search import stored_video_rows
from .filters import chunk_result, db_chunk_filter, db_video_filter, passes_filters, rank_by_value, stored_candidate, video_result
from .request import SOURCE_OVERSAMPLE_FACTOR, SearchRequest, SearchSort


def db_candidates(
    request: SearchRequest,
    *,
    limit: int,
    sort: SearchSort,
) -> list[dict[str, Any]]:
    """Load stored-video candidates using the DB lookup path and search ranking policy."""
    with get_connection() as connection:
        blacklisted_channels = tuple(list_blacklist_channel_identities(connection))
        payloads = search_stored_videos(
            connection,
            request.query,
            limit=limit,
            sort="popularity" if sort == "popularity" else "upload_date",
            where_sql=db_video_filter(
                request,
                blacklisted_channels,
            ),
        )
    return [
        {
            **stored_candidate(payload),
            "source": "db",
        }
        for payload in payloads
    ]


def search_db_chunks(
    request: SearchRequest,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Search stored transcript chunks directly while enforcing DB-backed filters and blacklists."""
    with get_connection() as connection:
        blacklisted_channels = tuple(list_blacklist_channel_identities(connection))
        rows = search_transcript_chunks(
            connection,
            request.query,
            None,
            limit=limit * SOURCE_OVERSAMPLE_FACTOR,
            where_sql=db_chunk_filter(
                request,
                blacklisted_channels,
            ),
            columns=["video_id", "chunk_index", "text"],
            min_similarity=request.result_min_similarity,
        )
        videos = chunk_videos(connection, rows)
    results: list[dict[str, Any]] = []
    for row in rows:
        video = videos.get(row["video_id"])
        if video is None:
            continue
        candidate = {**stored_candidate(video), "source": "db"}
        if not passes_filters(
            candidate,
            request,
            blacklisted_channels,
        ):
            continue
        results.append(chunk_result(row, video, warnings=[]))
        if len(results) == limit:
            break
    return results


def search_db_videos(
    request: SearchRequest,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Search stored transcripts semantically and return video-shaped DB results."""
    with get_connection() as connection:
        blacklisted_channels = tuple(list_blacklist_channel_identities(connection))
        scores = search_transcript_chunk_scores(
            connection,
            request.query,
            None,
            where_sql=db_chunk_filter(
                request,
                blacklisted_channels,
            ),
            min_similarity=request.result_min_similarity,
        )
        if not scores:
            return []
        candidates = scored_db_candidates(connection, scores)
    candidates = [
        candidate
        for candidate in candidates
        if passes_filters(
            candidate,
            request,
            blacklisted_channels,
        )
    ]
    candidates = rank_by_value(
        candidates,
        value=lambda candidate: candidate.get("score"),
    )
    return [video_result(candidate, warnings_by_video_id={}) for candidate in candidates[:limit]]


def scored_db_candidates(
    connection: LanceConnection,
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Attach semantic transcript scores to stored video candidates."""
    rows = stored_video_rows(
        connection,
        where_sql=f"youtube_video_id IN ({filter_in_values(list(scores))})",
    )
    return [
        {
            **stored_candidate(row),
            "source": "db",
            "transcript": row["transcript"],
            "score": scores[row["youtube_video_id"]],
        }
        for row in rows
        if row["youtube_video_id"] in scores
    ]


def chunk_videos(
    connection: LanceConnection,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load stored video metadata for semantic chunk rows in one filtered scan."""
    video_ids = sorted({row["video_id"] for row in rows})
    if not video_ids:
        return {}
    return {
        row["youtube_video_id"]: row
        for row in stored_video_rows(
            connection,
            where_sql=f"youtube_video_id IN ({filter_in_values(video_ids)})",
        )
    }


def db_semantic_candidates(request: SearchRequest) -> list[dict[str, Any]]:
    """Return stored-video candidates seeded by transcript semantic scores."""
    with get_connection() as connection:
        blacklisted_channels = tuple(list_blacklist_channel_identities(connection))
        scores = search_transcript_chunk_scores(
            connection,
            request.query,
            None,
            where_sql=db_chunk_filter(
                request,
                blacklisted_channels,
            ),
            min_similarity=request.result_min_similarity,
        )
        if not scores:
            return []
        return scored_db_candidates(connection, scores)
