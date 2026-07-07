"""Semantic title, transcript, and chunk search stages."""

from __future__ import annotations

from typing import Any

from ..storage import (
    auto_index_transcript_chunks,
    get_connection,
    index_search_candidate_titles,
    index_video_title_embeddings,
    list_blacklist_channel_identities,
    search_candidate_title_scores,
    search_transcript_chunk_scores,
    search_transcript_chunks,
    search_video_title_scores,
)
from ..storage.connection import LanceConnection, filter_in_values
from ..storage.search import stored_video_rows
from ..youtube.caption import fetch_transcript
from .filters import (
    candidate_id,
    candidate_ids,
    chunk_result,
    db_chunk_filter,
    passes_filters,
    rank_by_value,
    stored_candidate,
    video_result,
)
from .request import SOURCE_OVERSAMPLE_FACTOR, SearchRequest


def prefilter_by_title_semantics(
    request: SearchRequest,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    video_ids = candidate_ids(candidates)
    if not query.strip() or not video_ids:
        return candidates
    with get_connection() as connection:
        scores = title_semantic_scores(
            connection,
            request,
            query,
            candidates,
            session_id=session_id,
        )
    if not scores:
        return candidates

    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        video_id = candidate_id(candidate)
        score = scores.get(video_id) if video_id is not None else None
        if score is not None:
            scored_candidates.append({**candidate, "title_score": score})
    ranked_candidates = rank_by_value(
        scored_candidates,
        value=lambda candidate: candidate.get("title_score"),
    )
    threshold_candidates = [candidate for candidate in ranked_candidates if request.title_min_similarity is None or candidate["title_score"] >= request.title_min_similarity]
    if threshold_candidates:
        ranked_candidates = threshold_candidates
    return [{key: value for key, value in candidate.items() if key != "title_score"} for candidate in ranked_candidates[: title_prefilter_limit(request, len(video_ids))]]


def title_semantic_scores(
    connection: LanceConnection,
    request: SearchRequest,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    session_id: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    external_ids = candidate_ids([candidate for candidate in candidates if candidate.get("source") != "db"])
    if external_ids:
        index_search_candidate_titles(connection, session_id)
        scores.update(
            search_candidate_title_scores(
                connection,
                session_id,
                query,
                limit=title_prefilter_limit(request, len(external_ids)),
                min_similarity=request.title_min_similarity,
            )
        )

    db_ids = candidate_ids([candidate for candidate in candidates if candidate.get("source") == "db"])
    if db_ids:
        index_video_title_embeddings(connection, db_ids)
        scores.update(
            search_video_title_scores(
                connection,
                query,
                db_ids,
                limit=title_prefilter_limit(request, len(db_ids)),
                min_similarity=request.title_min_similarity,
            )
        )
    return scores


def title_prefilter_limit(
    request: SearchRequest,
    candidate_count: int,
) -> int:
    requested_limit = max(
        request.limit * request.title_prefilter_factor,
        request.title_prefilter_min,
    )
    return min(
        candidate_count,
        request.title_prefilter_max,
        requested_limit,
    )


def index_transcripts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        video_id = candidate_id(candidate)
        if video_id is None:
            continue
        if candidate.get("source") == "db":
            with get_connection() as connection:
                results.append(auto_index_transcript_chunks(connection, video_id))
            continue
        try:
            document = fetch_transcript(
                video_id,
                auto_index=False,
            )
            with get_connection() as connection:
                results.append(
                    auto_index_transcript_chunks(
                        connection,
                        document.metadata.video_id,
                    )
                )
        except Exception as error:
            results.append(
                {
                    "status": "failed",
                    "video_id": video_id,
                    "error": str(error),
                }
            )
    return results


def ensure_semantic_ready(
    candidates: list[dict[str, Any]],
    index_results: list[dict[str, Any]],
    ready_ids: set[str] | None = None,
) -> None:
    video_ids = candidate_ids(candidates)
    if not video_ids:
        return
    ready_ids = ready_ids if ready_ids is not None else semantic_ready_ids(candidates, index_results)
    if ready_ids:
        return
    raise RuntimeError(f"semantic search could not prepare transcript chunks{index_failure_detail(index_results)}")


def semantic_ready_ids(
    candidates: list[dict[str, Any]],
    index_results: list[dict[str, Any]],
) -> set[str]:
    video_ids = candidate_ids(candidates)
    indexed_ids = {str(result["video_id"]) for result in index_results if result.get("status") == "indexed" and result.get("video_id")}
    with get_connection() as connection:
        requested_ids = set(video_ids)
        stored_ids = {row["video_id"] for row in connection.rows("video_transcript_chunks") if row.get("video_id") in requested_ids and row.get("embedding") is not None}
    return indexed_ids | stored_ids


def index_failure_detail(index_results: list[dict[str, Any]]) -> str:
    if not index_results:
        return ""
    details: list[str] = []
    for result in index_results[:3]:
        video_id = result.get("video_id", "unknown")
        detail = result.get("error") or result.get("detail") or result.get("status", "")
        details.append(f"{video_id}: {detail}")
    return ": " + "; ".join(details)


def index_warnings(
    index_results: list[dict[str, Any]],
    ready_ids: set[str],
) -> dict[str, list[str]]:
    warnings: dict[str, list[str]] = {}
    for result in index_results:
        video_id = result.get("video_id")
        if not isinstance(video_id, str) or video_id in ready_ids:
            continue
        if result.get("status") not in {"failed", "skipped"}:
            continue
        detail = result.get("error") or result.get("detail") or result.get("status", "")
        warnings.setdefault(video_id, []).append(f"{video_id}: {detail}")
    return warnings


def rank_by_transcript(
    request: SearchRequest,
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    video_ids = candidate_ids(candidates)
    if not video_ids:
        return candidates
    with get_connection() as connection:
        scores = search_transcript_chunk_scores(
            connection,
            query,
            video_ids,
            min_similarity=request.result_min_similarity,
        )
    if not scores:
        return []
    scored_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        video_id = candidate_id(candidate)
        score = scores.get(video_id) if video_id is not None else None
        if score is not None:
            scored_candidates.append({**candidate, "score": score})
    return rank_by_value(
        scored_candidates,
        value=lambda candidate: candidate.get("score"),
    )


def search_chunks(
    request: SearchRequest,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    warnings_by_video_id: dict[str, list[str]],
) -> list[dict[str, Any]]:
    video_ids = candidate_ids(candidates)
    if not video_ids:
        return []
    with get_connection() as connection:
        rows = search_transcript_chunks(
            connection,
            query,
            video_ids,
            limit=request.limit,
            min_similarity=request.result_min_similarity,
        )
        videos = chunk_videos(connection, rows)
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            chunk_result(
                row,
                videos.get(row["video_id"]),
                warnings=warnings_by_video_id.get(row["video_id"]),
            )
        )
    return results


def search_db_chunks(
    request: SearchRequest,
    *,
    limit: int,
) -> list[dict[str, Any]]:
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
    rows = stored_video_rows(
        connection,
        where_sql=f"youtube_video_id IN ({filter_in_values(list(scores))})",
    )
    return [
        {
            **stored_candidate(row),
            "source": "db",
            "score": scores[row["youtube_video_id"]],
        }
        for row in rows
        if row["youtube_video_id"] in scores
    ]


def chunk_videos(
    connection: LanceConnection,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
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
