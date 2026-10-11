"""Search workflow orchestration for channel scope, mode, and return-type choices.

Candidate videos come from YouTube search, or from channel listings when the request names channels.
Native mode stays metadata-first and returns video objects after local filters, ranking, and limits.
Relevance order is delegated to YouTube: each query's results keep YouTube's ranking, and several queries are interleaved rather than rescored.
Semantic mode runs title, transcript, chunk-index, and scoring stages, then returns chunks by default.
Full transcript output remains an explicit transcript command rather than a search side effect.
"""

from __future__ import annotations

from typing import Any

from ..config import MAX_CONCURRENT_METADATA_FETCHES
from ..execution import collect_concurrently
from ..youtube import YtDlpClient
from .discovery import channel_listings, youtube_candidates
from .preprocessing import (
    ChunkResult,
    VideoResult,
    candidate_id,
    dedupe_candidates,
    filter_candidates,
    normalized_upload_date,
    popularity,
    rank_by_value,
    video_result,
)
from .request import ResolvedRequest, SearchRequest, resolve_request

YOUTUBE_OVERSAMPLE_FACTOR = 3
DATE_FILTER_OVERSAMPLE_FACTOR = 12


def search_videos(request: SearchRequest) -> list[VideoResult] | list[ChunkResult]:
    """Run the public search workflow across the requested channel scope, filters, and result shapes."""
    resolved = resolve_request(request)

    candidates = collect_candidates(resolved)
    candidates = dedupe_candidates(candidates)
    candidates = filter_candidates(candidates, resolved)
    if request.mode == "semantic":
        from .semantic import search_transcripts

        return search_transcripts(resolved, candidates)
    if resolved.sort == "recency":
        candidates = rank_by_value(
            candidates,
            value=lambda candidate: candidate.get("upload_date"),
        )
    elif resolved.sort == "popularity":
        candidates = rank_by_value(
            candidates,
            value=popularity,
        )

    return [video_result(candidate) for candidate in candidates[: request.limit]]


def collect_candidates(resolved: ResolvedRequest) -> list[dict[str, Any]]:
    """Collect candidates from the requested channels, or from YouTube search when none are named, hydrated with any metadata the filters or sort need.

    Collection oversamples more under an upload-date filter because it discards candidates afterwards.
    """
    request = resolved.request
    fields = required_fields(resolved)
    date_filtered = resolved.upload_cutoff is not None
    date_fetch_limit = request.limit * DATE_FILTER_OVERSAMPLE_FACTOR
    if not request.channels:
        entries = youtube_candidates(
            resolved,
            fetch_limit=date_fetch_limit
            if date_filtered
            else request.limit * YOUTUBE_OVERSAMPLE_FACTOR,
        )
        return hydrate_candidates(entries, resolved, fields=fields)
    candidates: list[dict[str, Any]] = []
    for listing in channel_listings(
        request,
        limit=date_fetch_limit if date_filtered else request.limit,
    ):
        candidates.extend(
            hydrate_candidates(listing, resolved, fields=fields, newest_first=True)
        )
    return candidates


def required_fields(resolved: ResolvedRequest) -> frozenset[str]:
    """Name the candidate fields that active filters or native result ordering compare."""
    request = resolved.request
    native_sort = resolved.sort if request.mode == "native" else None
    fields: set[str] = set()
    if request.min_views is not None or native_sort == "popularity":
        fields.add("view_count")
    if (
        resolved.min_duration_seconds is not None
        or resolved.max_duration_seconds is not None
    ):
        fields.add("duration")
    if resolved.upload_cutoff is not None or native_sort == "recency":
        fields.add("upload_date")
    return frozenset(fields)


def hydrate_candidates(
    candidates: list[dict[str, Any]],
    resolved: ResolvedRequest,
    *,
    fields: frozenset[str],
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    """Fetch full metadata for candidates that pass the known filters but lack a field the filters or sort need.

    Flat listings omit fields (channel listings carry no view counts or upload dates), so unknown values pass a lenient prefilter and only its survivors are fetched, concurrently and in order.
    A newest-first listing under an upload cutoff is fetched in batches that stop at the first video older than the cutoff, because every later video is older too.
    Failed fetches leave candidates unhydrated for the strict filter to judge.
    Hydrated candidates also keep the raw response under "metadata", so later stages of this search reuse it instead of requesting it again.
    """
    survivors = filter_candidates(candidates, resolved, allow_missing=True)
    stop_before = resolved.upload_cutoff if newest_first else None
    batch_size = (
        MAX_CONCURRENT_METADATA_FETCHES if stop_before is not None else len(survivors)
    )
    client = YtDlpClient()
    hydrated: list[dict[str, Any]] = []
    for start in range(0, len(survivors), max(batch_size, 1)):
        batch = survivors[start : start + batch_size]
        missing = {
            video_id: video_id
            for candidate in batch
            if any(candidate.get(field) is None for field in fields)
            and (video_id := candidate_id(candidate)) is not None
        }
        metadata_by_id, _failures = collect_concurrently(
            missing,
            client.metadata,
            max_workers=MAX_CONCURRENT_METADATA_FETCHES,
        )
        for candidate in batch:
            video_id = candidate_id(candidate)
            metadata = metadata_by_id.get(video_id) if video_id is not None else None
            hydrated.append(
                candidate
                if video_id is None or metadata is None
                else {
                    **candidate,
                    **metadata,
                    "id": video_id,
                    "upload_date": normalized_upload_date(metadata.get("upload_date")),
                    "source": candidate.get("source"),
                    "metadata": metadata,
                }
            )
        if stop_before is not None and any(
            (upload_date := normalized_upload_date(candidate.get("upload_date")))
            is not None
            and upload_date < stop_before
            for candidate in hydrated[start:]
        ):
            break
    return hydrated
