"""Search workflow: collect candidates from YouTube search or the named channels, hydrate and filter them, then rank natively or hand them to semantic search.

Native mode stays metadata-first: relevance keeps YouTube's order, while recency and popularity sort locally.
Full transcript output remains an explicit transcript command rather than a search side effect.
"""

from __future__ import annotations

from typing import Any

from ..execution import collect_concurrently
from ..schemas import normalize_upload_date
from ..youtube import MAX_CONCURRENT_METADATA_FETCHES, YtDlpClient
from .candidates import (
    ChunkResult,
    VideoResult,
    dedupe_candidates,
    filter_candidates,
    popularity,
    rank_by_value,
    video_result,
)
from .discovery import channel_listings, youtube_candidates
from .request import ResolvedRequest, SearchRequest, resolve_request

YOUTUBE_OVERSAMPLE_FACTOR = 3
DATE_FILTER_OVERSAMPLE_FACTOR = 12


def search_videos(request: SearchRequest) -> list[VideoResult] | list[ChunkResult]:
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
            value=lambda candidate: candidate["upload_date"],
        )
    elif resolved.sort == "popularity":
        candidates = rank_by_value(
            candidates,
            value=popularity,
        )

    return [video_result(candidate) for candidate in candidates[: resolved.limit]]


def collect_candidates(resolved: ResolvedRequest) -> list[dict[str, Any]]:
    """Collection oversamples more under an upload-date filter because it discards candidates afterwards."""
    request = resolved.request
    fields = required_fields(resolved)
    date_filtered = resolved.upload_cutoff is not None
    date_fetch_limit = resolved.limit * DATE_FILTER_OVERSAMPLE_FACTOR
    if not request.channels:
        entries = youtube_candidates(
            resolved,
            fetch_limit=date_fetch_limit
            if date_filtered
            else resolved.limit * YOUTUBE_OVERSAMPLE_FACTOR,
        )
        return hydrate_candidates(entries, resolved, fields=fields)
    candidates: list[dict[str, Any]] = []
    for listing in channel_listings(
        request,
        limit=date_fetch_limit if date_filtered else resolved.limit,
    ):
        candidates.extend(
            hydrate_candidates(listing, resolved, fields=fields, newest_first=True)
        )
    return candidates


def required_fields(resolved: ResolvedRequest) -> frozenset[str]:
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
    """Flat listings can omit fields (channel listings omit upload dates), so unknown values pass a lenient prefilter, and only survivors missing a needed field are fetched.

    A newest-first listing under an upload cutoff is fetched in batches that stop at the first video older than the cutoff, because every later video is older too.
    Failed fetches leave candidates unhydrated for the strict filter to judge.
    Hydrated candidates keep the raw response under "metadata", so later stages of this search reuse it instead of requesting it again.
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
            candidate["id"]: candidate["id"]
            for candidate in batch
            if any(candidate.get(field) is None for field in fields)
        }
        metadata_by_id, _failures = collect_concurrently(
            missing,
            client.metadata,
            max_workers=MAX_CONCURRENT_METADATA_FETCHES,
        )
        for candidate in batch:
            metadata = metadata_by_id.get(candidate["id"])
            hydrated.append(
                candidate
                if metadata is None
                else {
                    **candidate,
                    **metadata,
                    "id": candidate["id"],
                    "upload_date": normalize_upload_date(metadata.get("upload_date")),
                    "source": candidate["source"],
                    "metadata": metadata,
                }
            )
        if stop_before is not None and any(
            candidate["upload_date"] is not None
            and candidate["upload_date"] < stop_before
            for candidate in hydrated[start:]
        ):
            break
    return hydrated
