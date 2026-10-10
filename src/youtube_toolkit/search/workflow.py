"""Search workflow orchestration for mixable source, mode, and return-type choices.

Sources discover candidate videos from YouTube, channel listings, or their combination.
Native mode stays metadata-first and returns video objects after local filters, ranking, and limits.
Relevance order is delegated to YouTube: each query's results keep YouTube's ranking, and several queries are interleaved rather than rescored.
Semantic mode runs title, transcript, chunk-index, and scoring stages, then returns chunks by default.
Full transcript output remains an explicit transcript command rather than a search side effect.
"""

from __future__ import annotations

from typing import Any

from ..config import MAX_CONCURRENT_METADATA_FETCHES
from ..execution import collect_concurrently
from ..youtube.client import YtDlpClient
from .discovery import channel_listings, youtube_candidates
from .filters import (
    candidate_id,
    dedupe_candidates,
    filter_candidates,
    normalized_upload_date,
    popularity,
    rank_by_value,
    upload_cutoff,
    video_result,
)
from .request import (
    SearchRequest,
    SearchSort,
    resolve_sort,
    validate_request,
)

SOURCE_OVERSAMPLE_FACTOR = 3
DATE_FILTER_SOURCE_OVERSAMPLE_FACTOR = 12


def search(request: SearchRequest) -> list[dict[str, Any]]:
    """Run the public search workflow across configured sources, filters, and result shapes."""
    validate_request(request)
    sort = resolve_sort(request)
    cutoff = upload_cutoff(request)

    candidates = filter_candidates(
        dedupe_candidates(collect_sources(request, sort=sort, cutoff=cutoff)),
        request,
        cutoff=cutoff,
    )
    if request.mode == "semantic":
        from .semantic import search_transcripts

        return search_transcripts(request, candidates)
    if sort == "recency":
        candidates = rank_by_value(
            candidates,
            value=lambda candidate: candidate.get("upload_date"),
        )
    elif sort == "popularity":
        candidates = rank_by_value(
            candidates,
            value=popularity,
        )

    return [video_result(candidate) for candidate in candidates[: request.limit]]


def collect_sources(
    request: SearchRequest,
    *,
    sort: SearchSort,
    cutoff: str | None,
) -> list[dict[str, Any]]:
    """Collect candidates from each requested source, hydrated with any metadata the filters or sort need.

    Sources oversample more under an upload-date filter because it discards candidates after collection.
    """
    fields = required_fields(request, sort=sort, cutoff=cutoff)
    date_fetch_limit = request.limit * DATE_FILTER_SOURCE_OVERSAMPLE_FACTOR
    candidates: list[dict[str, Any]] = []
    for source in request.sources:
        if source == "youtube":
            entries = youtube_candidates(
                request,
                fetch_limit=date_fetch_limit
                if cutoff is not None
                else request.limit * SOURCE_OVERSAMPLE_FACTOR,
                sort=sort,
            )
            candidates.extend(
                hydrate_candidates(entries, request, fields=fields, cutoff=cutoff)
            )
        elif source == "channel":
            for listing in channel_listings(
                request,
                limit=date_fetch_limit if cutoff is not None else request.limit,
            ):
                candidates.extend(
                    hydrate_candidates(
                        listing,
                        request,
                        fields=fields,
                        cutoff=cutoff,
                        newest_first=True,
                    )
                )
        else:
            raise ValueError(f"unsupported search source: {source}")
    return candidates


def required_fields(
    request: SearchRequest,
    *,
    sort: SearchSort,
    cutoff: str | None,
) -> frozenset[str]:
    """Name the candidate fields that active filters or native result ordering compare."""
    fields: set[str] = set()
    if request.min_views is not None or (
        request.mode == "native" and sort == "popularity"
    ):
        fields.add("view_count")
    if request.min_duration is not None or request.max_duration is not None:
        fields.add("duration")
    if cutoff is not None or (request.mode == "native" and sort == "recency"):
        fields.add("upload_date")
    return frozenset(fields)


def hydrate_candidates(
    candidates: list[dict[str, Any]],
    request: SearchRequest,
    *,
    fields: frozenset[str],
    cutoff: str | None,
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    """Fetch full metadata for candidates that pass the known filters but lack a field the filters or sort need.

    Flat listings omit fields (channel listings carry no view counts or upload dates), so unknown values pass a lenient prefilter and only its survivors are fetched, concurrently and in order.
    A newest-first listing under an upload cutoff is fetched in batches that stop at the first video older than the cutoff, because every later video is older too.
    Failed fetches leave candidates unhydrated for the strict filter to judge.
    Hydrated candidates also keep the raw response under "metadata", so later stages of this search reuse it instead of requesting it again.
    """
    survivors = filter_candidates(
        candidates, request, cutoff=cutoff, allow_missing=True
    )
    stop_before = cutoff if newest_first else None
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
