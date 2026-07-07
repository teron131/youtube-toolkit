"""Search workflow orchestration for mixable source, mode, and return-type choices.

Sources discover candidate videos from YouTube, channel listings, stored rows, or any combination of those.
Native mode stays metadata-first and returns video objects after local filters, ranking, and limits.
Semantic mode hydrates transcript text only when chunk search needs it, then returns chunks by default.
Full transcript output remains an explicit transcript command rather than a search side effect.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..identity import parse_channel_ref
from ..schemas import ChannelIdentity, extract_channel_identity
from ..storage import get_connection, search_stored_videos, store_search_candidates, upsert_video
from ..storage.search import stored_video_rows, video_row_matches_keyword
from ..youtube.channels import ChannelOptions, fetch_channel_metadata
from ..youtube.client import YtDlpClient
from ..youtube.search import SearchOptions as YouTubeSearchOptions
from .filters import (
    candidate_id,
    dedupe_candidates,
    filter_candidates,
    interlace_relevance,
    normalized_upload_date,
    popularity,
    rank_by_value,
    stored_candidate,
    upload_cutoff,
    video_result,
)
from .request import (
    CHANNEL_NAME_SEARCH_LIMIT,
    DATE_FILTER_SOURCE_OVERSAMPLE_FACTOR,
    SOURCE_OVERSAMPLE_FACTOR,
    ChannelTarget,
    SearchRequest,
    SearchSort,
    resolve_return_type,
    resolve_sort,
    validate_request,
)
from .semantic import (
    db_semantic_candidates,
    ensure_semantic_ready,
    index_transcripts,
    index_warnings,
    prefilter_by_title_semantics,
    rank_by_transcript,
    search_chunks,
    search_db_chunks,
    search_db_videos,
    semantic_ready_ids,
)


def search(request: SearchRequest) -> list[dict[str, Any]]:
    """Run the public search workflow across configured sources, filters, persistence, and result shapes."""
    validate_request(request)
    limit = request.limit
    return_type = resolve_return_type(request)
    sort = resolve_sort(request)
    session_id = uuid4().hex
    if request.sources == ("db",) and return_type == "chunk":
        return search_db_chunks(
            request,
            limit=limit,
        )
    if request.sources == ("db",) and request.mode == "semantic":
        return search_db_videos(
            request,
            limit=limit,
        )

    candidates = dedupe_candidates(
        collect_sources(
            request,
            limit,
        )
    )
    if upload_cutoff(request) is not None:
        candidates = filter_candidates(
            candidates,
            request,
            include_upload_date=False,
        )
        candidates = fetch_metadata_for_date_filters(candidates, request)
    candidates = filter_candidates(
        candidates,
        request,
    )
    stage_search_candidates(
        request,
        session_id,
        candidates,
    )
    if request.mode == "native" and sort == "recency":
        candidates = rank_by_value(
            candidates,
            value=lambda candidate: candidate.get("upload_date"),
        )
    elif request.mode == "native" and sort == "popularity":
        candidates = rank_by_value(
            candidates,
            value=popularity,
        )

    if return_type == "chunk" or request.mode == "semantic":
        if request.mode == "semantic":
            candidates = prefilter_by_title_semantics(
                request,
                request.query,
                candidates,
                session_id=session_id,
            )
        persist_metadata(candidates)
        index_results = index_transcripts(candidates)
        ready_ids = semantic_ready_ids(
            candidates,
            index_results,
        )
        warnings_by_video_id = index_warnings(
            index_results,
            ready_ids,
        )
        ensure_semantic_ready(
            candidates,
            index_results,
            ready_ids,
        )
        if return_type == "chunk":
            return search_chunks(
                request,
                request.query,
                candidates,
                warnings_by_video_id=warnings_by_video_id,
            )

        candidates = rank_by_transcript(
            request,
            request.query,
            candidates,
        )
    else:
        candidates = candidates[:limit]
        persist_metadata(candidates)
        warnings_by_video_id = {}
    return [video_result(candidate, warnings_by_video_id=warnings_by_video_id) for candidate in candidates[:limit]]


def collect_sources(
    request: SearchRequest,
    limit: int,
) -> list[dict[str, Any]]:
    """Collect raw candidates from each requested source with oversampling needed by later filters."""
    candidates: list[dict[str, Any]] = []
    source_fetch_limit = limit * SOURCE_OVERSAMPLE_FACTOR
    date_fetch_limit = limit * DATE_FILTER_SOURCE_OVERSAMPLE_FACTOR
    sort = resolve_sort(request)
    for source in request.sources:
        if source == "youtube":
            candidates.extend(
                youtube_candidates(
                    request,
                    fetch_limit=date_fetch_limit if upload_cutoff(request) is not None else source_fetch_limit,
                    sort=sort,
                )
            )
        elif source == "channel":
            candidates.extend(
                channel_candidates(
                    request,
                    limit=date_fetch_limit if upload_cutoff(request) is not None else limit,
                )
            )
        elif source == "db":
            if request.mode == "semantic":
                candidates.extend(db_semantic_candidates(request))
                continue
            candidates.extend(
                db_candidates(
                    request,
                    limit=date_fetch_limit if upload_cutoff(request) is not None else source_fetch_limit,
                    sort=sort,
                )
            )
        else:
            raise ValueError(f"unsupported search source: {source}")
    return candidates


def fetch_metadata_for_date_filters(
    candidates: list[dict[str, Any]],
    request: SearchRequest,
) -> list[dict[str, Any]]:
    """Hydrate YouTube candidates that need upload dates before upload-date filtering can run."""
    if upload_cutoff(request) is None:
        return candidates
    client = YtDlpClient()
    enriched_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("source") != "youtube" or candidate.get("upload_date") is not None:
            enriched_candidates.append(candidate)
            continue
        video_id = candidate_id(candidate)
        if video_id is None:
            enriched_candidates.append(candidate)
            continue
        try:
            metadata = client.metadata(video_id)
        except Exception:
            enriched_candidates.append(candidate)
            continue
        enriched_candidates.append(
            {
                **candidate,
                **metadata,
                "id": video_id,
                "upload_date": normalized_upload_date(metadata.get("upload_date")),
                "source": candidate.get("source"),
            }
        )
    return enriched_candidates


def youtube_candidates(
    request: SearchRequest,
    *,
    fetch_limit: int,
    sort: SearchSort,
) -> list[dict[str, Any]]:
    """Fetch and normalize YouTube search candidates for the requested ranking policy."""
    options = YouTubeSearchOptions(
        limit=fetch_limit,
        sort="popularity" if sort == "popularity" else "relevance",
        min_views=None,
        min_duration=None,
        max_duration=None,
    )
    candidate_lists = [
        youtube_search(
            query,
            options=options,
        )
        for query in (request.query, *request.extra_queries)
    ]
    if sort == "relevance":
        return [
            {
                **candidate,
                "source": "youtube",
            }
            for candidate in interlace_relevance(candidate_lists)
        ]
    candidates = [candidate for candidates in candidate_lists for candidate in candidates]
    return [
        {
            **candidate,
            "source": "youtube",
        }
        for candidate in dedupe_candidates(candidates)
    ]


def youtube_search(
    query: str,
    *,
    options: YouTubeSearchOptions,
) -> list[dict[str, Any]]:
    """Call the yt-dlp search adapter and keep only object entries."""
    client = YtDlpClient()
    payload = client.search(
        query,
        fetch=options.limit,
        sort=options.sort,
    )
    return [entry for entry in payload.get("entries") or [] if isinstance(entry, dict)]


def channel_candidates(
    request: SearchRequest,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch candidate videos from explicitly selected channel listings."""
    if not request.channels:
        raise ValueError("channel source requires at least one --channels value")
    candidates: list[dict[str, Any]] = []
    for selection in request.channels:
        channel_ref, channel_identity = resolve_channel_listing(selection)
        metadata = fetch_channel_metadata(
            channel_ref,
            options=ChannelOptions(last=limit),
        )
        candidates.extend(
            channel_entries(
                metadata,
                known_channel=channel_identity,
            )
        )
    return [
        {
            **candidate,
            "source": "channel",
        }
        for candidate in dedupe_candidates(candidates)
    ]


def resolve_channel_listing(selection: ChannelTarget) -> tuple[str, ChannelIdentity]:
    """Resolve a channel target to a concrete channel ref plus identity for listing fetches."""
    if selection.channel_name is None:
        channel_ref = selection.channel_ref()
        return (
            channel_ref,
            parse_channel_ref(channel_ref),
        )
    wanted_name = normalized_channel_name(selection.channel_name)
    search_options = YouTubeSearchOptions(
        limit=CHANNEL_NAME_SEARCH_LIMIT,
        sort="relevance",
        min_views=None,
        min_duration=None,
        max_duration=None,
    )
    for query in (selection.channel_name, f"{selection.channel_name} official YouTube channel"):
        for match in youtube_search(query, options=search_options):
            if not any(normalized_channel_name(value) == wanted_name for value in (match.get("channel"), match.get("uploader")) if isinstance(value, str)):
                continue
            identity = extract_channel_identity(match)
            channel_ref = identity.youtube_channel_id or identity.channel_handle
            if channel_ref is not None:
                return (
                    channel_ref,
                    identity,
                )
    raise ValueError(f"Could not resolve YouTube channel name: {selection.channel_name}")


def normalized_channel_name(value: str) -> str:
    """Normalize channel names for fuzzy YouTube-search confirmation."""
    return " ".join(value.casefold().split())


def db_candidates(
    request: SearchRequest,
    *,
    limit: int,
    sort: SearchSort,
) -> list[dict[str, Any]]:
    """Load stored-video candidates using the DB lookup path and search ranking policy."""
    with get_connection() as connection:
        if sort == "popularity":
            normalized_query = request.query.strip().casefold()
            rows = [row for row in stored_video_rows(connection) if video_row_matches_keyword(row, normalized_query)]
            return [
                {
                    **stored_candidate(row),
                    "source": "db",
                }
                for row in rows
            ]
        payloads = search_stored_videos(
            connection,
            request.query,
            limit=limit,
            sort="upload_date",
        )
    return [
        {
            **stored_candidate(payload),
            "source": "db",
        }
        for payload in payloads
    ]


def channel_entries(
    metadata: dict[str, Any],
    *,
    known_channel: ChannelIdentity,
) -> list[dict[str, Any]]:
    """Attach known channel identity fields to listing entries before candidate filtering."""
    channel_identity = extract_channel_identity(
        metadata,
        fallback_channel_id=known_channel.youtube_channel_id,
        fallback_channel_handle=known_channel.channel_handle,
    )
    return [
        {
            **entry,
            "channel_id": entry.get("channel_id") or channel_identity.youtube_channel_id,
            "channel_handle": entry.get("channel_handle") or channel_identity.channel_handle,
        }
        for entry in metadata.get("entries") or []
        if isinstance(entry, dict)
    ]


def persist_metadata(candidates: list[dict[str, Any]]) -> None:
    """Persist candidate metadata so future transcript, summary, and DB searches share one catalog."""
    if not candidates:
        return
    with get_connection() as connection:
        for candidate in candidates:
            upsert_video(connection, candidate)


def stage_search_candidates(
    request: SearchRequest,
    session_id: str,
    candidates: list[dict[str, Any]],
) -> None:
    """Persist non-DB search candidates for title-embedding prefiltering within this search session."""
    if not any(candidate.get("source") != "db" for candidate in candidates):
        return
    with get_connection() as connection:
        store_search_candidates(
            connection,
            session_id,
            request.query,
            candidates,
        )
