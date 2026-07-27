"""Search workflow orchestration for mixable source, mode, and return-type choices.

Sources discover candidate videos from YouTube, channel listings, stored rows, or any combination of those.
Native mode stays metadata-first and returns video objects after local filters, ranking, and limits.
Semantic mode runs title, transcript, chunk-index, and scoring stages, then returns chunks by default.
Full transcript output remains an explicit transcript command rather than a search side effect.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..capabilities import capability_available, require_capability
from ..identity import parse_channel_ref
from ..schemas import ChannelIdentity, extract_channel_identity
from ..youtube.channels import ChannelOptions, fetch_channel_metadata
from ..youtube.client import YtDlpClient
from .filters import (
    candidate_id,
    dedupe_candidates,
    filter_candidates,
    interlace_relevance,
    normalized_upload_date,
    popularity,
    rank_by_value,
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
    SemanticSearchState,
    ensure_semantic_index,
    fetch_semantic_transcripts,
    prefilter_by_title_semantics,
    prepare_semantic_index,
    project_semantic_results,
    score_semantic_index,
)


def search(request: SearchRequest) -> list[dict[str, Any]]:
    """Run the public search workflow across configured sources, filters, persistence, and result shapes."""
    validate_request(request)
    if "db" in request.sources:
        require_capability("storage", "database search")
        from .database import search_db_chunks, search_db_videos

    limit = request.limit
    return_type = resolve_return_type(request)
    sort = resolve_sort(request)
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
    semantic_state = SemanticSearchState(database_available=capability_available("storage")) if request.mode == "semantic" else None

    candidates = dedupe_candidates(
        collect_sources(
            request,
            limit,
        )
    )
    storage_warnings: list[str] = []
    if upload_cutoff(request) is not None:
        candidates, warnings = filter_search_candidates(
            candidates,
            request,
            semantic_state=semantic_state,
            include_upload_date=False,
        )
        storage_warnings.extend(warning for warning in warnings if warning not in storage_warnings)
        candidates = fetch_metadata_for_date_filters(candidates, request)
    candidates, warnings = filter_search_candidates(
        candidates,
        request,
        semantic_state=semantic_state,
    )
    storage_warnings.extend(warning for warning in warnings if warning not in storage_warnings)
    if semantic_state is not None:
        semantic_state.warnings.extend(warning for warning in storage_warnings if warning not in semantic_state.warnings)
        candidates = [
            {
                **candidate,
                "upload_date": normalized_upload_date(candidate.get("upload_date")),
            }
            for candidate in candidates
        ]
        session_id = uuid4().hex
        candidates = prefilter_by_title_semantics(
            request,
            candidates,
            semantic_state,
            session_id=session_id,
        )
        documents, transcript_failures = fetch_semantic_transcripts(
            candidates,
            semantic_state,
        )
        semantic_index = prepare_semantic_index(
            candidates,
            documents,
            semantic_state,
        )
        ensure_semantic_index(
            semantic_index,
            transcript_failures,
        )
        scores = score_semantic_index(
            request,
            candidates,
            semantic_index,
            semantic_state,
            return_type=return_type,
        )
        return project_semantic_results(
            request,
            candidates,
            semantic_index,
            scores,
            semantic_state,
            transcript_failures,
            return_type=return_type,
        )
    session_id = uuid4().hex
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

    candidates = candidates[:limit]
    persist_metadata(candidates)
    warnings_by_video_id = {}
    return [video_result(candidate, warnings_by_video_id=warnings_by_video_id) for candidate in candidates[:limit]]


def filter_search_candidates(
    candidates: list[dict[str, Any]],
    request: SearchRequest,
    *,
    semantic_state: SemanticSearchState | None = None,
    include_upload_date: bool = True,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Let fresh semantic search continue without stored blacklist state when DB access is denied."""
    if not capability_available("storage"):
        return (
            filter_candidates(
                candidates,
                request,
                blacklisted_channels=(),
                include_upload_date=include_upload_date,
            ),
            (),
        )
    if semantic_state is not None and not semantic_state.database_available:
        return (
            filter_candidates(
                candidates,
                request,
                blacklisted_channels=(),
                include_upload_date=include_upload_date,
            ),
            (),
        )
    try:
        return (
            filter_candidates(
                candidates,
                request,
                include_upload_date=include_upload_date,
            ),
            (),
        )
    except Exception as error:
        from ..storage.connection import is_database_write_permission_error

        if not is_database_write_permission_error(error):
            raise
        if request.mode != "semantic" or "db" in request.sources:
            raise
        if semantic_state is not None:
            semantic_state.database_available = False
        return (
            filter_candidates(
                candidates,
                request,
                blacklisted_channels=(),
                include_upload_date=include_upload_date,
            ),
            ("Database unavailable; stored blacklist was not applied",),
        )


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
            from .database import db_candidates, db_semantic_candidates

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
    youtube_sort = "popularity" if sort == "popularity" else "relevance"
    candidate_lists = [
        youtube_search(
            query,
            fetch_limit=fetch_limit,
            sort=youtube_sort,
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
    fetch_limit: int,
    sort: str,
) -> list[dict[str, Any]]:
    """Call the yt-dlp search adapter and keep only object entries."""
    client = YtDlpClient()
    payload = client.search(
        query,
        fetch=fetch_limit,
        sort=sort,
    )
    return [entry for entry in payload.get("entries") or [] if isinstance(entry, dict)]


def channel_candidates(
    request: SearchRequest,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch candidate videos from explicitly selected channel listings."""
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
    for query in (selection.channel_name, f"{selection.channel_name} official YouTube channel"):
        for match in youtube_search(
            query,
            fetch_limit=CHANNEL_NAME_SEARCH_LIMIT,
            sort="relevance",
        ):
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
    if not candidates or not capability_available("storage"):
        return
    from ..storage import get_connection, upsert_video

    with get_connection() as connection:
        for candidate in candidates:
            upsert_video(connection, candidate)


def stage_search_candidates(
    request: SearchRequest,
    session_id: str,
    candidates: list[dict[str, Any]],
) -> None:
    """Persist non-DB search candidates for title-embedding prefiltering within this search session."""
    if not capability_available("storage") or not any(candidate.get("source") != "db" for candidate in candidates):
        return
    from ..storage import get_connection, store_search_candidates

    with get_connection() as connection:
        store_search_candidates(
            connection,
            session_id,
            request.query,
            candidates,
        )
