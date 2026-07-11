"""Search candidate policy owns dedupe, local filters, ranking values, and public result projection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..identity import extract_video_id
from ..schemas import ChannelIdentity, extract_channel_identity
from ..storage import get_connection, list_blacklist_channel_identities
from ..storage.connection import filter_in_values
from .request import (
    VIDEO_RESULT_KEYS,
    SearchRequest,
    SortableValue,
    normalize_filter_date,
    parse_duration,
)


def interlace_relevance(candidate_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge multiple relevance-ranked lists fairly while deduping by video ID."""
    merged: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    longest_list = max((len(candidates) for candidates in candidate_lists), default=0)
    for candidate_index in range(longest_list):
        for candidates in candidate_lists:
            if candidate_index >= len(candidates):
                continue
            candidate = candidates[candidate_index]
            video_id = candidate_id(candidate)
            if video_id is None or video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            merged.append(candidate)
    return merged


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of each resolvable video candidate and normalize its ID field."""
    merged: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    for candidate in candidates:
        video_id = candidate_id(candidate)
        if video_id is None or video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)
        merged.append({**candidate, "id": video_id})
    return merged


def stored_candidate(video: dict[str, Any]) -> dict[str, Any]:
    """Translate stored-video payloads into the candidate shape consumed by search ranking."""
    video_id = video["youtube_video_id"] if "youtube_video_id" in video else video["video_id"]
    return {
        "id": video_id,
        "title": video["title"],
        "upload_date": video["upload_date"],
        "duration": video["duration"],
        "view_count": video["view_count"],
        "like_count": video["like_count"],
        "channel_id": video["channel_id"],
        "channel_handle": video["channel_handle"],
        "metadata_fetched_at": video["metadata_fetched_at"],
        "transcript_fetched_at": video["transcript_fetched_at"],
    }


def candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    video_ids: list[str] = []
    seen_video_ids: set[str] = set()
    for candidate in candidates:
        video_id = candidate_id(candidate)
        if video_id is not None and video_id not in seen_video_ids:
            seen_video_ids.add(video_id)
            video_ids.append(video_id)
    return video_ids


def candidate_id(candidate: dict[str, Any]) -> str | None:
    """Resolve a candidate's video ID from the known yt-dlp, stored, and URL fields."""
    for key in ("id", "video_id", "url", "webpage_url"):
        value = candidate.get(key)
        if isinstance(value, str):
            video_id = extract_video_id(value)
            if video_id is not None:
                return video_id
    return None


def is_blacklisted_channel(
    candidate: dict[str, Any],
    blacklisted_channels: tuple[ChannelIdentity, ...],
) -> bool:
    """Return whether a candidate's channel identity matches the local blacklist."""
    if not blacklisted_channels:
        return False
    candidate_channel = extract_channel_identity(candidate)
    if not candidate_channel.has_identity():
        return False
    for blacklisted_channel in blacklisted_channels:
        same_channel_id = candidate_channel.youtube_channel_id is not None and candidate_channel.youtube_channel_id == blacklisted_channel.youtube_channel_id
        same_channel_handle = candidate_channel.channel_handle is not None and candidate_channel.channel_handle == blacklisted_channel.channel_handle
        if same_channel_id or same_channel_handle:
            return True
    return False


def chunk_result(
    row: dict[str, Any],
    video: dict[str, Any] | None,
    *,
    warnings: list[str] | None,
) -> dict[str, Any]:
    """Semantic chunk results keep video metadata optional because chunk indexing can outlive video projection."""
    result = {
        "type": "chunk",
        "video_id": row["video_id"],
        "chunk_index": row["chunk_index"],
        "text": row["text"],
        "score": float(row["similarity"]),
        "video": None if video is None else {key: video[key] for key in VIDEO_RESULT_KEYS},
    }
    if warnings:
        result["warnings"] = warnings
    return result


def video_result(
    candidate: dict[str, Any],
    *,
    warnings_by_video_id: dict[str, list[str]],
) -> dict[str, Any]:
    """Video results expose the stable public key set plus preparation warnings when transcript work degraded."""
    result = {
        "type": "video",
        "video_id": candidate_id(candidate),
        **{key: candidate.get(key) for key in VIDEO_RESULT_KEYS},
        "source": candidate.get("source"),
    }
    if "score" in candidate:
        result["score"] = candidate["score"]
    warnings = warnings_by_video_id.get(str(result["video_id"]))
    if warnings:
        result["warnings"] = warnings
    return result


def rank_by_value(
    candidates: list[dict[str, Any]],
    *,
    value: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Sort candidates by a derived value while preserving unsortable candidates at the end."""
    present: list[tuple[SortableValue, dict[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    for candidate in candidates:
        sort_value = sortable(value(candidate))
        if sort_value is None:
            missing.append(candidate)
        else:
            present.append((sort_value, candidate))
    sorted_present = sorted(
        present,
        key=lambda item: item[0],
        reverse=True,
    )
    return [candidate for _value, candidate in sorted_present] + missing


def sortable(value: Any) -> SortableValue | None:
    """Normalize supported ranking values and reject booleans or unorderable payloads."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, tuple):
        if all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
            return value
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value or None
    return None


def popularity(candidate: dict[str, Any]) -> tuple[int | float, int | float] | None:
    """Popularity sorts by views first and likes second while treating absent counters as zero."""
    view_count = candidate.get("view_count")
    like_count = candidate.get("like_count")
    if isinstance(view_count, bool) or not isinstance(view_count, int | float):
        view_count = None
    if isinstance(like_count, bool) or not isinstance(like_count, int | float):
        like_count = None
    if view_count is None and like_count is None:
        return None
    return (
        view_count or 0,
        like_count or 0,
    )


def filter_candidates(
    candidates: list[dict[str, Any]],
    request: SearchRequest,
    *,
    include_upload_date: bool = True,
) -> list[dict[str, Any]]:
    """Apply local search filters with the current blacklist loaded from storage."""
    if not candidates:
        return []
    with get_connection() as connection:
        blacklisted_channels = tuple(list_blacklist_channel_identities(connection))
    return [
        candidate
        for candidate in candidates
        if passes_filters(
            candidate,
            request,
            blacklisted_channels,
            include_upload_date=include_upload_date,
        )
    ]


def passes_filters(
    candidate: dict[str, Any],
    request: SearchRequest,
    blacklisted_channels: tuple[ChannelIdentity, ...],
    *,
    include_upload_date: bool = True,
) -> bool:
    """Evaluate all local metadata and blacklist filters for a single candidate."""
    if candidate_id(candidate) is None:
        return False
    if is_blacklisted_channel(candidate, blacklisted_channels):
        return False

    view_count = candidate.get("view_count")
    if request.min_views is not None and (not isinstance(view_count, int) or view_count < request.min_views):
        return False

    duration = candidate.get("duration")
    min_duration_seconds = parse_duration(request.min_duration)
    max_duration_seconds = parse_duration(request.max_duration)
    if min_duration_seconds is not None and (not isinstance(duration, int | float) or duration < min_duration_seconds):
        return False
    if max_duration_seconds is not None and (not isinstance(duration, int | float) or duration > max_duration_seconds):
        return False

    cutoff = upload_cutoff(request)
    if include_upload_date and cutoff is not None:
        upload_date = normalized_upload_date(candidate.get("upload_date"))
        if upload_date is None or upload_date < cutoff:
            return False

    channel = channel_text(candidate)
    if not text_matches(channel, request.include_channel) or text_excluded(channel, request.exclude_channel):
        return False

    title = candidate.get("title")
    if not text_matches(title, request.include_title) or text_excluded(title, request.exclude_title):
        return False

    if request.mode == "native" and candidate.get("source") == "channel" and request.query.strip():
        return text_matches(title, (request.query,))
    return True


def channel_text(candidate: dict[str, Any]) -> str:
    """Channel filters match both handle and bare-handle aliases so callers can omit the at-sign."""
    values: list[str] = []
    for value in (
        candidate.get("channel"),
        candidate.get("uploader"),
        candidate.get("channel_handle"),
        candidate.get("channel_id"),
    ):
        if value in (None, ""):
            continue
        text = str(value)
        values.append(text)
        if text.startswith("@"):
            values.append(text[1:])
        else:
            values.append(f"@{text}")
    return " ".join(dict.fromkeys(values))


def text_matches(
    value: Any,
    needles: tuple[str, ...],
) -> bool:
    """Require every configured substring to match case-insensitively."""
    if not needles:
        return True
    haystack = str(value or "").casefold()
    return all(needle.casefold() in haystack for needle in needles)


def text_excluded(
    value: Any,
    needles: tuple[str, ...],
) -> bool:
    """Reject text when any configured substring matches case-insensitively."""
    if not needles:
        return False
    haystack = str(value or "").casefold()
    return any(needle.casefold() in haystack for needle in needles)


def db_chunk_filter(
    request: SearchRequest,
    blacklisted_channels: tuple[ChannelIdentity, ...],
) -> str | None:
    """Translate DB-applicable search filters into a SQL fragment for chunk search."""
    clauses: list[str] = []
    if request.min_views is not None:
        clauses.append(f"view_count >= {request.min_views}")

    min_duration_seconds = parse_duration(request.min_duration)
    max_duration_seconds = parse_duration(request.max_duration)
    if min_duration_seconds is not None:
        clauses.append(f"duration >= {min_duration_seconds}")
    if max_duration_seconds is not None:
        clauses.append(f"duration <= {max_duration_seconds}")

    cutoff = upload_cutoff(request)
    if cutoff is not None:
        clauses.append(f"upload_date >= '{cutoff}'")

    blacklisted_channel_ids = [channel.youtube_channel_id for channel in blacklisted_channels if channel.youtube_channel_id is not None]
    if blacklisted_channel_ids:
        clauses.append(f"(channel_id IS NULL OR channel_id NOT IN ({filter_in_values(blacklisted_channel_ids)}))")

    blacklisted_handles = [channel.channel_handle for channel in blacklisted_channels if channel.channel_handle is not None]
    if blacklisted_handles:
        clauses.append(f"(channel_handle IS NULL OR channel_handle NOT IN ({filter_in_values(blacklisted_handles)}))")

    if not clauses:
        return None
    return " AND ".join(f"({clause})" for clause in clauses)


def upload_cutoff(request: SearchRequest) -> str | None:
    """Resolve the strictest upload-date cutoff from absolute and relative filters."""
    cutoffs: list[str] = []
    if request.uploaded_after is not None:
        cutoffs.append(normalize_filter_date(request.uploaded_after))
    if request.uploaded_within_days is not None:
        cutoffs.append((datetime.now(UTC).astimezone().date() - timedelta(days=request.uploaded_within_days)).isoformat())
    return max(cutoffs) if cutoffs else None


def normalized_upload_date(value: Any) -> str | None:
    """Normalize candidate upload dates from yt-dlp or stored ISO text for comparisons."""
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    if len(stripped_value) == 8 and stripped_value.isdigit():
        return f"{stripped_value[:4]}-{stripped_value[4:6]}-{stripped_value[6:8]}"
    try:
        return date.fromisoformat(stripped_value).isoformat()
    except ValueError:
        return None
