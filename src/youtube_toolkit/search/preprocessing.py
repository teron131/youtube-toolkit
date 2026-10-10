"""Search candidate policy owns dedupe, local filters, ranking values, and public result projection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, Literal, NotRequired, TypedDict

from ..identity import extract_channel_identity, extract_video_id
from ..schemas import normalize_number
from .request import ResolvedRequest

SortableValue = str | int | float | tuple[int | float, ...]


class SearchVideoFields(TypedDict):
    """Video facts every search result carries, whether it is a video or a chunk of one."""

    title: str | None
    channel_id: str | None
    channel_handle: str | None
    upload_date: str | None
    duration: float | None
    view_count: int | None
    like_count: int | None


class VideoResult(SearchVideoFields):
    """One video in search output; `score` appears only for semantic search and `warnings` only when preparation degraded."""

    type: Literal["video"]
    video_id: str | None
    source: str | None
    score: NotRequired[float]
    warnings: NotRequired[list[str]]


class ChunkResult(TypedDict):
    """One transcript passage in semantic search output, with its video's facts nested under `video`."""

    type: Literal["chunk"]
    video_id: str
    chunk_index: int
    text: str
    score: float
    video: SearchVideoFields
    warnings: NotRequired[list[str]]


def interlace_relevance(
    candidate_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge each query's result list by taking turns, deduping by video ID.

    This is the ranking for multi-query search, not just a merge: every list arrives in YouTube's own relevance order, and nothing is scored locally because that would mean fetching and processing every candidate.
    Taking the next result from each list in turn assumes equal ranks are equally relevant, which keeps the merged order roughly descending at no cost.
    """
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


def candidate_id(candidate: dict[str, Any]) -> str | None:
    """Resolve a candidate's video ID from the known yt-dlp and URL fields."""
    for key in ("id", "video_id", "url", "webpage_url"):
        value = candidate.get(key)
        if isinstance(value, str):
            video_id = extract_video_id(value)
            if video_id is not None:
                return video_id
    return None


def video_result(
    candidate: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> VideoResult:
    """Video results expose the stable public key set plus preparation warnings when transcript work degraded."""
    result: VideoResult = {
        "type": "video",
        "video_id": candidate_id(candidate),
        **candidate_metadata(candidate),
        "source": candidate.get("source"),
    }
    if "score" in candidate:
        result["score"] = candidate["score"]
    if warnings:
        result["warnings"] = warnings
    return result


def candidate_metadata(candidate: dict[str, Any]) -> SearchVideoFields:
    """Project shared video facts, retaining a handle supplied under yt-dlp's uploader identity fields."""
    return {
        "title": candidate.get("title"),
        "channel_id": candidate.get("channel_id"),
        "channel_handle": extract_channel_identity(candidate).channel_handle,
        "upload_date": candidate.get("upload_date"),
        "duration": candidate.get("duration"),
        "view_count": candidate.get("view_count"),
        "like_count": candidate.get("like_count"),
    }


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
        if all(
            isinstance(item, int | float) and not isinstance(item, bool)
            for item in value
        ):
            return value
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value or None
    return None


def popularity(candidate: dict[str, Any]) -> tuple[float, float] | None:
    """Popularity sorts by views first and likes second while treating absent counters as zero."""
    view_count = normalize_number(candidate.get("view_count"))
    like_count = normalize_number(candidate.get("like_count"))
    if view_count is None and like_count is None:
        return None
    return view_count or 0.0, like_count or 0.0


def filter_candidates(
    candidates: list[dict[str, Any]],
    resolved: ResolvedRequest,
    *,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    """Keep the candidates that pass every metadata filter of the resolved request."""
    return [
        candidate
        for candidate in candidates
        if passes_filters(candidate, resolved, allow_missing=allow_missing)
    ]


def passes_filters(
    candidate: dict[str, Any],
    resolved: ResolvedRequest,
    *,
    allow_missing: bool,
) -> bool:
    """Decide whether one candidate satisfies every metadata filter of the resolved request.

    With allow_missing, absent view counts, durations, and upload dates pass so callers can fetch them before strict filtering.
    """
    if candidate_id(candidate) is None:
        return False
    request = resolved.request

    def checked(field: str) -> bool:
        return not allow_missing or candidate.get(field) is not None

    view_count = candidate.get("view_count")
    if (
        request.min_views is not None
        and checked("view_count")
        and (not isinstance(view_count, int) or view_count < request.min_views)
    ):
        return False

    duration = candidate.get("duration")
    if (
        resolved.min_duration_seconds is not None
        and checked("duration")
        and (
            not isinstance(duration, int | float)
            or duration < resolved.min_duration_seconds
        )
    ):
        return False
    if (
        resolved.max_duration_seconds is not None
        and checked("duration")
        and (
            not isinstance(duration, int | float)
            or duration > resolved.max_duration_seconds
        )
    ):
        return False

    if resolved.upload_cutoff is not None and checked("upload_date"):
        upload_date = normalized_upload_date(candidate.get("upload_date"))
        if upload_date is None or upload_date < resolved.upload_cutoff:
            return False

    channel = channel_text(candidate)
    if not text_matches(channel, request.include_channel) or text_excluded(
        channel, request.exclude_channel
    ):
        return False

    title = candidate.get("title")
    if not text_matches(title, request.include_title) or text_excluded(
        title, request.exclude_title
    ):
        return False

    if (
        request.mode == "native"
        and candidate.get("source") == "channel"
        and request.queries
    ):
        return any(text_matches(title, (query,)) for query in request.queries)
    return True


def channel_text(candidate: dict[str, Any]) -> str:
    """Channel filters match both handle and bare-handle aliases so callers can omit the at-sign."""
    values: list[str] = []
    for value in (
        candidate.get("channel"),
        candidate.get("uploader"),
        extract_channel_identity(candidate).channel_handle,
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


def normalized_upload_date(value: Any) -> str | None:
    """Normalize candidate upload dates from yt-dlp or ISO text for comparisons."""
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    if len(stripped_value) == 8 and stripped_value.isdigit():
        return f"{stripped_value[:4]}-{stripped_value[4:6]}-{stripped_value[6:8]}"
    try:
        return date.fromisoformat(stripped_value).isoformat()
    except ValueError:
        return None
