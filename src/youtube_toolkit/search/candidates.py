"""Search candidates are parsed once as yt-dlp entries arrive, then deduplicated, filtered, ranked, and projected into public results.

A parsed candidate is its yt-dlp entry with the canonical video `id`, its `source`, and an ISO `upload_date` or None, so later stages read those keys without re-deriving them.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import zip_longest
from typing import Any, Literal, NotRequired, TypedDict

from ..identity import extract_channel_identity, extract_video_id
from ..schemas import normalize_number, normalize_upload_date
from .request import ResolvedRequest

CandidateSource = Literal["youtube", "channel"]


class SearchVideoFields(TypedDict):
    title: str | None
    channel_id: str | None
    channel_handle: str | None
    upload_date: str | None
    duration: float | None
    view_count: int | None
    like_count: int | None


class VideoResult(SearchVideoFields):
    """`score` appears only for semantic search, and `warnings` only when preparation degraded."""

    type: Literal["video"]
    video_id: str
    source: CandidateSource
    score: NotRequired[float]
    warnings: NotRequired[list[str]]


class ChunkResult(TypedDict):
    type: Literal["chunk"]
    video_id: str
    chunk_index: int
    text: str
    score: float
    video: SearchVideoFields
    warnings: NotRequired[list[str]]


def parse_candidates(
    entries: list[dict[str, Any]], *, source: CandidateSource
) -> list[dict[str, Any]]:
    """Entries without a recognizable video ID are dropped, and a repeated video keeps its first entry."""
    parsed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        video_id = _video_id(entry)
        if video_id is not None and video_id not in parsed:
            parsed[video_id] = {
                **entry,
                "id": video_id,
                "source": source,
                "upload_date": normalize_upload_date(entry.get("upload_date")),
            }
    return list(parsed.values())


def interlace_relevance(
    entry_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """This is the ranking for multi-query search, not just a merge: every list arrives in YouTube's own relevance order, and nothing is scored locally because that would mean fetching and processing every candidate.

    Taking the next result from each list in turn assumes equal ranks are equally relevant, which keeps the merged order roughly descending at no cost.
    """
    return [
        entry
        for rank in zip_longest(*entry_lists)
        for entry in rank
        if entry is not None
    ]


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique.setdefault(candidate["id"], candidate)
    return list(unique.values())


def video_result(
    candidate: dict[str, Any],
    *,
    score: float | None = None,
    warnings: list[str] | None = None,
) -> VideoResult:
    result: VideoResult = {
        "type": "video",
        "video_id": candidate["id"],
        **video_fields(candidate),
        "source": candidate["source"],
    }
    if score is not None:
        result["score"] = score
    if warnings:
        result["warnings"] = warnings
    return result


def video_fields(candidate: dict[str, Any]) -> SearchVideoFields:
    """yt-dlp may supply the handle only under uploader_id or a channel URL, so it is extracted rather than read from channel_handle."""
    return {
        "title": candidate.get("title"),
        "channel_id": candidate.get("channel_id"),
        "channel_handle": extract_channel_identity(candidate).channel_handle,
        "upload_date": candidate["upload_date"],
        "duration": candidate.get("duration"),
        "view_count": candidate.get("view_count"),
        "like_count": candidate.get("like_count"),
    }


def rank_by_value[Value: (str, tuple[float, float])](
    candidates: list[dict[str, Any]],
    *,
    value: Callable[[dict[str, Any]], Value | None],
) -> list[dict[str, Any]]:
    """Candidates without a value follow the ranked ones in their original order."""
    keyed = [(value(candidate), candidate) for candidate in candidates]
    ranked = sorted(
        ((key, candidate) for key, candidate in keyed if key is not None),
        key=lambda item: item[0],
        reverse=True,
    )
    return [candidate for _key, candidate in ranked] + [
        candidate for key, candidate in keyed if key is None
    ]


def popularity(candidate: dict[str, Any]) -> tuple[float, float] | None:
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
    """With allow_missing, absent view counts, durations, and upload dates pass so callers can fetch them before strict filtering."""
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
        upload_date = candidate["upload_date"]
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
        and candidate["source"] == "channel"
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
    if not needles:
        return True
    haystack = str(value or "").casefold()
    return all(needle.casefold() in haystack for needle in needles)


def text_excluded(
    value: Any,
    needles: tuple[str, ...],
) -> bool:
    if not needles:
        return False
    haystack = str(value or "").casefold()
    return any(needle.casefold() in haystack for needle in needles)


def _video_id(entry: dict[str, Any]) -> str | None:
    for key in ("id", "video_id", "url", "webpage_url"):
        value = entry.get(key)
        if isinstance(value, str):
            video_id = extract_video_id(value)
            if video_id is not None:
                return video_id
    return None
