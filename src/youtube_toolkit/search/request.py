"""Search request contracts validate caller policy before workflows touch YouTube or embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, get_args

from ..identity import ChannelRef, is_channel_id, parse_channel_ref

SearchSource = Literal["youtube", "channel"]
SearchMode = Literal["native", "semantic"]
SearchReturnType = Literal["video", "chunk"]
SearchSort = Literal["relevance", "recency", "popularity"]
SOURCE_OVERSAMPLE_FACTOR = 3
DATE_FILTER_SOURCE_OVERSAMPLE_FACTOR = 12


@dataclass(frozen=True, slots=True)
class ChannelTarget:
    """Resolved channel-search target, separating exact channel refs from fuzzy channel names."""

    youtube_channel_id: str | None = None
    channel_handle: str | None = None
    channel_name: str | None = None
    exact_ref: str | None = None

    @classmethod
    def from_ref(cls, channel_ref: ChannelRef) -> ChannelTarget:
        """Interpret a user channel argument as either an exact ref or a channel-name lookup."""
        if not is_exact_channel_ref(channel_ref):
            channel_name = channel_ref.strip()
            if not channel_name:
                raise ValueError(
                    "channel target must include a name, handle, URL, or channel ID"
                )
            return cls(channel_name=channel_name)
        identity = parse_channel_ref(channel_ref)
        return cls(
            youtube_channel_id=identity.youtube_channel_id,
            channel_handle=identity.channel_handle,
            exact_ref=channel_ref.strip() if not identity.has_identity() else None,
        )

    def channel_ref(self) -> str:
        """Only resolved channel targets can be converted into concrete refs for channel listing fetches."""
        if self.channel_handle is not None:
            return self.channel_handle
        if self.youtube_channel_id is not None:
            return self.youtube_channel_id
        if self.exact_ref is not None:
            return self.exact_ref
        raise ValueError(
            "channel target must include a channel ID, handle, URL, or resolved name"
        )


def is_exact_channel_ref(channel_ref: ChannelRef) -> bool:
    stripped_ref = channel_ref.strip()
    stripped_path = stripped_ref.lstrip("/")
    return (
        stripped_ref.startswith(("@", "http://", "https://"))
        or stripped_path.startswith(("channel/", "user/", "c/"))
        or is_channel_id(stripped_path)
    )


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Validated public search contract shared by CLI and Python callers."""

    query: str
    limit: int
    extra_queries: tuple[str, ...] = ()
    sources: tuple[SearchSource, ...] = ("youtube",)
    mode: SearchMode = "native"
    return_type: SearchReturnType | None = None
    channels: tuple[ChannelTarget, ...] = ()
    sort: SearchSort | None = None
    min_views: int | None = None
    min_duration: str | None = None
    max_duration: str | None = None
    include_channel: tuple[str, ...] = ()
    exclude_channel: tuple[str, ...] = ()
    include_title: tuple[str, ...] = ()
    exclude_title: tuple[str, ...] = ()
    uploaded_after: str | None = None
    uploaded_within_days: int | None = None


def normalize_filter_date(value: str) -> str:
    """Validate and normalize the upload-date filter to ISO date form."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise ValueError("uploaded-after must be YYYY-MM-DD") from error


def parse_duration(value: str | None) -> float | None:
    """Parse seconds, MM:SS, or HH:MM:SS duration filters into seconds."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError("duration filters must be non-empty")
    if stripped.isdigit():
        return float(stripped)

    parts = stripped.split(":")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid duration: {value}")

    total_seconds = 0
    for part in parts:
        total_seconds = total_seconds * 60 + int(part)
    return float(total_seconds)


def resolve_return_type(request: SearchRequest) -> SearchReturnType:
    """Choose the default result shape from semantic mode while honoring explicit caller choice."""
    if request.return_type is not None:
        return request.return_type
    if request.mode == "semantic":
        return "chunk"
    return "video"


def resolve_sort(request: SearchRequest) -> SearchSort:
    """Choose the default ranking policy from source and mode while honoring explicit caller choice."""
    if request.sort is not None:
        return request.sort
    if request.sources == ("channel",) and request.mode == "native":
        return "recency"
    return "relevance"


def validate_request(request: SearchRequest) -> None:
    """Validate the public search contract before any network or embedding work starts."""
    if any(source not in get_args(SearchSource) for source in request.sources):
        raise ValueError("search source must be one of: youtube, channel")
    if len(set(request.sources)) != len(request.sources):
        raise ValueError("search sources must not contain duplicates")
    if request.mode not in get_args(SearchMode):
        raise ValueError("search mode must be one of: native, semantic")
    if request.return_type is not None and request.return_type not in get_args(
        SearchReturnType
    ):
        raise ValueError("search return type must be one of: video, chunk")
    if request.sort is not None and request.sort not in get_args(SearchSort):
        raise ValueError("search sort must be one of: relevance, recency, popularity")
    query_required = request.mode == "semantic" or "youtube" in request.sources
    if query_required and not request.query.strip():
        raise ValueError("search query must be non-empty")
    if request.limit <= 0:
        raise ValueError("search limit must be greater than 0")
    if request.min_views is not None and request.min_views < 0:
        raise ValueError("--min-views cannot be negative")
    min_duration_seconds = parse_duration(request.min_duration)
    max_duration_seconds = parse_duration(request.max_duration)
    if (
        min_duration_seconds is not None
        and max_duration_seconds is not None
        and min_duration_seconds > max_duration_seconds
    ):
        raise ValueError("--min-duration cannot be greater than --max-duration")
    if request.uploaded_within_days is not None and request.uploaded_within_days < 0:
        raise ValueError("uploaded-within-days must be non-negative")
    if request.uploaded_after is not None:
        normalize_filter_date(request.uploaded_after)
    if not request.sources:
        raise ValueError("search requires at least one source")
    if request.extra_queries and "youtube" not in request.sources:
        raise ValueError("extra queries require youtube source")
    if request.return_type == "chunk" and request.mode != "semantic":
        raise ValueError("chunk return requires semantic mode")
    if request.channels and "channel" not in request.sources:
        raise ValueError("--channels requires --source channel")
    if "channel" in request.sources and not request.channels:
        raise ValueError("channel source requires at least one --channels value")
    if (
        request.mode == "native"
        and request.sources == ("youtube",)
        and request.sort == "recency"
    ):
        raise ValueError("native YouTube search does not support --sort recency")
