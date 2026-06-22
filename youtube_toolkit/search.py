"""YouTube search with local metadata filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import ChannelIdentity, extract_channel_identity
from .storage import get_connection, list_blacklist_channel_identities, upsert_video
from .ytdlp_client import SEARCH_SORT_PARAMS, YtDlpClient

DEFAULT_LIMIT = 10
DEFAULT_MIN_VIEWS = 10_000
DEFAULT_MIN_DURATION = "3:00"
DEFAULT_MAX_DURATION = "8:00:00"


@dataclass(frozen=True, slots=True)
class SearchOptions:
    limit: int = DEFAULT_LIMIT
    sort: str = "relevance"
    min_views: int | None = DEFAULT_MIN_VIEWS
    min_duration: str | None = DEFAULT_MIN_DURATION
    max_duration: str | None = DEFAULT_MAX_DURATION
    include_channel: tuple[str, ...] = ()
    exclude_channel: tuple[str, ...] = ()
    include_title: tuple[str, ...] = ()
    exclude_title: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchFilters:
    min_views: int | None
    min_duration_seconds: float | None
    max_duration_seconds: float | None
    include_channel: tuple[str, ...]
    exclude_channel: tuple[str, ...]
    include_title: tuple[str, ...]
    exclude_title: tuple[str, ...]
    blacklisted_channels: tuple[ChannelIdentity, ...]


def search_videos(
    query: str,
    *,
    options: SearchOptions | None = None,
    client: YtDlpClient | None = None,
) -> list[dict[str, Any]]:
    options = options or SearchOptions()
    client = client or YtDlpClient()
    validate_options(query, options)
    payload = client.search(
        query,
        fetch=options.limit * 3,
        sort=options.sort,
    )
    with get_connection() as connection:
        blacklisted_channels = list_blacklist_channel_identities(connection)
        entries = filter_entries(
            payload.get("entries") or [],
            options,
            blacklisted_channels=blacklisted_channels,
        )
        for entry in entries:
            upsert_video(connection, entry)
    return entries


def validate_options(
    query: str,
    options: SearchOptions,
) -> None:
    if not query.strip():
        raise ValueError("query must be non-empty")
    if options.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if options.sort not in SEARCH_SORT_PARAMS:
        raise ValueError(f"--sort must be one of: {', '.join(sorted(SEARCH_SORT_PARAMS))}")
    if options.min_views is not None and options.min_views < 0:
        raise ValueError("--min-views cannot be negative")
    min_duration = parse_duration(options.min_duration)
    max_duration = parse_duration(options.max_duration)
    if min_duration is not None and max_duration is not None and min_duration > max_duration:
        raise ValueError("--min-duration cannot be greater than --max-duration")


def filter_entries(
    entries: list[dict[str, Any]],
    options: SearchOptions,
    *,
    blacklisted_channels: list[ChannelIdentity] | None = None,
) -> list[dict[str, Any]]:
    filters = resolve_filters(
        options,
        blacklisted_channels=blacklisted_channels or [],
    )
    filtered = [entry for entry in entries if is_video_entry(entry) and not is_blacklisted_channel(entry, filters.blacklisted_channels) and entry_matches_filters(entry, filters)]
    return filtered[: options.limit]


def resolve_filters(
    options: SearchOptions,
    *,
    blacklisted_channels: list[ChannelIdentity],
) -> SearchFilters:
    return SearchFilters(
        min_views=options.min_views,
        min_duration_seconds=parse_duration(options.min_duration),
        max_duration_seconds=parse_duration(options.max_duration),
        include_channel=options.include_channel,
        exclude_channel=options.exclude_channel,
        include_title=options.include_title,
        exclude_title=options.exclude_title,
        blacklisted_channels=tuple(blacklisted_channels),
    )


def is_video_entry(entry: dict[str, Any]) -> bool:
    if entry.get("ie_key") != "Youtube":
        return False
    url = str(entry.get("url") or entry.get("webpage_url") or "")
    return "watch?v=" in url or bool(entry.get("id"))


def is_blacklisted_channel(
    entry: dict[str, Any],
    blacklisted_channels: tuple[ChannelIdentity, ...],
) -> bool:
    if not blacklisted_channels:
        return False
    entry_channel = extract_channel_identity(entry)
    if not entry_channel.has_identity():
        return False
    return any(channel_identity_matches(entry_channel, blacklisted_channel) for blacklisted_channel in blacklisted_channels)


def channel_identity_matches(
    entry_channel: ChannelIdentity,
    blacklisted_channel: ChannelIdentity,
) -> bool:
    return (entry_channel.youtube_channel_id is not None and entry_channel.youtube_channel_id == blacklisted_channel.youtube_channel_id) or (
        entry_channel.channel_handle is not None and entry_channel.channel_handle == blacklisted_channel.channel_handle
    )


def entry_matches_filters(
    entry: dict[str, Any],
    filters: SearchFilters,
) -> bool:
    view_count = entry.get("view_count")
    if filters.min_views is not None and (not isinstance(view_count, int) or view_count < filters.min_views):
        return False

    duration = entry.get("duration")
    if filters.min_duration_seconds is not None and (not isinstance(duration, int | float) or duration < filters.min_duration_seconds):
        return False
    if filters.max_duration_seconds is not None and (not isinstance(duration, int | float) or duration > filters.max_duration_seconds):
        return False

    channel = entry.get("channel") or entry.get("uploader")
    if not text_matches(channel, filters.include_channel) or text_excluded(channel, filters.exclude_channel):
        return False

    title = entry.get("title")
    return text_matches(title, filters.include_title) and not text_excluded(title, filters.exclude_title)


def parse_duration(value: str | None) -> float | None:
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


def entry_url(entry: dict[str, Any]) -> str:
    url = entry.get("url") or entry.get("webpage_url")
    if isinstance(url, str) and url.startswith("http"):
        return url
    video_id = entry.get("id")
    if isinstance(video_id, str):
        return f"https://www.youtube.com/watch?v={video_id}"
    return "missing-url"


def format_search_results(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in entries:
        title = entry.get("title") or "Untitled video"
        channel = entry.get("channel") or entry.get("uploader") or "unknown-channel"
        view_count = format_count(entry.get("view_count"), "views")
        duration = format_duration(entry.get("duration"))
        lines.append(f"{entry_url(entry)} | {title} | {channel} | {view_count} | {duration}")
    return "\n".join(lines)


def format_count(
    value: int | None,
    noun: str,
) -> str:
    if value is None:
        return f"unknown-{noun}"
    return f"{value:,} {noun}"


def format_duration(duration: int | float | None) -> str:
    if not isinstance(duration, int | float):
        return "unknown-duration"
    total_seconds = int(duration)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
