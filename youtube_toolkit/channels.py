"""Recent YouTube channel listing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from .caption import fetch_transcript
from .schemas import extract_channel_identity
from .storage import (
    get_connection,
    list_missing_watchlist_video_ids,
    list_watchlist_channel_inputs,
    store_channel_listing,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from .ytdlp_client import YtDlpClient

DEFAULT_LAST_VIDEOS = 10
MIN_DAYS_WINDOW = 10
ALLOWED_CHANNEL_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
}
CHANNEL_PREFIXES = ("/@", "/channel/", "/user/", "/c/")


@dataclass(frozen=True, slots=True)
class ChannelOptions:
    last: int = DEFAULT_LAST_VIDEOS
    days: int | None = None


def normalize_channel_url(channel_input: str) -> str:
    candidate = channel_input.strip()
    if not candidate:
        raise ValueError("Expected a non-empty YouTube channel input")

    parsed = urlparse(candidate)
    if parsed.scheme in {"https", "http"}:
        hostname = (parsed.hostname or "").lower()
        if hostname not in ALLOWED_CHANNEL_HOSTS:
            raise ValueError(f"Unsupported YouTube host: {hostname or 'missing'}")
        path = parsed.path.rstrip("/")
    else:
        stripped_candidate = candidate.lstrip("/")
        if candidate.startswith("@"):
            path = f"/{candidate}"
        elif stripped_candidate.startswith(("channel/", "user/", "c/")):
            path = f"/{stripped_candidate}"
        elif stripped_candidate.startswith("UC"):
            path = f"/channel/{stripped_candidate}"
        else:
            path = f"/@{stripped_candidate.lstrip('@')}"

    base_path = path[: -len("/videos")] if path.endswith("/videos") else path
    if not any(base_path.startswith(prefix) for prefix in CHANNEL_PREFIXES):
        raise ValueError("Expected a YouTube channel URL, handle, or channel ID")
    return f"https://www.youtube.com{base_path}/videos"


def fetch_channel_videos(
    channel_input: str,
    *,
    options: ChannelOptions | None = None,
    client: YtDlpClient | None = None,
) -> dict[str, Any]:
    options = options or ChannelOptions()
    client = client or YtDlpClient()
    validate_channel_options(options)
    channel_url = normalize_channel_url(channel_input)
    metadata = get_recent_channel_entries(
        client,
        channel_url,
        options=options,
    )
    with get_connection() as connection:
        store_channel_listing(connection, metadata)
    return metadata


def validate_channel_options(options: ChannelOptions) -> None:
    if options.last <= 0:
        raise ValueError("--last must be greater than 0")
    if options.days is not None and options.days <= 0:
        raise ValueError("--days must be greater than 0")


def get_recent_channel_entries(
    client: YtDlpClient,
    channel_url: str,
    *,
    options: ChannelOptions,
) -> dict[str, Any]:
    if options.days is None:
        return client.channel_videos(channel_url, last=options.last)

    window_size = max(options.last, MIN_DAYS_WINDOW)
    while True:
        metadata = client.channel_videos(channel_url, last=window_size)
        entries = metadata.get("entries") or []
        metadata["entries"] = filter_entries_by_days(entries, days=options.days)
        if not should_expand_window(entries, days=options.days, window_size=window_size):
            return metadata
        window_size *= 2


def cutoff_date_for_days(days: int) -> date:
    return datetime.now(UTC).date() - timedelta(days=days)


def parse_upload_date(upload_date: str | None) -> date | None:
    if not upload_date or len(upload_date) != 8 or not upload_date.isdigit():
        return None
    return date(
        int(upload_date[:4]),
        int(upload_date[4:6]),
        int(upload_date[6:8]),
    )


def collect_dated_entries(entries: list[dict[str, Any]]) -> list[tuple[dict[str, Any], date]]:
    dated_entries: list[tuple[dict[str, Any], date]] = []
    for entry in entries:
        upload_date = parse_upload_date(entry.get("upload_date"))
        if upload_date is not None:
            dated_entries.append((entry, upload_date))
    return dated_entries


def filter_entries_by_days(
    entries: list[dict[str, Any]],
    *,
    days: int,
) -> list[dict[str, Any]]:
    cutoff_date = cutoff_date_for_days(days)
    return [entry for entry, upload_date in collect_dated_entries(entries) if upload_date >= cutoff_date]


def should_expand_window(
    entries: list[dict[str, Any]],
    *,
    days: int,
    window_size: int,
) -> bool:
    if len(entries) < window_size:
        return False
    dated_entries = [upload_date for _, upload_date in collect_dated_entries(entries)]
    if not dated_entries:
        return False
    return min(dated_entries) >= cutoff_date_for_days(days)


def format_channel_results(metadata: dict[str, Any]) -> str:
    lines: list[str] = []
    for entry in metadata.get("entries") or []:
        upload_date = format_upload_date(entry.get("upload_date"))
        title = entry.get("title") or "Untitled video"
        video_id = entry.get("id")
        video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else "missing-url"
        view_count = format_count(entry.get("view_count"), "views")
        like_count = format_count(entry.get("like_count"), "likes")
        lines.append(f"{upload_date} | {video_url} | {title} | {view_count} | {like_count}")
    return "\n".join(lines)


def format_upload_date(upload_date: str | None) -> str:
    parsed_upload_date = parse_upload_date(upload_date)
    if parsed_upload_date is None:
        return "unknown-date"
    return parsed_upload_date.isoformat()


def format_count(
    value: int | None,
    noun: str,
) -> str:
    if value is None:
        return f"unknown-{noun}"
    return f"{value:,} {noun}"


def add_watchlist_channel(channel_input: str) -> str:
    metadata = fetch_channel_videos(
        channel_input,
        options=ChannelOptions(last=1),
    )
    channel_identity = extract_channel_identity(metadata)
    with get_connection() as connection:
        store_channel_listing(connection, metadata)
        upsert_watchlist_channel(connection, channel_identity)
    return channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_input


def add_blacklist_channel(channel_input: str) -> str:
    metadata = fetch_channel_videos(
        channel_input,
        options=ChannelOptions(last=1),
    )
    channel_identity = extract_channel_identity(metadata)
    with get_connection() as connection:
        store_channel_listing(connection, metadata)
        upsert_blacklist_channel(connection, channel_identity)
    return channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_input


def sync_watchlist(
    *,
    options: ChannelOptions,
) -> list[str]:
    with get_connection() as connection:
        watched_channel_inputs = list_watchlist_channel_inputs(connection)

    if not watched_channel_inputs:
        return []

    for channel_input in watched_channel_inputs:
        fetch_channel_videos(
            channel_input,
            options=options,
        )

    with get_connection() as connection:
        missing_video_ids = list_missing_watchlist_video_ids(connection)

    for video_id in missing_video_ids:
        fetch_transcript(video_id)
    return missing_video_ids
