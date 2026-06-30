"""Recent YouTube channel listing workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from ..schemas import extract_channel_identity, parse_channel_identity_input
from ..storage import (
    consume_new_watchlist_videos,
    delete_blacklist_channel,
    delete_watchlist_channel,
    get_connection,
    list_missing_watchlist_video_ids,
    list_watchlist_channel_inputs,
    store_channel_listing,
    store_watchlist_channel_listing,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from .caption import fetch_transcript
from .client import YtDlpClient

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
    metadata = fetch_channel_metadata(
        channel_input,
        options=options,
        client=client,
    )
    with get_connection() as connection:
        store_channel_listing(connection, metadata)
    return metadata


def fetch_channel_metadata(
    channel_input: str,
    *,
    options: ChannelOptions | None = None,
    client: YtDlpClient | None = None,
) -> dict[str, Any]:
    options = options or ChannelOptions()
    client = client or YtDlpClient()
    validate_channel_options(options)
    channel_url = normalize_channel_url(channel_input)
    return get_recent_channel_entries(
        client,
        channel_url,
        options=options,
    )


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


def remove_watchlist_channel(channel_input: str) -> str:
    """Remove a channel from the local watchlist without deleting catalog data."""
    channel_identity = parse_channel_identity_input(channel_input)
    if not channel_identity.has_identity():
        metadata = fetch_channel_videos(
            channel_input,
            options=ChannelOptions(last=1),
        )
        channel_identity = extract_channel_identity(metadata)
    with get_connection() as connection:
        removed = delete_watchlist_channel(connection, channel_identity)
    label = channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_input
    if removed:
        return f"Removed {label} from watchlist"
    return f"{label} was not in watchlist"


def remove_blacklist_channel(channel_input: str) -> str:
    """Remove a channel from the local search blacklist without deleting catalog data."""
    channel_identity = parse_channel_identity_input(channel_input)
    if not channel_identity.has_identity():
        metadata = fetch_channel_videos(
            channel_input,
            options=ChannelOptions(last=1),
        )
        channel_identity = extract_channel_identity(metadata)
    with get_connection() as connection:
        removed = delete_blacklist_channel(connection, channel_identity)
    label = channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_input
    if removed:
        return f"Removed {label} from blacklist"
    return f"{label} was not in blacklist"


def sync_watchlist(
    *,
    options: ChannelOptions,
) -> dict[str, Any]:
    with get_connection() as connection:
        watched_channel_inputs = list_watchlist_channel_inputs(connection)

    if not watched_channel_inputs:
        return {
            "watched_channels": [],
            "new_video_ids": [],
            "hydrated_transcripts": [],
            "failed_transcripts": [],
        }

    new_video_ids: list[str] = []
    for channel_input in watched_channel_inputs:
        metadata = fetch_channel_metadata(
            channel_input,
            options=options,
        )
        with get_connection() as connection:
            new_video_ids.extend(store_watchlist_channel_listing(connection, metadata))

    with get_connection() as connection:
        missing_video_ids = list_missing_watchlist_video_ids(connection)

    hydrated_transcripts, failed_transcripts = hydrate_watchlist_transcripts(missing_video_ids)
    return {
        "watched_channels": watched_channel_inputs,
        "new_video_ids": new_video_ids,
        "hydrated_transcripts": hydrated_transcripts,
        "failed_transcripts": failed_transcripts,
    }


def hydrate_watchlist_transcripts(
    video_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    hydrated_transcripts: list[dict[str, Any]] = []
    failed_transcripts: list[dict[str, str]] = []
    for video_id in video_ids:
        try:
            document = fetch_transcript(video_id)
            hydrated_transcripts.append(
                {
                    "video_id": document.metadata.video_id,
                    "source": document.source,
                    "language": document.language,
                }
            )
        except Exception as exc:
            failed_transcripts.append(
                {
                    "video_id": video_id,
                    "error": str(exc),
                }
            )
    return hydrated_transcripts, failed_transcripts


def watch_new_videos(
    *,
    channels: list[str] | None = None,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    channel_identities = [parse_channel_identity_input(channel) for channel in channels or []]
    with get_connection() as connection:
        return consume_new_watchlist_videos(
            connection,
            channel_identities=channel_identities or None,
            include_all=include_all,
        )
