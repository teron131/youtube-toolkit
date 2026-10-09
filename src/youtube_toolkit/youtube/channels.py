"""Channel listing retrieval owns reference normalization, limits, and date-window expansion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from ..config import DEFAULT_LAST_VIDEOS
from ..identity import ChannelRef, normalize_channel_url
from .client import YtDlpClient

MIN_DAYS_WINDOW = 10


@dataclass(frozen=True, slots=True)
class ChannelOptions:
    """Caller policy for how much of a channel listing to inspect."""

    last: int | None = DEFAULT_LAST_VIDEOS
    days: int | None = None


class ChannelVideosClient(Protocol):
    """Minimal yt-dlp surface needed for channel listing workflows."""

    def channel_videos(
        self,
        channel_url: str,
        *,
        last: int | None,
        flat: bool = True,
    ) -> dict[str, Any]: ...


def fetch_channel_videos(
    channel_ref: ChannelRef,
    *,
    options: ChannelOptions | None = None,
    client: ChannelVideosClient | None = None,
) -> dict[str, Any]:
    """Resolve and validate a channel request, expanding date-window fetches until the listing crosses the cutoff."""
    options = options or ChannelOptions()
    client = client or YtDlpClient()
    validate_channel_options(options)
    channel_url = normalize_channel_url(channel_ref)
    if options.days is None:
        return client.channel_videos(
            channel_url,
            last=options.last,
        )

    window_size = max(options.last or MIN_DAYS_WINDOW, MIN_DAYS_WINDOW)
    while True:
        metadata = client.channel_videos(
            channel_url,
            last=window_size,
            flat=False,
        )
        entries = metadata.get("entries") or []
        dated_entries = collect_dated_entries(entries)
        cutoff_date = cutoff_date_for_days(options.days)
        metadata["entries"] = [
            entry for entry, upload_date in dated_entries if upload_date >= cutoff_date
        ]
        if not should_expand_window(
            entries,
            dated_entries,
            days=options.days,
            window_size=window_size,
        ):
            return metadata
        window_size *= 2


def validate_channel_options(options: ChannelOptions) -> None:
    """Validate channel listing limits before any YouTube request is made."""
    if options.last is not None and options.last <= 0:
        raise ValueError("--last must be greater than 0")
    if options.days is not None and options.days <= 0:
        raise ValueError("--days must be greater than 0")


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


def collect_dated_entries(
    entries: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], date]]:
    """Keep listing entries that have parseable upload dates for window expansion."""
    dated_entries: list[tuple[dict[str, Any], date]] = []
    for entry in entries:
        upload_date = parse_upload_date(entry.get("upload_date"))
        if upload_date is not None:
            dated_entries.append((entry, upload_date))
    return dated_entries


def should_expand_window(
    entries: list[dict[str, Any]],
    dated_entries: list[tuple[dict[str, Any], date]],
    *,
    days: int,
    window_size: int,
) -> bool:
    """Keep expanding only when the current full page is still newer than the requested window."""
    if len(entries) < window_size:
        return False
    if not dated_entries:
        return False
    oldest_upload_date = min(upload_date for _, upload_date in dated_entries)
    return oldest_upload_date >= cutoff_date_for_days(days)
