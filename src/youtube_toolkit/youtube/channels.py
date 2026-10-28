"""Channel listings own reference normalization, limits, and day windows that date flat listings with per-video metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from ..execution import collect_concurrently
from ..identity import ChannelRef, VideoRef, normalize_channel_url
from ..schemas import parse_upload_date
from .client import MAX_CONCURRENT_METADATA_FETCHES, YtDlpClient

DEFAULT_LAST_VIDEOS = 10
MIN_DAYS_WINDOW = 10


@dataclass(frozen=True, slots=True)
class ChannelOptions:
    """With `days`, `last` stops capping the result and only sizes the first listing request, to at least MIN_DAYS_WINDOW."""

    last: int = DEFAULT_LAST_VIDEOS
    days: int | None = None


class ChannelVideosClient(Protocol):
    def channel_videos(self, channel_url: str, *, last: int) -> dict[str, Any]: ...

    def metadata(self, video_ref: VideoRef, /) -> dict[str, Any]: ...


def fetch_channel_videos(
    channel_ref: ChannelRef,
    *,
    options: ChannelOptions | None = None,
    client: ChannelVideosClient | None = None,
) -> dict[str, Any]:
    """A day window keeps only videos uploaded inside it, each given its upload date, and leaves out videos whose metadata cannot be fetched.

    Flat listings omit upload dates, so a day window costs one metadata request per video, made newest first in concurrent batches until a batch reaches an older video or dates none.
    """
    options = options or ChannelOptions()
    client = client or YtDlpClient()
    validate_channel_options(options)
    channel_url = normalize_channel_url(channel_ref)
    if options.days is None:
        return client.channel_videos(
            channel_url,
            last=options.last,
        )

    cutoff_date = cutoff_date_for_days(options.days)
    window_size = max(options.last, MIN_DAYS_WINDOW)
    listing = client.channel_videos(channel_url, last=window_size)
    attempted: set[str] = set()
    upload_dates: dict[str, date] = {}
    while True:
        entries = listing.get("entries") or []
        pending = [
            video_id
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(video_id := entry.get("id"), str)
            and video_id not in attempted
        ]
        # A full listing may continue past its end, so it is refetched at twice the length before a batch would come up short.
        if (
            len(pending) < MAX_CONCURRENT_METADATA_FETCHES
            and len(entries) >= window_size
        ):
            window_size *= 2
            listing = client.channel_videos(channel_url, last=window_size)
            continue
        batch = pending[:MAX_CONCURRENT_METADATA_FETCHES]
        if not batch:
            break
        attempted.update(batch)
        batch_dates = fetch_upload_dates(client, batch)
        upload_dates |= batch_dates
        if not batch_dates or min(batch_dates.values()) < cutoff_date:
            break
    listing["entries"] = [
        {**entry, "upload_date": upload_date.strftime("%Y%m%d")}
        for entry in listing.get("entries") or []
        if isinstance(entry, dict)
        and (upload_date := upload_dates.get(entry.get("id"))) is not None
        and upload_date >= cutoff_date
    ]
    return listing


def validate_channel_options(options: ChannelOptions) -> None:
    if options.last <= 0:
        raise ValueError("--last must be greater than 0")
    if options.days is not None and options.days <= 0:
        raise ValueError("--days must be greater than 0")


def cutoff_date_for_days(days: int) -> date:
    """Count day windows back from today's UTC date, the clock yt-dlp upload dates use."""
    return datetime.now(UTC).date() - timedelta(days=days)


def fetch_upload_dates(
    client: ChannelVideosClient, video_ids: list[str]
) -> dict[str, date]:
    metadata_by_id, _failures = collect_concurrently(
        {video_id: video_id for video_id in video_ids},
        client.metadata,
        max_workers=MAX_CONCURRENT_METADATA_FETCHES,
    )
    return {
        video_id: upload_date
        for video_id, metadata in metadata_by_id.items()
        if (upload_date := parse_upload_date(metadata.get("upload_date"))) is not None
    }
