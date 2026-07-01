"""Recent YouTube channel listing workflows."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypedDict

import turso

from ..identity import ChannelRef, ChannelRefInput, YouTubeRefs, normalize_channel_url, parse_channel_ref
from ..schemas import ChannelIdentity, extract_channel_identity
from ..storage import (
    consume_new_watchlist_videos,
    delete_blacklist_channel,
    delete_watchlist_channel,
    get_connection,
    list_missing_watchlist_video_ids,
    list_watchlist_channel_refs,
    store_channel_listing,
    store_watchlist_channel_listing,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from .caption import fetch_transcript
from .client import YtDlpClient

DEFAULT_LAST_VIDEOS = 10
MIN_DAYS_WINDOW = 10
MAX_CONCURRENT_LIST_WORKERS = 5


@dataclass(frozen=True, slots=True)
class ChannelOptions:
    last: int = DEFAULT_LAST_VIDEOS
    days: int | None = None


class ChannelMembershipResult(TypedDict):
    channel_ref: ChannelRef
    channel: str


class ChannelRemovalResult(ChannelMembershipResult):
    removed: bool


def fetch_channel_videos(
    channel_ref: ChannelRef,
    *,
    options: ChannelOptions | None = None,
    client: YtDlpClient | None = None,
) -> dict[str, Any]:
    metadata = fetch_channel_metadata(
        channel_ref,
        options=options,
        client=client,
    )
    with get_connection() as connection:
        store_channel_listing(connection, metadata)
    return metadata


def fetch_channel_metadata(
    channel_ref: ChannelRef,
    *,
    options: ChannelOptions | None = None,
    client: YtDlpClient | None = None,
) -> dict[str, Any]:
    options = options or ChannelOptions()
    client = client or YtDlpClient()
    validate_channel_options(options)
    channel_url = normalize_channel_url(channel_ref)
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


def add_watchlist_channels(channel_refs: ChannelRefInput) -> list[ChannelMembershipResult]:
    return _add_channel_memberships(
        channel_refs,
        add_membership=upsert_watchlist_channel,
    )


def add_blacklist_channels(channel_refs: ChannelRefInput) -> list[ChannelMembershipResult]:
    return _add_channel_memberships(
        channel_refs,
        add_membership=upsert_blacklist_channel,
    )


def _add_channel_memberships(
    channel_refs: ChannelRefInput,
    *,
    add_membership: Callable[[turso.Connection, ChannelIdentity], int | None],
) -> list[ChannelMembershipResult]:
    fetched_channels = _map_concurrently(
        list(YouTubeRefs.parse(channel_refs)),
        lambda channel_ref: (
            channel_ref,
            fetch_channel_metadata(
                channel_ref,
                options=ChannelOptions(last=1),
            ),
        ),
    )

    results: list[ChannelMembershipResult] = []
    for channel_ref, metadata in fetched_channels:
        channel_identity = extract_channel_identity(metadata)
        channel = channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_ref
        with get_connection() as connection:
            store_channel_listing(connection, metadata)
            add_membership(connection, channel_identity)
        results.append(
            {
                "channel_ref": channel_ref,
                "channel": channel,
            }
        )
    return results


def remove_watchlist_channels(channel_refs: ChannelRefInput) -> list[ChannelRemovalResult]:
    """Remove channels from the local watchlist without deleting catalog data."""
    return _remove_channel_memberships(
        channel_refs,
        remove_membership=delete_watchlist_channel,
    )


def remove_blacklist_channels(channel_refs: ChannelRefInput) -> list[ChannelRemovalResult]:
    """Remove channels from the local search blacklist without deleting catalog data."""
    return _remove_channel_memberships(
        channel_refs,
        remove_membership=delete_blacklist_channel,
    )


def _remove_channel_memberships(
    channel_refs: ChannelRefInput,
    *,
    remove_membership: Callable[[turso.Connection, ChannelIdentity], bool],
) -> list[ChannelRemovalResult]:
    resolved_channels = _map_concurrently(
        list(YouTubeRefs.parse(channel_refs)),
        _resolve_channel_identity,
    )

    results: list[ChannelRemovalResult] = []
    for channel_ref, channel_identity in resolved_channels:
        channel = channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_ref
        with get_connection() as connection:
            removed = remove_membership(connection, channel_identity)
        results.append(
            {
                "channel_ref": channel_ref,
                "channel": channel,
                "removed": removed,
            }
        )
    return results


def _resolve_channel_identity(channel_ref: ChannelRef) -> tuple[ChannelRef, ChannelIdentity]:
    channel_identity = parse_channel_ref(channel_ref)
    if channel_identity.has_identity():
        return channel_ref, channel_identity

    metadata = fetch_channel_metadata(
        channel_ref,
        options=ChannelOptions(last=1),
    )
    return channel_ref, extract_channel_identity(metadata)


def _map_concurrently[WorkItem, WorkResult](
    items: list[WorkItem],
    worker: Callable[[WorkItem], WorkResult],
) -> list[WorkResult]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_LIST_WORKERS, len(items))) as executor:
        return list(executor.map(worker, items))


def sync_watchlist(
    *,
    options: ChannelOptions,
) -> dict[str, Any]:
    with get_connection() as connection:
        watched_channel_refs = list_watchlist_channel_refs(connection)

    if not watched_channel_refs:
        return {
            "watched_channels": [],
            "new_video_ids": [],
            "hydrated_transcripts": [],
            "failed_transcripts": [],
        }

    fetched_channels = _map_concurrently(
        watched_channel_refs,
        lambda channel_ref: (
            channel_ref,
            fetch_channel_metadata(
                channel_ref,
                options=options,
            ),
        ),
    )

    new_video_ids: list[str] = []
    for _channel_ref, metadata in fetched_channels:
        with get_connection() as connection:
            new_video_ids.extend(store_watchlist_channel_listing(connection, metadata))

    with get_connection() as connection:
        missing_video_ids = list_missing_watchlist_video_ids(connection)

    hydrated_transcripts, failed_transcripts = hydrate_watchlist_transcripts(missing_video_ids)
    return {
        "watched_channels": watched_channel_refs,
        "new_video_ids": new_video_ids,
        "hydrated_transcripts": hydrated_transcripts,
        "failed_transcripts": failed_transcripts,
    }


def hydrate_watchlist_transcripts(
    video_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    def hydrate(video_id: str) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        try:
            document = fetch_transcript(video_id)
            return (
                {
                    "video_id": document.metadata.video_id,
                    "source": document.source,
                    "language": document.language,
                },
                None,
            )
        except Exception as exc:
            return (
                None,
                {
                    "video_id": video_id,
                    "error": str(exc),
                },
            )

    hydrated_transcripts: list[dict[str, Any]] = []
    failed_transcripts: list[dict[str, str]] = []
    for hydrated, failed in _map_concurrently(video_ids, hydrate):
        if hydrated is not None:
            hydrated_transcripts.append(hydrated)
        if failed is not None:
            failed_transcripts.append(failed)
    return hydrated_transcripts, failed_transcripts


def watch_new_videos(
    *,
    channel_refs: ChannelRefInput | None = None,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    channel_identities = YouTubeRefs.parse(channel_refs).channel_identities()
    with get_connection() as connection:
        return consume_new_watchlist_videos(
            connection,
            channel_identities=channel_identities or None,
            include_all=include_all,
        )
