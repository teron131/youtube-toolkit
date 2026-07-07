"""Recent YouTube channel listing workflows."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from ..identity import ChannelRef, ChannelRefInput, YouTubeRefs, normalize_channel_url, parse_channel_ref
from ..schemas import (
    ChannelIdentity,
    ChannelRemovalResult,
    EmbeddingIndexResult,
    TranscriptFetchResult,
    WatchlistSyncResult,
    extract_channel_identity,
)
from ..storage import (
    auto_index_video_embedding,
    delete_blacklist_channel,
    delete_watchlist_channel,
    get_connection,
    list_missing_watchlist_video_ids,
    list_recent_watchlist_video_payloads,
    list_watchlist_channel_refs,
    list_watchlist_video_ids_missing_embeddings,
    store_channel_listing,
    store_watchlist_channel_listing,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from ..storage.connection import LanceConnection
from .caption import fetch_transcript
from .client import YtDlpClient

DEFAULT_LAST_VIDEOS = 10
MIN_DAYS_WINDOW = 10
MAX_CONCURRENT_LIST_WORKERS = 5


@dataclass(frozen=True, slots=True)
class ChannelOptions:
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
    client: ChannelVideosClient | None = None,
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
    if options.last is not None and options.last <= 0:
        raise ValueError("--last must be greater than 0")
    if options.days is not None and options.days <= 0:
        raise ValueError("--days must be greater than 0")


def get_recent_channel_entries(
    client: ChannelVideosClient,
    channel_url: str,
    *,
    options: ChannelOptions,
) -> dict[str, Any]:
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
        metadata["entries"] = [entry for entry, upload_date in dated_entries if upload_date >= cutoff_date]
        if not should_expand_window(
            entries,
            dated_entries,
            days=options.days,
            window_size=window_size,
        ):
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


def should_expand_window(
    entries: list[dict[str, Any]],
    dated_entries: list[tuple[dict[str, Any], date]],
    *,
    days: int,
    window_size: int,
) -> bool:
    if len(entries) < window_size:
        return False
    if not dated_entries:
        return False
    oldest_upload_date = min(upload_date for _, upload_date in dated_entries)
    return oldest_upload_date >= cutoff_date_for_days(days)


def add_watchlist_channels(channel_refs: ChannelRefInput) -> list[str]:
    return _add_channel_memberships(
        channel_refs,
        add_membership=upsert_watchlist_channel,
    )


def add_blacklist_channels(channel_refs: ChannelRefInput) -> list[str]:
    return _add_channel_memberships(
        channel_refs,
        add_membership=upsert_blacklist_channel,
    )


def list_watchlist_channels() -> list[str]:
    with get_connection() as connection:
        return list_watchlist_channel_refs(connection)


def _add_channel_memberships(
    channel_refs: ChannelRefInput,
    *,
    add_membership: Callable[[LanceConnection, ChannelIdentity], str | None],
) -> list[str]:
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

    channels: list[str] = []
    for channel_ref, metadata in fetched_channels:
        channel_identity = extract_channel_identity(metadata)
        channel = channel_identity.channel_handle or channel_identity.youtube_channel_id or channel_ref
        with get_connection() as connection:
            add_membership(connection, channel_identity)
        channels.append(channel)
    return channels


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
    remove_membership: Callable[[LanceConnection, ChannelIdentity], bool],
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
) -> WatchlistSyncResult:
    with get_connection() as connection:
        watched_channel_refs = list_watchlist_channel_refs(connection)

    if not watched_channel_refs:
        return {
            "channels": [],
            "new_videos": [],
            "transcripts": [],
            "embeddings": [],
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
        missing_embedding_video_ids = list_watchlist_video_ids_missing_embeddings(connection)

    transcripts = fetch_watchlist_transcripts(missing_video_ids)
    embeddings = index_watchlist_embeddings(missing_embedding_video_ids)
    embeddings.extend(transcript["embedding"] for transcript in transcripts if "embedding" in transcript)
    return {
        "channels": watched_channel_refs,
        "new_videos": new_video_ids,
        "transcripts": transcripts,
        "embeddings": embeddings,
    }


def fetch_watchlist_transcripts(
    video_ids: list[str],
) -> list[TranscriptFetchResult]:
    def fetch(video_id: str) -> TranscriptFetchResult:
        try:
            document = fetch_transcript(
                video_id,
                auto_index=False,
            )
            with get_connection() as connection:
                embedding = auto_index_video_embedding(
                    connection,
                    document.metadata.video_id,
                )
            return {
                "status": "stored",
                "video_id": document.metadata.video_id,
                "embedding": embedding,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "video_id": video_id,
                "error": str(exc),
            }

    return _map_concurrently(video_ids, fetch)


def index_watchlist_embeddings(video_ids: list[str]) -> list[EmbeddingIndexResult]:
    def index(video_id: str) -> EmbeddingIndexResult:
        with get_connection() as connection:
            return auto_index_video_embedding(connection, video_id)

    return _map_concurrently(video_ids, index)


def list_recent_watchlist_videos(
    *,
    channel_refs: ChannelRefInput | None = None,
    days: int,
) -> list[dict[str, Any]]:
    if days <= 0:
        raise ValueError("--days must be greater than 0")
    channel_identities = YouTubeRefs.parse(channel_refs).channel_identities()
    with get_connection() as connection:
        return list_recent_watchlist_video_payloads(
            connection,
            cutoff_upload_date=cutoff_date_for_days(days).isoformat(),
            channel_identities=channel_identities or None,
        )
