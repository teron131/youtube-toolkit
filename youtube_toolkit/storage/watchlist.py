"""Watchlist channel membership and discovered-video tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..schemas import ChannelIdentity, extract_channel_identity, extract_video_record
from .channels import find_matching_channel_rows, upsert_channel
from .connection import LanceConnection, filter_in_values
from .payloads import stored_video_payload
from .videos import upsert_video, video_payload_row


def upsert_watchlist_channel(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> str | None:
    youtube_channel_id = upsert_channel(connection, channel_identity)
    if youtube_channel_id is None:
        return None
    connection.upsert_rows(
        "channels_watchlist",
        "channel_id",
        [{"channel_id": youtube_channel_id}],
    )
    return youtube_channel_id


def delete_watchlist_channel(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> bool:
    """Remove a known channel from the watchlist membership table."""
    if not channel_identity.has_identity():
        return False

    matching_rows = find_matching_channel_rows(connection, channel_identity)
    channel_ids = [row["youtube_channel_id"] for row in matching_rows if row["youtube_channel_id"] is not None]
    if not channel_ids:
        return False
    existing_ids = {row["channel_id"] for row in connection.rows("channels_watchlist") if row["channel_id"] in set(channel_ids)}
    if not existing_ids:
        return False

    connection.delete_where(
        "channels_watchlist",
        f"channel_id IN ({filter_in_values(channel_ids)})",
    )
    removed_video_ids = [row["youtube_video_id"] for row in connection.rows("videos") if row["channel_id"] in set(channel_ids)]
    if removed_video_ids:
        connection.delete_where(
            "watchlist_videos",
            f"video_id IN ({filter_in_values(removed_video_ids)})",
        )
    return True


def store_watchlist_channel_listing(
    connection: LanceConnection,
    metadata: dict[str, Any],
) -> list[str]:
    """Store a watched channel listing and return newly discovered video IDs."""
    channel_identity = extract_channel_identity(metadata)
    youtube_channel_id = upsert_channel(connection, channel_identity)
    if youtube_channel_id is None:
        return []
    if not any(row["channel_id"] == youtube_channel_id for row in connection.rows("channels_watchlist")):
        return []

    new_video_ids: list[str] = []
    for entry in metadata.get("entries") or []:
        video_record = extract_video_record(entry)
        if video_record is None:
            continue
        stored_video = upsert_video(
            connection,
            entry,
            channel_identity=channel_identity,
        )
        if stored_video is not None and record_watchlist_video(connection, video_record.youtube_video_id):
            new_video_ids.append(video_record.youtube_video_id)
    return new_video_ids


def record_watchlist_video(
    connection: LanceConnection,
    youtube_video_id: str,
) -> bool:
    if any(row["video_id"] == youtube_video_id for row in connection.rows("watchlist_videos")):
        return False
    connection.upsert_rows(
        "watchlist_videos",
        "video_id",
        [
            {
                "video_id": youtube_video_id,
                "discovered_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            }
        ],
    )
    return True


def list_watchlist_channel_refs(connection: LanceConnection) -> list[str]:
    watched_ids = {row["channel_id"] for row in connection.rows("channels_watchlist")}
    rows = sorted(
        [row for row in connection.rows("channels") if row["youtube_channel_id"] in watched_ids],
        key=lambda row: row["channel_handle"] or row["youtube_channel_id"] or "",
    )
    return [row["channel_handle"] or row["youtube_channel_id"] for row in rows if row["channel_handle"] or row["youtube_channel_id"]]


def list_missing_watchlist_video_ids(connection: LanceConnection) -> list[str]:
    watchlist_video_ids = {row["video_id"] for row in connection.rows("watchlist_videos")}
    videos = [row for row in connection.rows("videos") if row["youtube_video_id"] in watchlist_video_ids and not row["transcript"]]
    return [
        row["youtube_video_id"]
        for row in sorted(
            videos,
            key=lambda row: (
                row["upload_date"] or "",
                row["youtube_video_id"],
            ),
            reverse=True,
        )
    ]


def list_watchlist_video_ids_missing_embeddings(connection: LanceConnection) -> list[str]:
    watchlist_video_ids = {row["video_id"] for row in connection.rows("watchlist_videos")}
    indexed_video_ids = {row["video_id"] for row in connection.rows("video_embeddings")}
    videos = [row for row in connection.rows("videos") if row["youtube_video_id"] in watchlist_video_ids and row["transcript"] and row["youtube_video_id"] not in indexed_video_ids]
    return [
        row["youtube_video_id"]
        for row in sorted(
            videos,
            key=lambda row: (
                row["upload_date"] or "",
                row["youtube_video_id"],
            ),
            reverse=True,
        )
    ]


def list_recent_watchlist_video_payloads(
    connection: LanceConnection,
    *,
    cutoff_upload_date: str,
    channel_identities: list[ChannelIdentity] | None = None,
) -> list[dict[str, Any]]:
    """Return objectively recent videos from watched channels without mutating storage."""
    watched_ids = {row["channel_id"] for row in connection.rows("channels_watchlist")}
    allowed_channel_ids = filter_channel_ids(connection, channel_identities)
    rows = [
        row
        for row in connection.rows("videos")
        if row["channel_id"] in watched_ids
        and row["upload_date"] is not None
        and row["upload_date"] >= cutoff_upload_date
        and (allowed_channel_ids is None or row["channel_id"] in allowed_channel_ids)
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            row["upload_date"] or "",
            row["youtube_video_id"],
        ),
        reverse=True,
    )
    return [watchlist_video_payload(video_payload_row(connection, row)) for row in rows]


def filter_channel_ids(
    connection: LanceConnection,
    channel_identities: list[ChannelIdentity] | None,
) -> set[str] | None:
    if not channel_identities:
        return None
    allowed_ids: set[str] = set()
    for identity in channel_identities:
        if not identity.has_identity():
            raise ValueError("Channel filters must include a handle or channel ID")
        for row in find_matching_channel_rows(connection, identity):
            if row["youtube_channel_id"] is not None:
                allowed_ids.add(row["youtube_channel_id"])
    return allowed_ids


def watchlist_video_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = stored_video_payload(row)
    transcript = payload["transcript"]
    metadata_payload = {key: value for key, value in payload.items() if key != "transcript"}
    return {
        **metadata_payload,
        "url": f"https://www.youtube.com/watch?v={payload['video_id']}",
        "transcript_status": "available" if transcript else "missing",
    }
