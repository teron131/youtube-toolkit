"""Watchlist storage owns channel membership, discovered-video timestamps, and recent-video projections."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..schemas import ChannelIdentity, extract_channel_identity, extract_video_record
from .channels import find_matching_channel_rows, upsert_channel
from .connection import LanceConnection, filter_equal, filter_in_values, filter_literal
from .payloads import stored_video_payload
from .videos import upsert_video


def upsert_watchlist_channel(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> str | None:
    """Ensure a channel exists and mark it as a watched channel."""
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
    if (
        connection.delete_where(
            "channels_watchlist",
            f"channel_id IN ({filter_in_values(channel_ids)})",
        )
        == 0
    ):
        return False

    removed_video_ids = [
        row["youtube_video_id"]
        for row in connection.rows(
            "videos",
            where_sql=f"channel_id IN ({filter_in_values(channel_ids)})",
            columns=["youtube_video_id"],
        )
    ]
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
    if (
        connection.count_rows(
            "channels_watchlist",
            where_sql=filter_equal("channel_id", youtube_channel_id),
        )
        == 0
    ):
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
    """Record the first watchlist discovery timestamp for a video."""
    return connection.insert_if_absent(
        "watchlist_videos",
        "video_id",
        [
            {
                "video_id": youtube_video_id,
                "discovered_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            }
        ],
    )


def list_watchlist_channel_refs(connection: LanceConnection) -> list[str]:
    """Display watched channels by handle when available, otherwise by durable channel ID."""
    watched_ids = {
        row["channel_id"]
        for row in connection.rows(
            "channels_watchlist",
            columns=["channel_id"],
        )
    }
    if not watched_ids:
        return []
    rows = sorted(
        connection.rows(
            "channels",
            where_sql=f"youtube_channel_id IN ({filter_in_values(sorted(watched_ids))})",
            columns=[
                "youtube_channel_id",
                "channel_handle",
            ],
        ),
        key=lambda row: row["channel_handle"] or row["youtube_channel_id"] or "",
    )
    return [row["channel_handle"] or row["youtube_channel_id"] for row in rows if row["channel_handle"] or row["youtube_channel_id"]]


def list_missing_watchlist_video_ids(connection: LanceConnection) -> list[str]:
    """Transcript hydration jobs consume watched videos that have not yet persisted transcript text."""
    watchlist_video_ids = {
        row["video_id"]
        for row in connection.rows(
            "watchlist_videos",
            columns=["video_id"],
        )
    }
    if not watchlist_video_ids:
        return []
    rows = connection.rows(
        "videos",
        where_sql=(f"youtube_video_id IN ({filter_in_values(sorted(watchlist_video_ids))}) AND (transcript IS NULL OR transcript = '')"),
        columns=[
            "youtube_video_id",
            "upload_date",
        ],
        order_by=[
            ("upload_date", False),
            ("youtube_video_id", False),
        ],
    )
    return [row["youtube_video_id"] for row in rows]


def list_watchlist_video_ids_missing_embeddings(connection: LanceConnection) -> list[str]:
    """Embedding jobs consume watched videos only after transcript text is already stored."""
    watchlist_video_ids = {
        row["video_id"]
        for row in connection.rows(
            "watchlist_videos",
            columns=["video_id"],
        )
    }
    if not watchlist_video_ids:
        return []
    watchlist_filter_values = filter_in_values(sorted(watchlist_video_ids))
    indexed_video_ids = {
        row["video_id"]
        for row in connection.rows(
            "video_embeddings",
            where_sql=f"video_id IN ({watchlist_filter_values})",
            columns=["video_id"],
        )
    }
    exclusion_filter = ""
    if indexed_video_ids:
        exclusion_filter = f" AND youtube_video_id NOT IN ({filter_in_values(sorted(indexed_video_ids))})"
    rows = connection.rows(
        "videos",
        where_sql=(f"youtube_video_id IN ({watchlist_filter_values}) AND transcript IS NOT NULL AND transcript != ''{exclusion_filter}"),
        columns=[
            "youtube_video_id",
            "upload_date",
        ],
        order_by=[
            ("upload_date", False),
            ("youtube_video_id", False),
        ],
    )
    return [row["youtube_video_id"] for row in rows]


def list_recent_watchlist_video_payloads(
    connection: LanceConnection,
    *,
    cutoff_upload_date: str,
    channel_identities: list[ChannelIdentity] | None = None,
) -> list[dict[str, Any]]:
    """Recent watchlist reads are projections only; sync and transcript hydration own mutations."""
    watched_ids = {
        row["channel_id"]
        for row in connection.rows(
            "channels_watchlist",
            columns=["channel_id"],
        )
    }
    allowed_channel_ids = filter_channel_ids(connection, channel_identities)
    selected_channel_ids = watched_ids if allowed_channel_ids is None else watched_ids & allowed_channel_ids
    if not selected_channel_ids:
        return []
    rows = connection.rows(
        "videos",
        where_sql=(f"channel_id IN ({filter_in_values(sorted(selected_channel_ids))}) AND upload_date IS NOT NULL AND upload_date >= {filter_literal(cutoff_upload_date)}"),
        order_by=[
            ("upload_date", False),
            ("youtube_video_id", False),
        ],
    )
    channel_handles = {
        row["youtube_channel_id"]: row["channel_handle"]
        for row in connection.rows(
            "channels",
            where_sql=f"youtube_channel_id IN ({filter_in_values(sorted(selected_channel_ids))})",
            columns=[
                "youtube_channel_id",
                "channel_handle",
            ],
        )
    }
    return [
        watchlist_video_payload(
            {
                **row,
                "channel_handle": channel_handles.get(row["channel_id"]),
            }
        )
        for row in rows
    ]


def filter_channel_ids(
    connection: LanceConnection,
    channel_identities: list[ChannelIdentity] | None,
) -> set[str] | None:
    """Resolve optional channel filters into stored channel IDs for watchlist listing queries."""
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
    """Project a stored video into the public recent-watchlist payload without transcript text."""
    payload = stored_video_payload(row)
    transcript = payload["transcript"]
    metadata_payload = {key: value for key, value in payload.items() if key != "transcript"}
    return {
        **metadata_payload,
        "url": f"https://www.youtube.com/watch?v={payload['video_id']}",
        "transcript_status": "available" if transcript else "missing",
    }
