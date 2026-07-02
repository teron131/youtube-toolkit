"""Watchlist channel membership and discovered-video tracking."""

from __future__ import annotations

from typing import Any

import turso

from ..schemas import ChannelIdentity, extract_channel_identity, extract_video_record
from .channels import find_matching_channel_rows, upsert_channel
from .payloads import stored_video_payload
from .videos import upsert_video


def upsert_watchlist_channel(
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> int | None:
    channel_id = upsert_channel(connection, channel_identity)
    if channel_id is None:
        return None
    connection.execute(
        """
        INSERT OR IGNORE INTO channels_watchlist (channel_id)
        VALUES (?)
        """,
        (channel_id,),
    )
    return channel_id


def delete_watchlist_channel(
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> bool:
    """Remove a known channel from the watchlist membership table."""
    if not channel_identity.has_identity():
        return False

    matching_rows = find_matching_channel_rows(connection, channel_identity)
    if not matching_rows:
        return False

    channel_ids = [int(row["id"]) for row in matching_rows]
    placeholders = ", ".join("?" for _ in channel_ids)
    membership_rows = connection.execute(
        f"""
        SELECT channel_id
        FROM channels_watchlist
        WHERE channel_id IN ({placeholders})
        """,
        channel_ids,
    ).fetchall()
    if not membership_rows:
        return False

    connection.execute(
        f"""
        DELETE FROM channels_watchlist
        WHERE channel_id IN ({placeholders})
        """,
        channel_ids,
    )
    connection.execute(
        f"""
        DELETE FROM watchlist_videos
        WHERE video_id IN (
            SELECT id
            FROM videos
            WHERE channel_id IN ({placeholders})
        )
        """,
        channel_ids,
    )
    return True


def store_watchlist_channel_listing(
    connection: turso.Connection,
    metadata: dict[str, Any],
) -> list[str]:
    """Store a watched channel listing and return newly discovered video IDs."""
    channel_identity = extract_channel_identity(metadata)
    channel_id = upsert_channel(connection, channel_identity)
    if channel_id is None or not is_watchlist_channel(connection, channel_id):
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
        if stored_video is not None and record_watchlist_video(connection, int(stored_video["database_row_id"])):
            new_video_ids.append(video_record.youtube_video_id)
    return new_video_ids


def is_watchlist_channel(
    connection: turso.Connection,
    channel_id: int,
) -> bool:
    return (
        connection.execute(
            """
            SELECT 1
            FROM channels_watchlist
            WHERE channel_id = ?
            """,
            (channel_id,),
        ).fetchone()
        is not None
    )


def record_watchlist_video(
    connection: turso.Connection,
    video_row_id: int,
) -> bool:
    existing_row = connection.execute(
        """
        SELECT 1
        FROM watchlist_videos
        WHERE video_id = ?
        """,
        (video_row_id,),
    ).fetchone()
    if existing_row is not None:
        return False
    connection.execute(
        """
        INSERT OR IGNORE INTO watchlist_videos (video_id)
        VALUES (?)
        """,
        (video_row_id,),
    )
    return True


def list_watchlist_channel_refs(connection: turso.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT channels.youtube_channel_id, channels.channel_handle
        FROM channels_watchlist
        JOIN channels ON channels.id = channels_watchlist.channel_id
        ORDER BY COALESCE(channels.channel_handle, channels.youtube_channel_id)
        """
    ).fetchall()
    return [row["channel_handle"] or row["youtube_channel_id"] for row in rows if row["channel_handle"] or row["youtube_channel_id"]]


def list_missing_watchlist_video_ids(connection: turso.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT videos.youtube_video_id
        FROM videos
        JOIN watchlist_videos ON watchlist_videos.video_id = videos.id
        WHERE videos.transcript IS NULL OR videos.transcript = ''
        ORDER BY videos.upload_date DESC, videos.id DESC
        """
    ).fetchall()
    return [row["youtube_video_id"] for row in rows]


def list_recent_watchlist_video_payloads(
    connection: turso.Connection,
    *,
    cutoff_upload_date: str,
    channel_identities: list[ChannelIdentity] | None = None,
) -> list[dict[str, Any]]:
    """Return objectively recent videos from watched channels without mutating storage."""
    channel_filter_sql = ""
    parameters: list[Any] = [cutoff_upload_date]
    if channel_identities:
        channel_filter_sql, channel_filter_parameters = build_channel_identity_filter(channel_identities)
        parameters.extend(channel_filter_parameters)

    rows = connection.execute(
        f"""
        SELECT
            videos.id,
            videos.youtube_video_id,
            videos.title,
            videos.upload_date,
            videos.view_count,
            videos.transcript,
            channels.youtube_channel_id,
            channels.channel_handle
        FROM channels_watchlist
        JOIN channels ON channels.id = channels_watchlist.channel_id
        JOIN videos ON videos.channel_id = channels.id
        WHERE videos.upload_date IS NOT NULL
            AND videos.upload_date >= ?
            {channel_filter_sql}
        ORDER BY videos.upload_date DESC, videos.id DESC
        """,
        parameters,
    ).fetchall()
    return [watchlist_video_payload(row) for row in rows]


def build_channel_identity_filter(
    channel_identities: list[ChannelIdentity],
) -> tuple[str, list[Any]]:
    if not channel_identities:
        raise ValueError("Channel filters must include at least one channel")

    clauses: list[str] = []
    parameters: list[Any] = []
    for identity in channel_identities:
        if not identity.has_identity():
            raise ValueError("Channel filters must include a handle or channel ID")
        identity_clauses: list[str] = []
        if identity.youtube_channel_id is not None:
            identity_clauses.append("channels.youtube_channel_id = ?")
            parameters.append(identity.youtube_channel_id)
        if identity.channel_handle is not None:
            identity_clauses.append("channels.channel_handle = ?")
            parameters.append(identity.channel_handle)
        clauses.append(f"({' OR '.join(identity_clauses)})")
    return f"AND ({' OR '.join(clauses)})", parameters


def watchlist_video_payload(row: turso.Row) -> dict[str, Any]:
    payload = stored_video_payload(row)
    transcript = payload["transcript"]
    metadata_payload = {key: value for key, value in payload.items() if key != "transcript"}
    return {
        **metadata_payload,
        "url": f"https://www.youtube.com/watch?v={payload['video_id']}",
        "transcript_status": "available" if transcript else "missing",
    }
