"""Watchlist channel membership, discovery queue, and consume policy."""

from __future__ import annotations

from typing import Any

import turso

from ..schemas import ChannelIdentity, extract_channel_identity, extract_video_record
from .channels import find_matching_channel_rows, upsert_channel
from .payloads import stored_video_payload
from .videos import find_video_row, upsert_video


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
    """Store a watched channel listing and queue only newly discovered videos."""
    channel_identity = extract_channel_identity(metadata)
    channel_id = upsert_channel(connection, channel_identity)
    if channel_id is None or not is_watchlist_channel(connection, channel_id):
        return []

    new_video_ids: list[str] = []
    for entry in metadata.get("entries") or []:
        video_record = extract_video_record(entry)
        if video_record is None:
            continue
        existing_row = find_video_row(connection, video_record.youtube_video_id)
        stored_video = upsert_video(
            connection,
            entry,
            channel_identity=channel_identity,
        )
        if existing_row is None and stored_video is not None:
            insert_watchlist_video(connection, int(stored_video["database_row_id"]))
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


def insert_watchlist_video(
    connection: turso.Connection,
    video_row_id: int,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO watchlist_videos (video_id)
        VALUES (?)
        """,
        (video_row_id,),
    )


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
        WHERE (videos.transcript IS NULL OR videos.transcript = '')
            AND watchlist_videos.seen_at IS NULL
        ORDER BY videos.upload_date DESC, videos.id DESC
        """
    ).fetchall()
    return [row["youtube_video_id"] for row in rows]


def consume_new_watchlist_videos(
    connection: turso.Connection,
    *,
    channel_identities: list[ChannelIdentity] | None = None,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    """Return scoped new watchlist videos and mark only returned rows as seen."""
    if include_all == bool(channel_identities):
        raise ValueError("Pass either include_all=True or one or more channels")

    connection.execute("BEGIN IMMEDIATE")
    try:
        rows = select_new_watchlist_video_rows(
            connection,
            channel_identities=channel_identities,
            include_all=include_all,
        )
        if not rows:
            connection.execute("COMMIT")
            return []

        video_row_ids = [int(row["id"]) for row in rows]
        placeholders = ", ".join("?" for _ in video_row_ids)
        connection.execute(
            f"""
            UPDATE watchlist_videos
            SET seen_at = CURRENT_TIMESTAMP
            WHERE video_id IN ({placeholders})
                AND seen_at IS NULL
            """,
            video_row_ids,
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return [new_watchlist_video_payload(row) for row in rows]


def select_new_watchlist_video_rows(
    connection: turso.Connection,
    *,
    channel_identities: list[ChannelIdentity] | None,
    include_all: bool,
) -> list[turso.Row]:
    channel_filter_sql = ""
    parameters: list[Any] = []
    if not include_all:
        channel_filter_sql, parameters = build_channel_identity_filter(channel_identities or [])

    return connection.execute(
        f"""
        SELECT
            videos.id,
            videos.youtube_video_id,
            videos.title,
            videos.upload_date,
            videos.view_count,
            videos.transcript,
            channels.youtube_channel_id,
            channels.channel_handle,
            watchlist_videos.discovered_at
        FROM watchlist_videos
        JOIN videos ON videos.id = watchlist_videos.video_id
        LEFT JOIN channels ON channels.id = videos.channel_id
        WHERE watchlist_videos.seen_at IS NULL
            {channel_filter_sql}
        ORDER BY videos.upload_date DESC, watchlist_videos.discovered_at DESC, videos.id DESC
        """,
        parameters,
    ).fetchall()


def build_channel_identity_filter(
    channel_identities: list[ChannelIdentity],
) -> tuple[str, list[Any]]:
    if not channel_identities:
        raise ValueError("Pass either include_all=True or one or more channels")

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


def new_watchlist_video_payload(row: turso.Row) -> dict[str, Any]:
    payload = stored_video_payload(row)
    transcript = payload["transcript"]
    return {
        **payload,
        "url": f"https://www.youtube.com/watch?v={payload['video_id']}",
        "watchlist_discovered_at": row["discovered_at"],
        "transcript_status": "available" if transcript else "missing",
    }
