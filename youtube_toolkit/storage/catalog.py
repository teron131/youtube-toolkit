"""Local YouTube catalog persistence for channels, videos, watchlists, and blacklists."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..schemas import ChannelIdentity, VideoRecord, extract_channel_identity, extract_video_record


def upsert_channel(
    connection: sqlite3.Connection,
    channel_identity: ChannelIdentity,
) -> int | None:
    if not channel_identity.has_identity():
        return None

    matching_rows = find_matching_channel_rows(connection, channel_identity)
    if matching_rows:
        return merge_channel_rows(connection, matching_rows, channel_identity)

    cursor = connection.execute(
        """
        INSERT INTO channels (youtube_channel_id, channel_handle)
        VALUES (?, ?)
        """,
        (channel_identity.youtube_channel_id, channel_identity.channel_handle),
    )
    return int(cursor.lastrowid)


def find_matching_channel_rows(
    connection: sqlite3.Connection,
    channel_identity: ChannelIdentity,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT id, youtube_channel_id, channel_handle
        FROM channels
        WHERE (? IS NOT NULL AND youtube_channel_id = ?)
            OR (? IS NOT NULL AND channel_handle = ?)
        ORDER BY id
        """,
        (
            channel_identity.youtube_channel_id,
            channel_identity.youtube_channel_id,
            channel_identity.channel_handle,
            channel_identity.channel_handle,
        ),
    ).fetchall()


def merge_channel_rows(
    connection: sqlite3.Connection,
    rows: list[sqlite3.Row],
    channel_identity: ChannelIdentity,
) -> int:
    primary_row = rows[0]
    primary_channel_id = primary_row["youtube_channel_id"]
    primary_channel_handle = primary_row["channel_handle"]
    for duplicate_row in rows[1:]:
        if primary_channel_id is None:
            primary_channel_id = duplicate_row["youtube_channel_id"]
        if primary_channel_handle is None:
            primary_channel_handle = duplicate_row["channel_handle"]
        connection.execute(
            "UPDATE videos SET channel_id = ? WHERE channel_id = ?",
            (primary_row["id"], duplicate_row["id"]),
        )
        connection.execute(
            "DELETE FROM channels WHERE id = ?",
            (duplicate_row["id"],),
        )

    connection.execute(
        """
        UPDATE channels
        SET youtube_channel_id = ?, channel_handle = ?
        WHERE id = ?
        """,
        (
            channel_identity.youtube_channel_id or primary_channel_id,
            channel_identity.channel_handle or primary_channel_handle,
            primary_row["id"],
        ),
    )
    return int(primary_row["id"])


def upsert_watchlist_channel(
    connection: sqlite3.Connection,
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


def upsert_blacklist_channel(
    connection: sqlite3.Connection,
    channel_identity: ChannelIdentity,
) -> int | None:
    channel_id = upsert_channel(connection, channel_identity)
    if channel_id is None:
        return None
    connection.execute(
        """
        INSERT OR IGNORE INTO channels_blacklist (channel_id)
        VALUES (?)
        """,
        (channel_id,),
    )
    return channel_id


def upsert_video(
    connection: sqlite3.Connection,
    metadata: dict[str, Any],
    *,
    transcript: str | None = None,
    channel_identity: ChannelIdentity | None = None,
) -> int | None:
    video_record = extract_video_record(metadata, transcript=transcript)
    if video_record is None:
        return None

    resolved_channel = channel_identity
    if resolved_channel is None or not resolved_channel.has_identity():
        resolved_channel = extract_channel_identity(metadata)
    channel_id = upsert_channel(connection, resolved_channel)
    existing_row = find_video_row(connection, video_record.youtube_video_id)
    if existing_row is None:
        return insert_video(connection, video_record, channel_id)

    update_video(connection, existing_row, video_record, channel_id)
    return int(existing_row["id"])


def find_video_row(
    connection: sqlite3.Connection,
    youtube_video_id: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT id, channel_id, title, upload_date, view_count, transcript
        FROM videos
        WHERE youtube_video_id = ?
        """,
        (youtube_video_id,),
    ).fetchone()


def insert_video(
    connection: sqlite3.Connection,
    video_record: VideoRecord,
    channel_id: int | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO videos (
            youtube_video_id,
            channel_id,
            title,
            upload_date,
            view_count,
            transcript
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            video_record.youtube_video_id,
            channel_id,
            video_record.title,
            video_record.upload_date,
            video_record.view_count,
            video_record.transcript,
        ),
    )
    return int(cursor.lastrowid)


def update_video(
    connection: sqlite3.Connection,
    existing_row: sqlite3.Row,
    video_record: VideoRecord,
    channel_id: int | None,
) -> None:
    next_channel_id = channel_id if channel_id is not None else existing_row["channel_id"]
    next_title = video_record.title if video_record.title is not None else existing_row["title"]
    next_upload_date = video_record.upload_date if video_record.upload_date is not None else existing_row["upload_date"]
    next_view_count = video_record.view_count if video_record.view_count is not None else existing_row["view_count"]
    next_transcript = video_record.transcript if video_record.transcript is not None else existing_row["transcript"]
    connection.execute(
        """
        UPDATE videos
        SET channel_id = ?,
            title = ?,
            upload_date = ?,
            view_count = ?,
            transcript = ?
        WHERE id = ?
        """,
        (
            next_channel_id,
            next_title,
            next_upload_date,
            next_view_count,
            next_transcript,
            existing_row["id"],
        ),
    )
    if (
        next_title,
        next_upload_date,
        next_transcript,
    ) != (
        existing_row["title"],
        existing_row["upload_date"],
        existing_row["transcript"],
    ):
        connection.execute(
            "DELETE FROM video_embeddings WHERE video_id = ?",
            (existing_row["id"],),
        )


def store_channel_listing(
    connection: sqlite3.Connection,
    metadata: dict[str, Any],
) -> None:
    channel_identity = extract_channel_identity(metadata)
    upsert_channel(connection, channel_identity)
    for entry in metadata.get("entries") or []:
        upsert_video(
            connection,
            entry,
            channel_identity=channel_identity,
        )


def list_watchlist_channel_inputs(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT channels.youtube_channel_id, channels.channel_handle
        FROM channels_watchlist
        JOIN channels ON channels.id = channels_watchlist.channel_id
        ORDER BY COALESCE(channels.channel_handle, channels.youtube_channel_id)
        """
    ).fetchall()
    return [row["channel_handle"] or row["youtube_channel_id"] for row in rows if row["channel_handle"] or row["youtube_channel_id"]]


def list_blacklist_channel_identities(connection: sqlite3.Connection) -> list[ChannelIdentity]:
    rows = connection.execute(
        """
        SELECT channels.youtube_channel_id, channels.channel_handle
        FROM channels_blacklist
        JOIN channels ON channels.id = channels_blacklist.channel_id
        ORDER BY COALESCE(channels.channel_handle, channels.youtube_channel_id)
        """
    ).fetchall()
    return [
        ChannelIdentity(
            youtube_channel_id=row["youtube_channel_id"],
            channel_handle=row["channel_handle"],
        )
        for row in rows
    ]


def list_missing_watchlist_video_ids(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT videos.youtube_video_id
        FROM videos
        JOIN channels_watchlist ON channels_watchlist.channel_id = videos.channel_id
        WHERE videos.transcript IS NULL OR videos.transcript = ''
        ORDER BY videos.upload_date DESC, videos.id DESC
        """
    ).fetchall()
    return [row["youtube_video_id"] for row in rows]
