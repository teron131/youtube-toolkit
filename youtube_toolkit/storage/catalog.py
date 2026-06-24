"""Local YouTube catalog persistence for channels, videos, watchlists, and blacklists."""

from __future__ import annotations

from typing import Any

import turso

from ..schemas import ChannelIdentity, VideoRecord, extract_channel_identity, extract_video_record


def upsert_channel(
    connection: turso.Connection,
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
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> list[turso.Row]:
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
    connection: turso.Connection,
    rows: list[turso.Row],
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


def upsert_blacklist_channel(
    connection: turso.Connection,
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
    return True


def delete_blacklist_channel(
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> bool:
    """Remove a known channel from the blacklist membership table."""
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
        FROM channels_blacklist
        WHERE channel_id IN ({placeholders})
        """,
        channel_ids,
    ).fetchall()
    if not membership_rows:
        return False

    connection.execute(
        f"""
        DELETE FROM channels_blacklist
        WHERE channel_id IN ({placeholders})
        """,
        channel_ids,
    )
    return True


def upsert_video(
    connection: turso.Connection,
    metadata: dict[str, Any],
    *,
    transcript: str | None = None,
    channel_identity: ChannelIdentity | None = None,
) -> dict[str, Any] | None:
    video_record = extract_video_record(metadata, transcript=transcript)
    if video_record is None:
        return None

    resolved_channel = channel_identity
    if resolved_channel is None or not resolved_channel.has_identity():
        resolved_channel = extract_channel_identity(metadata)
    channel_id = upsert_channel(connection, resolved_channel)
    existing_row = find_video_row(connection, video_record.youtube_video_id)
    if existing_row is None:
        row_id = insert_video(connection, video_record, channel_id)
        return find_video_payload(connection, row_id)

    update_video(connection, existing_row, video_record, channel_id)
    return find_video_payload(connection, int(existing_row["id"]))


def upsert_video_summary(
    connection: turso.Connection,
    youtube_video_id: str,
    summary: str,
) -> int | None:
    row = find_video_row(connection, youtube_video_id)
    if row is None:
        return None

    clean_summary = summary.strip()
    if not clean_summary:
        raise ValueError("Video summary cannot be empty")

    connection.execute(
        """
        UPDATE videos
        SET summary = ?
        WHERE id = ?
        """,
        (
            clean_summary,
            row["id"],
        ),
    )
    return int(row["id"])


def find_video_row(
    connection: turso.Connection,
    youtube_video_id: str,
) -> turso.Row | None:
    return connection.execute(
        """
        SELECT id, channel_id, title, upload_date, view_count, transcript, summary
        FROM videos
        WHERE youtube_video_id = ?
        """,
        (youtube_video_id,),
    ).fetchone()


def insert_video(
    connection: turso.Connection,
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


def find_video_payload(
    connection: turso.Connection,
    video_row_id: int,
) -> dict[str, Any] | None:
    """Return the caller-facing stored values for a video row."""
    row = connection.execute(
        """
        SELECT
            videos.id,
            videos.youtube_video_id,
            videos.title,
            videos.upload_date,
            videos.view_count,
            videos.transcript,
            channels.youtube_channel_id,
            channels.channel_handle
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.id = videos.channel_id
        WHERE videos.id = ?
        """,
        (video_row_id,),
    ).fetchone()
    if row is None:
        return None
    return stored_video_payload(row)


def stored_video_payload(row: turso.Row) -> dict[str, Any]:
    """Project a stored video row into the public payload shape."""
    return {
        "video_id": row["youtube_video_id"],
        "database_row_id": int(row["id"]),
        "title": row["title"],
        "upload_date": row["upload_date"],
        "view_count": row["view_count"],
        "channel_id": row["youtube_channel_id"],
        "channel_handle": row["channel_handle"],
        "transcript": row["transcript"],
    }


def update_video(
    connection: turso.Connection,
    existing_row: turso.Row,
    video_record: VideoRecord,
    channel_id: int | None,
) -> None:
    next_channel_id = keep_existing_if_missing(channel_id, existing_row["channel_id"])
    next_title = keep_existing_if_missing(video_record.title, existing_row["title"])
    next_upload_date = keep_existing_if_missing(video_record.upload_date, existing_row["upload_date"])
    next_view_count = keep_existing_if_missing(video_record.view_count, existing_row["view_count"])
    next_transcript = keep_existing_if_missing(video_record.transcript, existing_row["transcript"])
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
    if video_embedding_source_changed(
        existing_row,
        title=next_title,
        upload_date=next_upload_date,
        transcript=next_transcript,
    ):
        clear_video_embedding(connection, int(existing_row["id"]))
    if next_transcript != existing_row["transcript"]:
        clear_video_summary(connection, int(existing_row["id"]))


def keep_existing_if_missing[StoredValue](
    incoming_value: StoredValue | None,
    existing_value: StoredValue,
) -> StoredValue:
    """Use stored data when an incoming update omits a field."""
    return existing_value if incoming_value is None else incoming_value


def video_embedding_source_changed(
    existing_row: turso.Row,
    *,
    title: str | None,
    upload_date: str | None,
    transcript: str | None,
) -> bool:
    """Return whether video fields used for embeddings changed."""
    return (
        title,
        upload_date,
        transcript,
    ) != (
        existing_row["title"],
        existing_row["upload_date"],
        existing_row["transcript"],
    )


def clear_video_embedding(
    connection: turso.Connection,
    video_row_id: int,
) -> None:
    """Clear cached embedding data for a stored video."""
    connection.execute(
        """
        UPDATE videos
        SET embedding_content_hash = NULL,
            embedding = NULL
        WHERE id = ?
        """,
        (video_row_id,),
    )


def clear_video_summary(
    connection: turso.Connection,
    video_row_id: int,
) -> None:
    """Clear cached summary data for a stored video."""
    connection.execute(
        """
        UPDATE videos
        SET summary = NULL
        WHERE id = ?
        """,
        (video_row_id,),
    )


def store_channel_listing(
    connection: turso.Connection,
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


def list_watchlist_channel_inputs(connection: turso.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT channels.youtube_channel_id, channels.channel_handle
        FROM channels_watchlist
        JOIN channels ON channels.id = channels_watchlist.channel_id
        ORDER BY COALESCE(channels.channel_handle, channels.youtube_channel_id)
        """
    ).fetchall()
    return [row["channel_handle"] or row["youtube_channel_id"] for row in rows if row["channel_handle"] or row["youtube_channel_id"]]


def list_blacklist_channel_identities(connection: turso.Connection) -> list[ChannelIdentity]:
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


def list_missing_watchlist_video_ids(connection: turso.Connection) -> list[str]:
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
