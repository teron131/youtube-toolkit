"""Video persistence, sparse update, and derived-cache invalidation policy."""

from __future__ import annotations

from typing import Any

import turso

from ..schemas import ChannelIdentity, VideoRecord, extract_channel_identity, extract_video_record
from .channels import upsert_channel
from .payloads import stored_video_payload


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
