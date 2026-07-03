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
    youtube_channel_id = upsert_channel(connection, resolved_channel)
    existing_row = find_video_row(connection, video_record.youtube_video_id)
    if existing_row is None:
        insert_video(connection, video_record, youtube_channel_id)
        return find_video_payload(connection, video_record.youtube_video_id)

    update_video(connection, existing_row, video_record, youtube_channel_id)
    return find_video_payload(connection, video_record.youtube_video_id)


def upsert_video_summary(
    connection: turso.Connection,
    youtube_video_id: str,
    summary: str,
) -> str | None:
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
        WHERE youtube_video_id = ?
        """,
        (
            clean_summary,
            youtube_video_id,
        ),
    )
    return youtube_video_id


def find_video_row(
    connection: turso.Connection,
    youtube_video_id: str,
) -> turso.Row | None:
    return connection.execute(
        """
        SELECT youtube_video_id, channel_id, title, upload_date, transcript, summary
        FROM videos
        WHERE youtube_video_id = ?
        """,
        (youtube_video_id,),
    ).fetchone()


def insert_video(
    connection: turso.Connection,
    video_record: VideoRecord,
    youtube_channel_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO videos (
            youtube_video_id,
            channel_id,
            title,
            upload_date,
            transcript
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            video_record.youtube_video_id,
            youtube_channel_id,
            video_record.title,
            video_record.upload_date,
            video_record.transcript,
        ),
    )


def find_video_payload(
    connection: turso.Connection,
    youtube_video_id: str,
) -> dict[str, Any] | None:
    """Return the caller-facing stored values for a video row."""
    row = connection.execute(
        """
        SELECT
            videos.youtube_video_id,
            videos.channel_id,
            videos.title,
            videos.upload_date,
            videos.transcript,
            channels.channel_handle
        FROM videos AS videos
        LEFT JOIN channels AS channels ON channels.youtube_channel_id = videos.channel_id
        WHERE videos.youtube_video_id = ?
        """,
        (youtube_video_id,),
    ).fetchone()
    if row is None:
        return None
    return stored_video_payload(row)


def update_video(
    connection: turso.Connection,
    existing_row: turso.Row,
    video_record: VideoRecord,
    youtube_channel_id: str | None,
) -> None:
    next_channel_id = keep_existing_if_missing(youtube_channel_id, existing_row["channel_id"])
    next_title = keep_existing_if_missing(video_record.title, existing_row["title"])
    next_upload_date = keep_existing_if_missing(video_record.upload_date, existing_row["upload_date"])
    next_transcript = keep_existing_if_missing(video_record.transcript, existing_row["transcript"])
    connection.execute(
        """
        UPDATE videos
        SET channel_id = ?,
            title = ?,
            upload_date = ?,
            transcript = ?
        WHERE youtube_video_id = ?
        """,
        (
            next_channel_id,
            next_title,
            next_upload_date,
            next_transcript,
            existing_row["youtube_video_id"],
        ),
    )
    if video_embedding_source_changed(
        existing_row,
        channel_id=next_channel_id,
        title=next_title,
        transcript=next_transcript,
    ):
        clear_video_embedding(connection, existing_row["youtube_video_id"])
    if next_transcript != existing_row["transcript"]:
        clear_video_summary(connection, existing_row["youtube_video_id"])


def keep_existing_if_missing[StoredValue](
    incoming_value: StoredValue | None,
    existing_value: StoredValue,
) -> StoredValue:
    """Use stored data when an incoming update omits a field."""
    return existing_value if incoming_value is None else incoming_value


def video_embedding_source_changed(
    existing_row: turso.Row,
    *,
    channel_id: str | None,
    title: str | None,
    transcript: str | None,
) -> bool:
    """Return whether video fields used for embeddings changed."""
    return (
        channel_id,
        title,
        transcript,
    ) != (
        existing_row["channel_id"],
        existing_row["title"],
        existing_row["transcript"],
    )


def clear_video_embedding(
    connection: turso.Connection,
    youtube_video_id: str,
) -> None:
    """Clear cached embedding data for a stored video."""
    connection.execute(
        """
        UPDATE videos
        SET embedding_content_hash = NULL,
            embedding = NULL
        WHERE youtube_video_id = ?
        """,
        (youtube_video_id,),
    )


def clear_video_summary(
    connection: turso.Connection,
    youtube_video_id: str,
) -> None:
    """Clear cached summary data for a stored video."""
    connection.execute(
        """
        UPDATE videos
        SET summary = NULL
        WHERE youtube_video_id = ?
        """,
        (youtube_video_id,),
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
