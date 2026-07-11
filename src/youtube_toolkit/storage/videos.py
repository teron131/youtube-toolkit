"""Video persistence, sparse update, and derived-cache invalidation policy."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..schemas import ChannelIdentity, VideoRecord, extract_channel_identity, extract_video_record
from .channels import upsert_channel
from .connection import LanceConnection, filter_literal, transcript_chunk_metadata
from .payloads import stored_video_payload


def upsert_video(
    connection: LanceConnection,
    metadata: dict[str, Any],
    *,
    transcript: str | None = None,
    channel_identity: ChannelIdentity | None = None,
) -> dict[str, Any] | None:
    """Upsert sparse video metadata while preserving omitted stored values and invalidating derived caches."""
    video_record = extract_video_record(metadata, transcript=transcript)
    if video_record is None:
        return None

    metadata_fetched_at = current_utc_isoformat() if has_metadata_fields(metadata) else None
    transcript_fetched_at = current_utc_isoformat() if transcript is not None else None
    resolved_channel = channel_identity
    if resolved_channel is None or not resolved_channel.has_identity():
        resolved_channel = extract_channel_identity(metadata)
    youtube_channel_id = upsert_channel(connection, resolved_channel)
    existing_row = find_video_row(connection, video_record.youtube_video_id)
    if existing_row is None:
        connection.upsert_rows(
            "videos",
            "youtube_video_id",
            [
                {
                    "youtube_video_id": video_record.youtube_video_id,
                    "channel_id": youtube_channel_id,
                    "title": video_record.title,
                    "upload_date": video_record.upload_date,
                    "duration": video_record.duration,
                    "view_count": video_record.view_count,
                    "like_count": video_record.like_count,
                    "transcript": video_record.transcript,
                    "summary": None,
                    "metadata_fetched_at": metadata_fetched_at,
                    "transcript_fetched_at": transcript_fetched_at,
                }
            ],
        )
        return find_video_payload(connection, video_record.youtube_video_id)

    update_video(
        connection,
        existing_row,
        video_record,
        youtube_channel_id,
        metadata_fetched_at=metadata_fetched_at,
        transcript_fetched_at=transcript_fetched_at,
    )
    return find_video_payload(connection, video_record.youtube_video_id)


def upsert_video_summary(
    connection: LanceConnection,
    youtube_video_id: str,
    summary: str,
) -> str | None:
    """Store a non-empty summary only for videos already present in the catalog."""
    row = find_video_row(connection, youtube_video_id)
    if row is None:
        return None

    clean_summary = summary.strip()
    if not clean_summary:
        raise ValueError("Video summary cannot be empty")

    connection.upsert_rows(
        "videos",
        "youtube_video_id",
        [{**row, "summary": clean_summary}],
    )
    return youtube_video_id


def find_video_row(
    connection: LanceConnection,
    youtube_video_id: str,
) -> dict[str, Any] | None:
    """Return the raw stored video row keyed by YouTube video ID."""
    return connection.find_row(
        "videos",
        "youtube_video_id",
        youtube_video_id,
    )


def find_video_payload(
    connection: LanceConnection,
    youtube_video_id: str,
) -> dict[str, Any] | None:
    """Return the caller-facing stored values for a video row."""
    row = find_video_row(connection, youtube_video_id)
    if row is None:
        return None
    return stored_video_payload(video_payload_row(connection, row))


def update_video(
    connection: LanceConnection,
    existing_row: dict[str, Any],
    video_record: VideoRecord,
    youtube_channel_id: str | None,
    *,
    metadata_fetched_at: str | None,
    transcript_fetched_at: str | None,
) -> None:
    """Apply sparse video updates and clear caches whose source fields changed."""
    next_channel_id = keep_existing_if_missing(youtube_channel_id, existing_row["channel_id"])
    next_title = keep_existing_if_missing(video_record.title, existing_row["title"])
    next_upload_date = keep_existing_if_missing(video_record.upload_date, existing_row["upload_date"])
    next_duration = keep_existing_if_missing(video_record.duration, existing_row["duration"])
    next_view_count = keep_existing_if_missing(video_record.view_count, existing_row["view_count"])
    next_like_count = keep_existing_if_missing(video_record.like_count, existing_row["like_count"])
    next_transcript = keep_existing_if_missing(video_record.transcript, existing_row["transcript"])
    next_metadata_fetched_at = keep_existing_if_missing(metadata_fetched_at, existing_row["metadata_fetched_at"])
    next_transcript_fetched_at = keep_existing_if_missing(transcript_fetched_at, existing_row["transcript_fetched_at"])
    connection.upsert_rows(
        "videos",
        "youtube_video_id",
        [
            {
                **existing_row,
                "channel_id": next_channel_id,
                "title": next_title,
                "upload_date": next_upload_date,
                "duration": next_duration,
                "view_count": next_view_count,
                "like_count": next_like_count,
                "transcript": next_transcript,
                "metadata_fetched_at": next_metadata_fetched_at,
                "transcript_fetched_at": next_transcript_fetched_at,
            }
        ],
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
        clear_video_transcript_chunks(connection, existing_row["youtube_video_id"])
    else:
        updated_row = {
            **existing_row,
            "channel_id": next_channel_id,
            "title": next_title,
            "upload_date": next_upload_date,
            "duration": next_duration,
            "view_count": next_view_count,
            "like_count": next_like_count,
        }
        refresh_video_transcript_chunk_metadata(connection, updated_row)


def video_payload_row(
    connection: LanceConnection,
    video: dict[str, Any],
) -> dict[str, Any]:
    """Join channel handle data onto a stored video row for payload projection."""
    channel = connection.find_row("channels", "youtube_channel_id", video["channel_id"])
    return {
        **video,
        "channel_handle": channel["channel_handle"] if channel is not None else None,
    }


def has_metadata_fields(metadata: dict[str, Any]) -> bool:
    """Return whether incoming data represents a metadata fetch, not just an ID shell."""
    return any(key != "id" and metadata.get(key) is not None for key in metadata)


def current_utc_isoformat() -> str:
    """Return the storage timestamp format for video fetch boundaries."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def keep_existing_if_missing[StoredValue](
    incoming_value: StoredValue | None,
    existing_value: StoredValue,
) -> StoredValue:
    """Use stored data when an incoming update omits a field."""
    return existing_value if incoming_value is None else incoming_value


def video_embedding_source_changed(
    existing_row: dict[str, Any],
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


def refresh_video_transcript_chunk_metadata(
    connection: LanceConnection,
    video: dict[str, Any],
) -> None:
    """Refresh derived chunk filter columns without changing chunk embeddings."""
    rows = connection.rows(
        "video_transcript_chunks",
        where_sql=f"video_id = {filter_literal(video['youtube_video_id'])}",
    )
    if not rows:
        return
    metadata = transcript_chunk_metadata(video_payload_row(connection, video))
    connection.upsert_rows(
        "video_transcript_chunks",
        [
            "video_id",
            "chunk_index",
        ],
        [
            {
                **row,
                **metadata,
            }
            for row in rows
        ],
    )


def clear_video_embedding(
    connection: LanceConnection,
    youtube_video_id: str,
) -> None:
    """Clear cached embedding data for a stored video."""
    connection.delete_where(
        "video_embeddings",
        f"video_id = {filter_literal(youtube_video_id)}",
    )


def clear_video_transcript_chunks(
    connection: LanceConnection,
    youtube_video_id: str,
) -> None:
    """Clear transcript chunk embeddings for a stored video."""
    connection.delete_where(
        "video_transcript_chunks",
        f"video_id = {filter_literal(youtube_video_id)}",
    )


def clear_video_summary(
    connection: LanceConnection,
    youtube_video_id: str,
) -> None:
    """Clear cached summary data for a stored video."""
    row = find_video_row(connection, youtube_video_id)
    if row is None:
        return
    connection.upsert_rows(
        "videos",
        "youtube_video_id",
        [{**row, "summary": None}],
    )


def store_channel_listing(
    connection: LanceConnection,
    metadata: dict[str, Any],
) -> None:
    """Persist channel listing metadata without marking videos as watchlist discoveries."""
    channel_identity = extract_channel_identity(metadata)
    upsert_channel(connection, channel_identity)
    for entry in metadata.get("entries") or []:
        upsert_video(
            connection,
            entry,
            channel_identity=channel_identity,
        )
