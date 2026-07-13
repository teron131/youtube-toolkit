"""Video persistence, sparse update, and derived-cache invalidation policy."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from typing import Any

from ..schemas import ChannelIdentity, VideoRecord, extract_channel_identity, extract_video_record
from .channels import upsert_channel
from .connection import LanceConnection, filter_literal, transcript_chunk_metadata
from .payloads import stored_video_payload

VIDEO_AGENT_EDITABLE_FIELDS = (
    "transcript",
    "summary",
)
VIDEO_EMBEDDING_SOURCE_FIELDS = ("channel_id", "title", "transcript")
VIDEO_CHUNK_METADATA_FIELDS = (
    "channel_id",
    "title",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
)
VIDEO_DEPENDENT_TABLE_KEYS = (
    ("watchlist_videos", "video_id"),
    ("video_embeddings", "video_id"),
    ("video_title_embeddings", "video_id"),
    ("video_transcript_chunks", "video_id"),
)


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

    connection.update_where(
        "videos",
        f"youtube_video_id = {filter_literal(youtube_video_id)}",
        {"summary": clean_summary},
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


def get_video_record(
    connection: LanceConnection,
    youtube_video_id: str,
) -> dict[str, Any]:
    """Return one complete catalog record, including summary text, or fail for an unknown video."""
    row = require_video_row(connection, youtube_video_id)
    payload = stored_video_payload(video_payload_row(connection, row))
    return {
        **payload,
        "summary": row["summary"],
    }


def edit_video_content(
    connection: LanceConnection,
    youtube_video_id: str,
    values: Mapping[str, Any],
    *,
    clear_fields: Collection[str] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Edit agent-owned video content while rejecting provider-owned metadata changes."""
    existing_row = require_video_row(connection, youtube_video_id)
    unknown_fields = set(values) - set(VIDEO_AGENT_EDITABLE_FIELDS)
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unsupported agent-owned video fields: {unknown}")
    normalized_values: dict[str, str] = {}
    for field, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Video {field} cannot be empty; use --clear {field}")
        normalized_values[field] = value.strip()
    normalized_clear_fields = set(clear_fields)
    unknown_clear_fields = normalized_clear_fields - set(VIDEO_AGENT_EDITABLE_FIELDS)
    if unknown_clear_fields:
        unknown = ", ".join(sorted(unknown_clear_fields))
        raise ValueError(f"Unsupported agent-owned video fields to clear: {unknown}")
    conflicting_fields = set(normalized_values) & normalized_clear_fields
    if conflicting_fields:
        conflicts = ", ".join(sorted(conflicting_fields))
        raise ValueError(f"Cannot update and clear the same video fields: {conflicts}")
    if not normalized_values and not normalized_clear_fields:
        raise ValueError("Video edit requires at least one field change")

    changes = {
        **normalized_values,
        **dict.fromkeys(normalized_clear_fields),
    }
    proposed_transcript = changes.get("transcript", existing_row["transcript"])
    if proposed_transcript != existing_row["transcript"] and "summary" not in changes:
        changes["summary"] = None

    updated_row, invalidated = apply_video_changes(
        connection,
        existing_row,
        changes,
        dry_run=dry_run,
    )
    changed_fields = {
        field: {
            "before_chars": len(existing_row[field] or ""),
            "after_chars": len(updated_row[field] or ""),
        }
        for field in VIDEO_AGENT_EDITABLE_FIELDS
        if updated_row[field] != existing_row[field]
    }
    cleared_fields = [field for field in VIDEO_AGENT_EDITABLE_FIELDS if existing_row[field] is not None and updated_row[field] is None]
    return {
        "video_id": youtube_video_id,
        "changed": changed_fields,
        "cleared": cleared_fields,
        "invalidated": invalidated,
        "dry_run": dry_run,
    }


def delete_video_record(
    connection: LanceConnection,
    youtube_video_id: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete one catalog video and its owned dependent rows while preserving channel identity."""
    require_video_row(connection, youtube_video_id)
    row_counts = {
        table_name: connection.count_rows(
            table_name,
            where_sql=f"{key} = {filter_literal(youtube_video_id)}",
        )
        for table_name, key in (*VIDEO_DEPENDENT_TABLE_KEYS, ("videos", "youtube_video_id"))
    }
    if not dry_run:
        for table_name, key in VIDEO_DEPENDENT_TABLE_KEYS:
            connection.delete_where(
                table_name,
                f"{key} = {filter_literal(youtube_video_id)}",
            )
        connection.delete_where(
            "videos",
            f"youtube_video_id = {filter_literal(youtube_video_id)}",
        )
    return {
        "video_id": youtube_video_id,
        "deleted": not dry_run,
        "rows": row_counts,
        "dry_run": dry_run,
    }


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
    changes = {
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
    if next_transcript != existing_row["transcript"]:
        changes["summary"] = None
    updated_row, _ = apply_video_changes(connection, existing_row, changes)
    if next_transcript == existing_row["transcript"] and not any(existing_row[field] != updated_row[field] for field in VIDEO_CHUNK_METADATA_FIELDS):
        refresh_video_transcript_chunk_metadata(connection, updated_row)


def apply_video_changes(
    connection: LanceConnection,
    existing_row: dict[str, Any],
    changes: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Apply one prepared row change and own every derived-data invalidation caused by it."""
    updated_row = {
        **existing_row,
        **changes,
    }
    if updated_row == existing_row:
        return updated_row, []

    invalidated = video_invalidations(existing_row, updated_row)
    if dry_run:
        return updated_row, invalidated

    connection.upsert_rows(
        "videos",
        "youtube_video_id",
        [updated_row],
    )
    youtube_video_id = existing_row["youtube_video_id"]
    if "video_embedding" in invalidated:
        clear_video_embedding(connection, youtube_video_id)
    if "video_title_embedding" in invalidated:
        clear_video_title_embedding(connection, youtube_video_id)
    if "transcript_chunks" in invalidated:
        clear_video_transcript_chunks(connection, youtube_video_id)
    elif any(existing_row[field] != updated_row[field] for field in VIDEO_CHUNK_METADATA_FIELDS):
        refresh_video_transcript_chunk_metadata(connection, updated_row)
    return updated_row, invalidated


def video_invalidations(
    existing_row: Mapping[str, Any],
    updated_row: Mapping[str, Any],
) -> list[str]:
    """Describe the derived values made stale by one proposed video row change."""
    invalidated: list[str] = []
    if any(existing_row[field] != updated_row[field] for field in VIDEO_EMBEDDING_SOURCE_FIELDS):
        invalidated.append("video_embedding")
    if existing_row["title"] != updated_row["title"]:
        invalidated.append("video_title_embedding")
    if existing_row["transcript"] != updated_row["transcript"]:
        invalidated.append("transcript_chunks")
        if existing_row["summary"] is not None and updated_row["summary"] is None:
            invalidated.append("summary")
    return invalidated


def require_video_row(
    connection: LanceConnection,
    youtube_video_id: str,
) -> dict[str, Any]:
    """Return one raw catalog row or raise the agent-facing missing-record error."""
    row = find_video_row(connection, youtube_video_id)
    if row is None:
        raise ValueError(f"Video is not stored: {youtube_video_id}")
    return row


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


def refresh_video_transcript_chunk_metadata(
    connection: LanceConnection,
    video: dict[str, Any],
) -> None:
    """Refresh derived chunk filter columns without changing chunk embeddings."""
    metadata = transcript_chunk_metadata(video_payload_row(connection, video))
    connection.update_where(
        "video_transcript_chunks",
        f"video_id = {filter_literal(video['youtube_video_id'])}",
        metadata,
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


def clear_video_title_embedding(
    connection: LanceConnection,
    youtube_video_id: str,
) -> None:
    """Clear cached title embedding data for a stored video."""
    connection.delete_where(
        "video_title_embeddings",
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
