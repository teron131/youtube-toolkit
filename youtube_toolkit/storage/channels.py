"""Channel identity persistence and duplicate-row merge policy."""

from __future__ import annotations

from typing import Any

from ..schemas import ChannelIdentity
from .connection import LanceConnection, filter_in_values


def upsert_channel(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> str | None:
    matching_rows = find_matching_channel_rows(connection, channel_identity)
    if channel_identity.youtube_channel_id is None:
        if not matching_rows:
            return None
        return matching_rows[0]["youtube_channel_id"]

    if matching_rows:
        return merge_channel_rows(connection, matching_rows, channel_identity)

    connection.upsert_rows(
        "channels",
        "youtube_channel_id",
        [
            {
                "youtube_channel_id": channel_identity.youtube_channel_id,
                "channel_handle": channel_identity.channel_handle,
            }
        ],
    )
    return channel_identity.youtube_channel_id


def find_matching_channel_rows(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in connection.rows("channels")
            if (channel_identity.youtube_channel_id is not None and row["youtube_channel_id"] == channel_identity.youtube_channel_id)
            or (channel_identity.channel_handle is not None and row["channel_handle"] == channel_identity.channel_handle)
        ],
        key=lambda row: row["youtube_channel_id"] or "",
    )


def merge_channel_rows(
    connection: LanceConnection,
    rows: list[dict[str, Any]],
    channel_identity: ChannelIdentity,
) -> str:
    final_channel_id = channel_identity.youtube_channel_id or next(
        (row["youtube_channel_id"] for row in rows if row["youtube_channel_id"] is not None),
        None,
    )
    if final_channel_id is None:
        raise ValueError("Channel merge requires a YouTube channel ID")
    final_channel_handle = channel_identity.channel_handle or next(
        (row["channel_handle"] for row in rows if row["channel_handle"] is not None),
        None,
    )

    duplicate_channel_ids = [row["youtube_channel_id"] for row in rows if row["youtube_channel_id"] is not None and row["youtube_channel_id"] != final_channel_id]
    if duplicate_channel_ids:
        merge_duplicate_channel_references(
            connection,
            final_channel_id=final_channel_id,
            duplicate_channel_ids=duplicate_channel_ids,
        )

    connection.upsert_rows(
        "channels",
        "youtube_channel_id",
        [
            {
                "youtube_channel_id": final_channel_id,
                "channel_handle": final_channel_handle,
            }
        ],
    )
    if duplicate_channel_ids:
        connection.delete_where(
            "channels",
            f"youtube_channel_id IN ({filter_in_values(duplicate_channel_ids)})",
        )

    if channel_embedding_source_changed(rows, final_channel_id, final_channel_handle):
        clear_channel_video_embeddings(connection, final_channel_id)

    return final_channel_id


def merge_duplicate_channel_references(
    connection: LanceConnection,
    *,
    final_channel_id: str,
    duplicate_channel_ids: list[str],
) -> None:
    duplicate_ids = set(duplicate_channel_ids)
    videos = [
        {
            **row,
            "channel_id": final_channel_id,
        }
        for row in connection.rows("videos")
        if row["channel_id"] in duplicate_ids
    ]
    connection.upsert_rows("videos", "youtube_video_id", videos)
    for table_name in ("channels_watchlist", "channels_blacklist"):
        existing_ids = {row["channel_id"] for row in connection.rows(table_name)}
        if existing_ids & duplicate_ids:
            connection.upsert_rows(table_name, "channel_id", [{"channel_id": final_channel_id}])
            connection.delete_where(
                table_name,
                f"channel_id IN ({filter_in_values(duplicate_channel_ids)})",
            )


def channel_embedding_source_changed(
    rows: list[dict[str, Any]],
    final_channel_id: str,
    final_channel_handle: str | None,
) -> bool:
    return any(row["youtube_channel_id"] != final_channel_id for row in rows) or (
        final_channel_handle is not None and any(row["channel_handle"] != final_channel_handle for row in rows)
    )


def clear_channel_video_embeddings(
    connection: LanceConnection,
    youtube_channel_id: str,
) -> None:
    video_ids = [row["youtube_video_id"] for row in connection.rows("videos") if row["channel_id"] == youtube_channel_id]
    if not video_ids:
        return
    connection.delete_where(
        "video_embeddings",
        f"video_id IN ({filter_in_values(video_ids)})",
    )
