"""Channel identity persistence and duplicate-row merge policy."""

from __future__ import annotations

import turso

from ..schemas import ChannelIdentity


def upsert_channel(
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> str | None:
    matching_rows = find_matching_channel_rows(connection, channel_identity)
    if channel_identity.youtube_channel_id is None:
        if not matching_rows:
            return None
        return matching_rows[0]["youtube_channel_id"]

    if matching_rows:
        return merge_channel_rows(connection, matching_rows, channel_identity)

    connection.execute(
        """
        INSERT INTO channels (youtube_channel_id, channel_handle)
        VALUES (?, ?)
        """,
        (channel_identity.youtube_channel_id, channel_identity.channel_handle),
    )
    return channel_identity.youtube_channel_id


def find_matching_channel_rows(
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> list[turso.Row]:
    return connection.execute(
        """
        SELECT youtube_channel_id, channel_handle
        FROM channels
        WHERE (? IS NOT NULL AND youtube_channel_id = ?)
            OR (? IS NOT NULL AND channel_handle = ?)
        ORDER BY youtube_channel_id
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

    connection.execute(
        """
        INSERT OR IGNORE INTO channels (youtube_channel_id)
        VALUES (?)
        """,
        (final_channel_id,),
    )

    for duplicate_row in rows:
        duplicate_channel_id = duplicate_row["youtube_channel_id"]
        if duplicate_channel_id is not None and duplicate_channel_id != final_channel_id:
            connection.execute(
                "UPDATE videos SET channel_id = ? WHERE channel_id = ?",
                (final_channel_id, duplicate_channel_id),
            )
            connection.execute(
                "UPDATE OR IGNORE channels_watchlist SET channel_id = ? WHERE channel_id = ?",
                (final_channel_id, duplicate_channel_id),
            )
            connection.execute(
                "DELETE FROM channels_watchlist WHERE channel_id = ?",
                (duplicate_channel_id,),
            )
            connection.execute(
                "UPDATE OR IGNORE channels_blacklist SET channel_id = ? WHERE channel_id = ?",
                (final_channel_id, duplicate_channel_id),
            )
            connection.execute(
                "DELETE FROM channels_blacklist WHERE channel_id = ?",
                (duplicate_channel_id,),
            )
            connection.execute(
                "DELETE FROM channels WHERE youtube_channel_id = ?",
                (duplicate_channel_id,),
            )

    if final_channel_handle is not None:
        connection.execute(
            """
            UPDATE channels
            SET channel_handle = ?
            WHERE youtube_channel_id = ?
            """,
            (
                final_channel_handle,
                final_channel_id,
            ),
        )

    if channel_embedding_source_changed(rows, final_channel_id, final_channel_handle):
        clear_channel_video_embeddings(connection, final_channel_id)

    return final_channel_id


def channel_embedding_source_changed(
    rows: list[turso.Row],
    final_channel_id: str,
    final_channel_handle: str | None,
) -> bool:
    return any(row["youtube_channel_id"] != final_channel_id for row in rows) or (
        final_channel_handle is not None and any(row["channel_handle"] != final_channel_handle for row in rows)
    )


def clear_channel_video_embeddings(
    connection: turso.Connection,
    youtube_channel_id: str,
) -> None:
    connection.execute(
        """
        UPDATE videos
        SET embedding_content_hash = NULL,
            embedding = NULL
        WHERE channel_id = ?
        """,
        (youtube_channel_id,),
    )
