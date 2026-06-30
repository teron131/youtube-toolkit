"""Channel identity persistence and duplicate-row merge policy."""

from __future__ import annotations

import turso

from ..schemas import ChannelIdentity


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
