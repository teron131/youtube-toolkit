"""Blacklist channel membership persistence."""

from __future__ import annotations

import turso

from ..schemas import ChannelIdentity
from .channels import find_matching_channel_rows, upsert_channel


def upsert_blacklist_channel(
    connection: turso.Connection,
    channel_identity: ChannelIdentity,
) -> str | None:
    youtube_channel_id = upsert_channel(connection, channel_identity)
    if youtube_channel_id is None:
        return None
    connection.execute(
        """
        INSERT OR IGNORE INTO channels_blacklist (channel_id)
        VALUES (?)
        """,
        (youtube_channel_id,),
    )
    return youtube_channel_id


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

    channel_ids = [row["youtube_channel_id"] for row in matching_rows if row["youtube_channel_id"] is not None]
    if not channel_ids:
        return False
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


def list_blacklist_channel_identities(connection: turso.Connection) -> list[ChannelIdentity]:
    rows = connection.execute(
        """
        SELECT channels.youtube_channel_id, channels.channel_handle
        FROM channels_blacklist
        JOIN channels ON channels.youtube_channel_id = channels_blacklist.channel_id
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
