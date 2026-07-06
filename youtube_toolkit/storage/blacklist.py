"""Blacklist channel membership persistence."""

from __future__ import annotations

from ..schemas import ChannelIdentity
from .channels import find_matching_channel_rows, upsert_channel
from .connection import LanceConnection, filter_in_values


def upsert_blacklist_channel(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> str | None:
    youtube_channel_id = upsert_channel(connection, channel_identity)
    if youtube_channel_id is None:
        return None
    connection.upsert_rows(
        "channels_blacklist",
        "channel_id",
        [{"channel_id": youtube_channel_id}],
    )
    return youtube_channel_id


def delete_blacklist_channel(
    connection: LanceConnection,
    channel_identity: ChannelIdentity,
) -> bool:
    """Remove a known channel from the blacklist membership table."""
    if not channel_identity.has_identity():
        return False

    matching_rows = find_matching_channel_rows(connection, channel_identity)
    channel_ids = [row["youtube_channel_id"] for row in matching_rows if row["youtube_channel_id"] is not None]
    if not channel_ids:
        return False

    existing_ids = {row["channel_id"] for row in connection.rows("channels_blacklist") if row["channel_id"] in set(channel_ids)}
    if not existing_ids:
        return False

    connection.delete_where(
        "channels_blacklist",
        f"channel_id IN ({filter_in_values(channel_ids)})",
    )
    return True


def list_blacklist_channel_identities(connection: LanceConnection) -> list[ChannelIdentity]:
    blacklisted_ids = {row["channel_id"] for row in connection.rows("channels_blacklist")}
    rows = sorted(
        [row for row in connection.rows("channels") if row["youtube_channel_id"] in blacklisted_ids],
        key=lambda row: row["channel_handle"] or row["youtube_channel_id"] or "",
    )
    return [
        ChannelIdentity(
            youtube_channel_id=row["youtube_channel_id"],
            channel_handle=row["channel_handle"],
        )
        for row in rows
    ]
