"""Public payload projections shared by storage readers and writers."""

from __future__ import annotations

from typing import Any

import turso


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
