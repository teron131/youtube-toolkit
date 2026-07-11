"""Public payload projections shared by storage readers and writers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def stored_video_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a stored video row into the public payload shape."""
    return {
        "video_id": row["youtube_video_id"],
        "title": row["title"],
        "upload_date": row["upload_date"],
        "duration": row["duration"],
        "view_count": row["view_count"],
        "like_count": row["like_count"],
        "channel_id": row["channel_id"],
        "channel_handle": row["channel_handle"],
        "transcript": row["transcript"],
        "metadata_fetched_at": row["metadata_fetched_at"],
        "transcript_fetched_at": row["transcript_fetched_at"],
    }
