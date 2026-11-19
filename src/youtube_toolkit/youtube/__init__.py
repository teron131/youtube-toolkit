"""The only code that talks to YouTube: yt-dlp commands, caption downloads, comment samples, and channel listings, each fetched fresh."""

from .captions import (
    CaptionTranscriptUnavailable,
    extract_sponsorblock_intervals,
    fetch_caption_transcript,
)
from .channels import (
    DEFAULT_LAST_VIDEOS,
    ChannelOptions,
    cutoff_date_for_days,
    fetch_channel_videos,
)
from .client import (
    MAX_CONCURRENT_METADATA_FETCHES,
    YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
    YtDlpClient,
)
from .comments import DEFAULT_COMMENT_LIMIT, MAX_COMMENT_LIMIT, search_comments

__all__ = [
    "DEFAULT_COMMENT_LIMIT",
    "DEFAULT_LAST_VIDEOS",
    "MAX_COMMENT_LIMIT",
    "MAX_CONCURRENT_METADATA_FETCHES",
    "YTDLP_DOWNLOAD_TIMEOUT_SECONDS",
    "CaptionTranscriptUnavailable",
    "ChannelOptions",
    "YtDlpClient",
    "cutoff_date_for_days",
    "extract_sponsorblock_intervals",
    "fetch_caption_transcript",
    "fetch_channel_videos",
    "search_comments",
]
