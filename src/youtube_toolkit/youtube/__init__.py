"""The only code that talks to YouTube: yt-dlp commands, caption downloads, comment samples, and channel listings, each fetched fresh."""

from .caption import CaptionTranscriptUnavailable, fetch_caption_transcript
from .channels import ChannelOptions, fetch_channel_videos
from .client import MAX_COMMENT_LIMIT, YtDlpClient
from .comments import DEFAULT_COMMENT_LIMIT, search_comments

__all__ = [
    "DEFAULT_COMMENT_LIMIT",
    "MAX_COMMENT_LIMIT",
    "CaptionTranscriptUnavailable",
    "ChannelOptions",
    "YtDlpClient",
    "fetch_caption_transcript",
    "fetch_channel_videos",
    "search_comments",
]
