"""YouTube acquisition package backed by yt-dlp at the adapter edge."""

from .caption import TranscriptOptions, fetch_transcript
from .channels import ChannelOptions, fetch_channel_videos
from .client import YtDlpClient

__all__ = [
    "ChannelOptions",
    "TranscriptOptions",
    "YtDlpClient",
    "fetch_channel_videos",
    "fetch_transcript",
]
