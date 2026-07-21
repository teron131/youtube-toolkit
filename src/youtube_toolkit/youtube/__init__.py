"""YouTube acquisition package backed by yt-dlp at the adapter edge."""

from .caption import TranscriptOptions, fetch_transcript
from .channels import ChannelOptions, fetch_channel_videos
from .client import YtDlpClient
from .vision import sample_vision

__all__ = [
    "ChannelOptions",
    "TranscriptOptions",
    "YtDlpClient",
    "fetch_channel_videos",
    "fetch_transcript",
    "sample_vision",
]
