"""YouTube acquisition adapters."""

from .caption import CaptionTranscriptUnavailable, fetch_caption_transcript
from .channels import ChannelOptions, fetch_channel_videos
from .client import YtDlpClient

__all__ = [
    "CaptionTranscriptUnavailable",
    "ChannelOptions",
    "YtDlpClient",
    "fetch_caption_transcript",
    "fetch_channel_videos",
]
