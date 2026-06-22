"""yt-dlp adapter package."""

from .audio import AudioOptions, download_transcription_audio
from .caption import TranscriptOptions, fetch_transcript
from .channels import ChannelOptions, fetch_channel_videos
from .client import SEARCH_SORT_PARAMS, YtDlpClient, build_ffmpeg_audio_args
from .search import SearchOptions, search_videos

__all__ = [
    "SEARCH_SORT_PARAMS",
    "AudioOptions",
    "ChannelOptions",
    "SearchOptions",
    "TranscriptOptions",
    "YtDlpClient",
    "build_ffmpeg_audio_args",
    "download_transcription_audio",
    "fetch_channel_videos",
    "fetch_transcript",
    "search_videos",
]
