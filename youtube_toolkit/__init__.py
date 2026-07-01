"""Library-first YouTube toolkit."""

from .convert_zh import convert_zh
from .identity import ChannelRef, ChannelRefInput, VideoRef, VideoRefInput, YouTubeRefs, extract_video_id, normalize_youtube_url
from .model_config import EmbeddingConfig, ModelConfig
from .schemas import TranscriptDocument, TranscriptSection, VideoMetadata
from .youtube import SearchOptions, TranscriptOptions, fetch_transcript, search_videos

__all__ = [
    "ChannelRef",
    "ChannelRefInput",
    "EmbeddingConfig",
    "ModelConfig",
    "SearchOptions",
    "TranscriptDocument",
    "TranscriptOptions",
    "TranscriptSection",
    "VideoMetadata",
    "VideoRef",
    "VideoRefInput",
    "YouTubeRefs",
    "convert_zh",
    "extract_video_id",
    "fetch_transcript",
    "normalize_youtube_url",
    "search_videos",
]
