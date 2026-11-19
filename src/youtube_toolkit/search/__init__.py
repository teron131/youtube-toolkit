"""Video search across YouTube or named channels, ranked by metadata or semantically by caption transcripts."""

from .candidates import ChunkResult, VideoResult
from .request import (
    DEFAULT_CHANNEL_SEARCH_LIMIT,
    DEFAULT_YOUTUBE_SEARCH_LIMIT,
    ChannelTarget,
    SearchRequest,
    parse_duration,
)
from .workflow import search_videos

__all__ = [
    "DEFAULT_CHANNEL_SEARCH_LIMIT",
    "DEFAULT_YOUTUBE_SEARCH_LIMIT",
    "ChannelTarget",
    "ChunkResult",
    "SearchRequest",
    "VideoResult",
    "parse_duration",
    "search_videos",
]
