"""Video and transcript-chunk search across YouTube or named channels: native mode ranks metadata, and semantic mode embeds fresh caption transcripts within the call."""

from .candidates import ChunkResult, VideoResult
from .request import ChannelTarget, SearchRequest
from .workflow import search_videos

__all__ = [
    "ChannelTarget",
    "ChunkResult",
    "SearchRequest",
    "VideoResult",
    "search_videos",
]
