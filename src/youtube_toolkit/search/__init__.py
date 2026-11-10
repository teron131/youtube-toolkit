"""Video search across YouTube or named channels, ranked by metadata or semantically by caption transcripts."""

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
