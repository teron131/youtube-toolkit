"""Curate screenshots for visual context: rank a video's keyframes against its transcript with image embeddings, or spread picks over time slots without them, then lay them out as timestamped collages."""

from .workflow import sample_vision

__all__ = [
    "sample_vision",
]
