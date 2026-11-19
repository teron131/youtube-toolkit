"""Pick a video's most informative keyframes and lay them out as collages, and write a video's distinct frames, keyframes, or single frames as JPEGs for frame downloads."""

from .candidates import collect_candidates, save_frame
from .workflow import sample_vision

__all__ = [
    "collect_candidates",
    "sample_vision",
    "save_frame",
]
