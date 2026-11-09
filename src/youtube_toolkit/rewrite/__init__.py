"""Transcript rewrites through the LLM: a translation or a cleaned-up copy, either line by line on the captions' own timestamps or as an article of paragraphs under the video's chapter titles."""

from .requests import RewriteError
from .workflow import rewrite_transcript

__all__ = [
    "RewriteError",
    "rewrite_transcript",
]
