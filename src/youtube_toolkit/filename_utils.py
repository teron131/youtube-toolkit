"""Names a video's files for its title and channel, the one naming rule that transcript exports and downloads share."""

from __future__ import annotations

import re
import unicodedata

from .schemas import VideoMetadata

UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_WHITESPACE = re.compile(r"\s+")
REPEATED_UNDERSCORES = re.compile(r"_+")


def filename_stem(metadata: VideoMetadata, label: str) -> str:
    """`TITLE_CHANNEL_LABEL`, leaving out empty parts; a video without a title goes by its ID."""
    parts = (
        _normalize_filename_component(metadata.title or "") or metadata.video_id,
        _normalize_filename_component(metadata.channel or ""),
        _normalize_filename_component(label),
    )
    return "_".join(part for part in parts if part)


def _normalize_filename_component(value: str) -> str:
    """Characters a filename cannot hold are removed and each run of whitespace becomes one underscore: `  TITLE: "SUBTITLE"  PART? ` → `TITLE_SUBTITLE_PART`."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = UNSAFE_FILENAME_CHARACTERS.sub("", normalized)
    normalized = FILENAME_WHITESPACE.sub("_", normalized)
    normalized = REPEATED_UNDERSCORES.sub("_", normalized)
    return normalized.strip(" ._")
