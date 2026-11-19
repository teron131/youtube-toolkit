"""The name cleanup that video folders, saved entries, and downloads share, and download names built from the title and channel."""

from __future__ import annotations

import re
import unicodedata

from .schemas import VideoMetadata

REPEATED_UNDERSCORES = re.compile(r"_+")
# Linux caps a file name at 255 bytes and scripts such as Chinese take three a character, so the title and channel are cut by bytes, leaving room for the longest label and the names yt-dlp gives its partial files.
TITLE_BYTES = 120
CHANNEL_BYTES = 60


def filename_stem(metadata: VideoMetadata, label: str) -> str:
    """`TITLE_CHANNEL_LABEL`, leaving out empty parts; a video without a title goes by its ID.

    The label is the toolkit's own, such as `1080p_00h01m02.812s`, so it stays as given and keeps its time's dot.
    """
    parts = (
        _clipped(normalize_filename_component(metadata.title or ""), TITLE_BYTES)
        or metadata.video_id,
        _clipped(normalize_filename_component(metadata.channel or ""), CHANNEL_BYTES),
        label,
    )
    return "_".join(part for part in parts if part)


def normalize_filename_component(value: str) -> str:
    """Every character but letters, marks, and digits of any script becomes an underscore, runs of them one, so a name is legal on every system, safe unquoted in a shell, and splits into words at underscores alone: `  GPT-5.6 vs. o3: "Live" [4K]? ` → `GPT_5_6_vs_o3_Live_4K`."""
    replaced = "".join(
        character if unicodedata.category(character)[0] in "LMN" else "_"
        for character in unicodedata.normalize("NFKC", value)
    )
    return REPEATED_UNDERSCORES.sub("_", replaced).strip("_")


def _clipped(text: str, limit: int) -> str:
    """At most limit UTF-8 bytes, never half a character."""
    return text.encode()[:limit].decode(errors="ignore").rstrip("_")
