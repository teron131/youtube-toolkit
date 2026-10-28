"""Lay selected frames out as timestamped 2x3 JPEG pages sized for vision models, and publish them as the video's only page set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..ffmpeg_utils import FRAME_HEIGHT, FRAME_WIDTH
from .candidates import DecodedFrame

TIMESTAMP_FOOTER_HEIGHT = 24
COLLAGE_PADDING = 6
# A page must stay within vision models' per-image limit (Claude: 2576 px long edge, about 3.75 MP), or the model shrinks every tile.
COLLAGE_COLUMNS = 2
COLLAGE_ROWS = 3
FRAMES_PER_COLLAGE = COLLAGE_COLUMNS * COLLAGE_ROWS
JPEG_QUALITY = 80
JPEG_SUBSAMPLING = "4:2:2"
TIMESTAMP_FONT = ImageFont.load_default(size=16)


def render_timestamped_collages(
    frames: list[DecodedFrame], output_stem: Path
) -> list[Path]:
    tile_height = FRAME_HEIGHT + TIMESTAMP_FOOTER_HEIGHT
    collage_width = FRAME_WIDTH * COLLAGE_COLUMNS + COLLAGE_PADDING * (
        COLLAGE_COLUMNS + 1
    )
    collage_height = tile_height * COLLAGE_ROWS + COLLAGE_PADDING * (COLLAGE_ROWS + 1)
    output_paths: list[Path] = []
    for page, offset in enumerate(range(0, len(frames), FRAMES_PER_COLLAGE), start=1):
        collage = np.full(
            (collage_height, collage_width, 3), (10, 12, 15), dtype=np.uint8
        )
        for index, selected in enumerate(frames[offset : offset + FRAMES_PER_COLLAGE]):
            tile = add_timestamp_footer(selected.image, selected.timestamp_seconds)
            row, column = divmod(index, COLLAGE_COLUMNS)
            x = COLLAGE_PADDING + column * (FRAME_WIDTH + COLLAGE_PADDING)
            y = COLLAGE_PADDING + row * (tile_height + COLLAGE_PADDING)
            collage[y : y + tile_height, x : x + FRAME_WIDTH] = tile

        output_path = output_stem.with_name(f"{output_stem.name}_page{page}.jpg")
        Image.fromarray(collage).save(
            output_path,
            "JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=True,
            progressive=True,
        )
        output_paths.append(output_path)
    return output_paths


def publish_collages(
    staged_paths: list[Path],
    destination: Path,
    video_id: str,
) -> list[Path]:
    for stale_path in destination.glob(f"{video_id}_vision_page*.jpg"):
        stale_path.unlink()
    published: list[Path] = []
    for page, staged_path in enumerate(staged_paths, start=1):
        output_path = destination / f"{video_id}_vision_page{page}.jpg"
        staged_path.replace(output_path)
        published.append(output_path.resolve())
    return published


def add_timestamp_footer(
    frame: np.ndarray,
    timestamp_seconds: float,
) -> np.ndarray:
    footer = np.full(
        (TIMESTAMP_FOOTER_HEIGHT, frame.shape[1], 3), (15, 17, 21), dtype=np.uint8
    )
    footer[0] = (47, 52, 61)
    text = format_timestamp(timestamp_seconds)
    image = Image.fromarray(footer)
    draw = ImageDraw.Draw(image)
    draw.text(
        (10, TIMESTAMP_FOOTER_HEIGHT // 2),
        text,
        fill=(211, 216, 224),
        font=TIMESTAMP_FONT,
        anchor="lm",
    )
    return np.concatenate((frame, np.asarray(image)), axis=0)


def format_timestamp(timestamp_seconds: float) -> str:
    total_seconds = round(timestamp_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"
