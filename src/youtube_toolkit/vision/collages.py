"""Lay selected frames out as JPEG pages in a grid of the requested frame size, each frame with its time printed beneath unless annotation is off.

A page is deliberately not fitted to vision models' image limits: the frame resolution is the caller's choice, and a model downscales an oversized page itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .candidates import DecodedFrame

TIMESTAMP_FOOTER_HEIGHT = 24
COLLAGE_PADDING = 6
JPEG_QUALITY = 80
JPEG_SUBSAMPLING = "4:2:2"
TIMESTAMP_FONT = ImageFont.load_default(size=16)


def render_collages(
    frames: list[DecodedFrame],
    output_stem: Path,
    *,
    rows: int,
    columns: int,
    frame_size: tuple[int, int],
    timestamp_annotation: bool,
) -> list[Path]:
    width, height = frame_size
    tile_height = height + (TIMESTAMP_FOOTER_HEIGHT if timestamp_annotation else 0)
    collage_width = columns * width + COLLAGE_PADDING * (columns + 1)
    collage_height = rows * tile_height + COLLAGE_PADDING * (rows + 1)
    frames_per_page = rows * columns
    output_paths: list[Path] = []
    for page, offset in enumerate(range(0, len(frames), frames_per_page), start=1):
        collage = np.full(
            (collage_height, collage_width, 3), (10, 12, 15), dtype=np.uint8
        )
        for index, selected in enumerate(frames[offset : offset + frames_per_page]):
            tile = np.asarray(
                Image.fromarray(selected.image).resize(
                    frame_size, Image.Resampling.LANCZOS
                )
            )
            if timestamp_annotation:
                tile = add_timestamp_footer(tile, selected.timestamp_seconds)
            row, column = divmod(index, columns)
            x = COLLAGE_PADDING + column * (width + COLLAGE_PADDING)
            y = COLLAGE_PADDING + row * (tile_height + COLLAGE_PADDING)
            collage[y : y + tile_height, x : x + width] = tile

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
    """Hours show only from one hour on, and seconds round to the nearest: 65.4 → `01:05`, 3725.9 → `1:02:06`."""
    total_seconds = round(timestamp_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"
