"""Own video preflight, timed-caption policy, temporary media lifecycle, and vision result publication."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..artifacts import create_artifact_directory
from ..config import DEFAULT_VISION_MAX_FRAMES, MAX_VISION_FRAMES
from ..identity import VideoRef, require_video_id
from ..transcript import fetch_transcript
from ..youtube.caption import TranscriptOptions
from ..youtube.client import YtDlpClient
from .frames import FRAME_HEIGHT, FRAME_WIDTH, SelectedFrame, select_frames
from .transcript import select_transcript_windows

MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
TIMESTAMP_FOOTER_HEIGHT = 24
COLLAGE_PADDING = 6
COLLAGE_COLUMNS = 2
COLLAGE_ROWS = 3
FRAMES_PER_COLLAGE = COLLAGE_COLUMNS * COLLAGE_ROWS
JPEG_QUALITY = 80
JPEG_SUBSAMPLING = "4:2:2"
TIMESTAMP_FONT = ImageFont.load_default(size=16)


def sample_vision(
    video_ref: VideoRef,
    *,
    output_dir: Path | None = None,
    max_frames: int = DEFAULT_VISION_MAX_FRAMES,
) -> dict[str, Any]:
    """Use timed captions to shortlist windows, then select images without model APIs or STT charges.

    Missing, untimed, or unavailable captions fall back to evenly distributed visual windows.
    Only the requested windows are decoded; downloaded media is removed after collages publish.
    """
    if not 1 <= max_frames <= MAX_VISION_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_VISION_FRAMES}")
    video_id = require_video_id(video_ref)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for video vision extraction")
    client = YtDlpClient()
    metadata = client.metadata(video_ref)
    fetched_video_id = require_video_id(str(metadata.get("id") or ""))
    if fetched_video_id != video_id:
        raise ValueError(
            f"yt-dlp returned video {fetched_video_id} for requested video {video_id}"
        )
    duration = metadata.get("duration")
    if (
        not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"Missing valid duration for {video_id}")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError(
            f"Video is longer than {MAX_VIDEO_DURATION_SECONDS // 3600} hours"
        )
    transcript_warning = None
    try:
        document = fetch_transcript(
            video_ref,
            options=TranscriptOptions(allow_stt=False),
            client=client,
            metadata=metadata,
        )
        windows = select_transcript_windows(document, max_windows=max_frames)
        if not windows:
            transcript_warning = (
                "No usable timed transcript passages; using visual sampling."
            )
    except RuntimeError as error:
        windows = []
        transcript_warning = (
            f"Timed captions unavailable; using visual sampling: {error}"
        )
    destination = output_dir
    try:
        destination = (
            (output_dir or create_artifact_directory("vision", video_id))
            .expanduser()
            .resolve()
        )
        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="work-", dir=destination) as work_dir_name:
            work_dir = Path(work_dir_name)
            video_path = work_dir / f"{video_id}.mp4"
            client.download_video_for_vision(
                video_ref, video_path, max_bytes=MAX_VIDEO_BYTES
            )
            selected = select_frames(
                video_path,
                windows,
                ffmpeg_path,
                duration=float(duration),
                max_frames=max_frames,
            )
            staged_collages = render_timestamped_collages(selected, work_dir / "vision")
            collage_paths = publish_collages(staged_collages, destination, video_id)
    except OSError as error:
        raise RuntimeError(
            f"Vision artifact filesystem error under {destination}: {error}"
        ) from error
    return {
        "video_id": video_id,
        "duration_seconds": float(duration),
        "heatmap_available": bool(metadata.get("heatmap")),
        "selection_mode": "transcript" if windows else "visual",
        "transcript_warning": transcript_warning,
        "frame_count": len(selected),
        "timestamps_seconds": [frame.timestamp_seconds for frame in selected],
        "collage_paths": [str(path) for path in collage_paths],
        "frames": [frame.to_dict() for frame in selected],
    }


def format_timestamp(timestamp_seconds: float) -> str:
    """Format a frame timestamp compactly for its visible footer."""
    total_seconds = round(timestamp_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def add_timestamp_footer(
    frame: np.ndarray,
    timestamp_seconds: float,
) -> np.ndarray:
    """Append a Pillow-rendered timestamp without covering frame content."""
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


def render_timestamped_collages(
    frames: list[SelectedFrame], output_stem: Path
) -> list[Path]:
    """Assemble six chronological 360p screenshots per page without decoding the video again."""
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
    """Replace the previous collage set only after every new page renders."""
    for stale_path in destination.glob(f"{video_id}_vision_page*.jpg"):
        stale_path.unlink()
    published: list[Path] = []
    for page, staged_path in enumerate(staged_paths, start=1):
        output_path = destination / f"{video_id}_vision_page{page}.jpg"
        staged_path.replace(output_path)
        published.append(output_path.resolve())
    return published
