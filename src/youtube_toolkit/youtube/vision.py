"""Select bounded timestamped visual context from one YouTube video."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..cache import cache_root
from ..identity import VideoRef, require_video_id
from .client import YtDlpClient

MAX_OUTPUT_FRAMES = 96
MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 10 * 60
# Budget one quarter of the host for each job while capping low-resolution codec parallelism.
FFMPEG_THREADS = max(1, min(4, (os.process_cpu_count() or 1) // 4))
SCENE_CHANGE_FRACTION = 0.05
REPLAY_PEAK_PERCENTILE = 75
REPLAY_CENTER_RADIUS_SECONDS = 2
MIN_SCENE_CHANGE_SPACING_SECONDS = 8
THUMBNAIL_WIDTH = 160
THUMBNAIL_HEIGHT = 90
BASE_DEDUP_THRESHOLD = 0.35
SUPPLEMENT_DEDUP_THRESHOLD = 0.60
STATIONARY_DURATION_SECONDS = 3
BLUR_NEIGHBOR_RADIUS = 2
BLUR_RATIO_THRESHOLD = 1.5
BLUR_DELTA_THRESHOLD = 2.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
TIMESTAMP_FOOTER_HEIGHT = 24
COLLAGE_PADDING = 6
COLLAGE_COLUMNS = 2
COLLAGE_ROWS = 3
FRAMES_PER_COLLAGE = COLLAGE_COLUMNS * COLLAGE_ROWS
JPEG_QUALITY = 80
JPEG_SUBSAMPLING = "4:2:2"
TIMESTAMP_FONT = ImageFont.load_default(size=16)


@dataclass(frozen=True)
class ScoredFrame:
    """Associate a sampled frame with its timestamp and incoming scene change."""

    index: int
    timestamp_seconds: float
    score: float


@dataclass(frozen=True)
class HeatmapBin:
    """Represent one normalized viewer-replay interval from yt-dlp."""

    start_seconds: float
    end_seconds: float
    value: float


def sample_vision(
    video_ref: VideoRef,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Sample and render bounded visual context from one video."""
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
    if not isinstance(duration, int | float) or duration <= 0:
        raise ValueError(f"Missing valid duration for {video_id}")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError(
            f"Video is longer than {MAX_VIDEO_DURATION_SECONDS // 3600} hours"
        )
    heatmap = heatmap_bins(metadata.get("heatmap"))

    destination = (
        (output_dir or cache_root() / "vision" / video_id).expanduser().resolve()
    )
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="work-", dir=destination) as work_dir_name:
            work_dir = Path(work_dir_name)
            video_path = work_dir / f"{video_id}.mp4"
            client.download_video_for_vision(
                video_ref, video_path, max_bytes=MAX_VIDEO_BYTES
            )
            selected = select_frames(video_path, heatmap, ffmpeg_path)
            staged_collages = render_timestamped_collages(
                video_path,
                selected,
                work_dir / "vision",
                ffmpeg_path,
            )
            collage_paths = publish_collages(staged_collages, destination, video_id)
    except OSError as error:
        raise RuntimeError(
            f"Vision artifact filesystem error under {destination}: {error}"
        ) from error

    return {
        "video_id": video_id,
        "duration_seconds": float(duration),
        "heatmap_available": bool(heatmap),
        "frame_count": len(selected),
        "timestamps_seconds": [frame.timestamp_seconds for frame in selected],
        "collage_paths": [str(path) for path in collage_paths],
    }


def heatmap_bins(raw_heatmap: object) -> list[HeatmapBin]:
    """Normalize optional yt-dlp replay data into validated bins."""
    if raw_heatmap is None:
        return []
    if not isinstance(raw_heatmap, list):
        raise ValueError("yt-dlp returned an invalid replay heatmap")
    try:
        bins = [
            HeatmapBin(
                start_seconds=float(item["start_time"]),
                end_seconds=float(item["end_time"]),
                value=float(item["value"]),
            )
            for item in raw_heatmap
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("yt-dlp returned an invalid replay heatmap") from error
    if any(
        not math.isfinite(item.start_seconds)
        or not math.isfinite(item.end_seconds)
        or not math.isfinite(item.value)
        or item.start_seconds < 0
        or item.end_seconds <= item.start_seconds
        or item.value < 0
        for item in bins
    ):
        raise ValueError("yt-dlp returned an invalid replay heatmap")
    return bins


def select_frames(
    video_path: Path,
    heatmap: list[HeatmapBin],
    ffmpeg_path: str,
) -> list[ScoredFrame]:
    """Apply scene, replay, stationary, blur, deduplication, and output-cap policy."""
    frames, frame_blur_scores = frame_scores(video_path, ffmpeg_path)
    if not frames:
        raise ValueError("FFmpeg produced no one-FPS scene frames")
    thumbnails = extract_thumbnails(video_path, len(frames), ffmpeg_path)

    scene_changes = select_strong_scene_changes(frames)
    sharp_valleys = discard_motion_blur(
        relocate_to_valleys(frames, scene_changes), frame_blur_scores
    )
    selected = add_distinct_candidates(
        [],
        sharp_valleys,
        thumbnails,
        BASE_DEDUP_THRESHOLD,
    )

    replay_regions = select_replay_regions(heatmap)
    replay_frames = stable_replay_frames(frames, replay_regions)
    selected = add_distinct_candidates(
        selected,
        discard_motion_blur(replay_frames, frame_blur_scores),
        thumbnails,
        SUPPLEMENT_DEDUP_THRESHOLD,
    )
    stationary_frames = stationary_midpoints(video_path, frames, ffmpeg_path)
    selected = add_distinct_candidates(
        selected,
        discard_motion_blur(stationary_frames, frame_blur_scores),
        thumbnails,
        SUPPLEMENT_DEDUP_THRESHOLD,
    )
    if not selected:
        finite_frames = [
            frame for frame in frames if math.isfinite(frame_blur_scores[frame.index])
        ]
        selected = (
            [min(finite_frames, key=lambda frame: frame_blur_scores[frame.index])]
            if finite_frames
            else [frames[len(frames) // 2]]
        )
    return limit_frames(selected)


def frame_scores(
    video_path: Path,
    ffmpeg_path: str,
) -> tuple[list[ScoredFrame], list[float]]:
    """Score scene change and blur for every second of video in one decode pass."""
    metadata = parse_frame_metadata(
        run_ffmpeg(
            ffmpeg_path,
            [
                "-i",
                str(video_path),
                "-vf",
                "fps=1,select='gte(scene,0)',blurdetect,metadata=print:file=-",
                "-fps_mode",
                "vfr",
                "-f",
                "null",
                "-",
            ],
        ),
        ("lavfi.scene_score", "lavfi.blur"),
    )
    frames = [
        ScoredFrame(index=index, timestamp_seconds=timestamp_seconds, score=scene)
        for index, (timestamp_seconds, (scene, _blur)) in enumerate(metadata)
    ]
    return frames, [blur for _timestamp_seconds, (_scene, blur) in metadata]


def parse_frame_metadata(
    output: bytes,
    value_keys: tuple[str, ...],
) -> list[tuple[float, list[float]]]:
    """Parse each FFmpeg frame header followed by one numeric line per value key, in key order."""
    lines = output.decode().splitlines()
    label = ", ".join(value_keys)
    stride = len(value_keys) + 1
    if len(lines) % stride:
        raise ValueError(f"FFmpeg returned incomplete {label} metadata")
    frames: list[tuple[float, list[float]]] = []
    for offset in range(0, len(lines), stride):
        frame_line, *value_lines = lines[offset : offset + stride]
        timestamp_match = re.search(r"pts_time:([0-9.]+)", frame_line)
        if timestamp_match is None or not all(
            line.startswith(f"{key}=")
            for key, line in zip(value_keys, value_lines, strict=True)
        ):
            raise ValueError(f"FFmpeg returned invalid {label} metadata")
        try:
            frames.append(
                (
                    float(timestamp_match.group(1)),
                    [float(line.split("=", 1)[1]) for line in value_lines],
                )
            )
        except ValueError as error:
            raise ValueError(f"FFmpeg returned invalid {label} metadata") from error
    return frames


def stationary_midpoints(
    video_path: Path,
    frames: list[ScoredFrame],
    ffmpeg_path: str,
) -> list[ScoredFrame]:
    """Return midpoint frames from runs stationary for at least three seconds."""
    output = run_ffmpeg(
        ffmpeg_path,
        [
            "-i",
            str(video_path),
            "-vf",
            (
                f"fps=1,scale={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT}:flags=area,format=gray,gblur=sigma=1,freezedetect=n=0.001:d={STATIONARY_DURATION_SECONDS},metadata=print:file=-"
            ),
            "-fps_mode",
            "vfr",
            "-f",
            "null",
            "-",
        ],
    ).decode()
    midpoints: list[ScoredFrame] = []
    start_seconds: float | None = None
    for line in output.splitlines():
        if line.startswith("lavfi.freezedetect.freeze_start="):
            start_seconds = float(line.split("=", 1)[1])
        elif (
            line.startswith("lavfi.freezedetect.freeze_end=")
            and start_seconds is not None
        ):
            end_seconds = float(line.split("=", 1)[1])
            midpoint = (start_seconds + end_seconds) / 2
            midpoints.append(
                min(frames, key=lambda frame: abs(frame.timestamp_seconds - midpoint))
            )
            start_seconds = None
    if start_seconds is not None:
        midpoint = (start_seconds + frames[-1].timestamp_seconds) / 2
        midpoints.append(
            min(frames, key=lambda frame: abs(frame.timestamp_seconds - midpoint))
        )
    return midpoints


def extract_thumbnails(
    video_path: Path,
    frame_count: int,
    ffmpeg_path: str,
) -> np.ndarray:
    """Decode normalized grayscale thumbnails for perceptual comparison."""
    output = run_ffmpeg(
        ffmpeg_path,
        [
            "-i",
            str(video_path),
            "-vf",
            f"fps=1,scale={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT}:flags=area,format=gray,gblur=sigma=1",
            "-fps_mode",
            "vfr",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
    )
    thumbnails = np.frombuffer(output, dtype=np.uint8)
    expected_size = frame_count * THUMBNAIL_HEIGHT * THUMBNAIL_WIDTH
    if thumbnails.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} thumbnail bytes, received {thumbnails.size}"
        )
    return thumbnails.reshape(-1, THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH)


def select_strong_scene_changes(frames: list[ScoredFrame]) -> list[ScoredFrame]:
    """Keep the strongest five-percent scene changes with temporal suppression."""
    candidate_count = math.ceil((len(frames) - 1) * SCENE_CHANGE_FRACTION)
    candidates = sorted(
        frames[1:], key=lambda frame: (-frame.score, frame.timestamp_seconds)
    )[:candidate_count]
    selected: list[ScoredFrame] = []
    for candidate in candidates:
        if all(
            abs(candidate.timestamp_seconds - retained.timestamp_seconds)
            >= MIN_SCENE_CHANGE_SPACING_SECONDS
            for retained in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda frame: frame.timestamp_seconds)


def relocate_to_valleys(
    frames: list[ScoredFrame],
    scene_changes: list[ScoredFrame],
) -> list[ScoredFrame]:
    """Move strong scene changes to the first stable local minimum after each transition."""
    valleys: list[ScoredFrame] = []
    for change_number, scene_change in enumerate(scene_changes):
        stop = (
            scene_changes[change_number + 1].index
            if change_number + 1 < len(scene_changes)
            else len(frames)
        )
        centers = list(range(scene_change.index + 2, stop - 1))
        if not centers:
            valleys.append(scene_change)
            continue
        rolling_scores = {
            center: sum(frames[index].score for index in range(center - 1, center + 2))
            / 3
            for center in centers
        }
        local_minimum = next(
            (
                center
                for center in centers[1:-1]
                if rolling_scores[center] <= rolling_scores[center - 1]
                and rolling_scores[center] <= rolling_scores[center + 1]
            ),
            None,
        )
        valley_index = (
            local_minimum
            if local_minimum is not None
            else min(centers, key=rolling_scores.__getitem__)
        )
        valleys.append(frames[valley_index])
    return valleys


def select_replay_regions(bins: list[HeatmapBin]) -> list[tuple[float, float]]:
    """Keep local replay maxima above the per-video 75th percentile."""
    if not bins:
        return []
    threshold = float(
        np.percentile([item.value for item in bins], REPLAY_PEAK_PERCENTILE)
    )
    peak_indexes = [
        index
        for index in range(1, len(bins) - 1)
        if bins[index].value > bins[index - 1].value
        and bins[index].value >= bins[index + 1].value
        and bins[index].value >= threshold
    ]
    return [
        (bins[index].start_seconds, bins[index].end_seconds) for index in peak_indexes
    ]


def stable_replay_frames(
    frames: list[ScoredFrame],
    regions: list[tuple[float, float]],
) -> list[ScoredFrame]:
    """Choose the lowest rolling scene-change frame within two seconds of each replay midpoint."""
    selected: list[ScoredFrame] = []
    for start_seconds, end_seconds in regions:
        midpoint = (start_seconds + end_seconds) / 2
        candidates = [
            frame
            for frame in frames
            if midpoint - REPLAY_CENTER_RADIUS_SECONDS
            <= frame.timestamp_seconds
            <= midpoint + REPLAY_CENTER_RADIUS_SECONDS
            and 0 < frame.index < len(frames) - 1
        ]
        if not candidates:
            continue
        selected.append(
            min(
                candidates,
                key=lambda frame: (
                    sum(
                        frames[index].score
                        for index in range(frame.index - 1, frame.index + 2)
                    )
                    / 3
                ),
            )
        )
    return selected


def discard_motion_blur(
    frames: list[ScoredFrame],
    frame_blur_scores: list[float],
) -> list[ScoredFrame]:
    """Discard transient blur outliers while retaining intentionally soft scenes."""
    selected: list[ScoredFrame] = []
    for frame in frames:
        start = max(0, frame.index - BLUR_NEIGHBOR_RADIUS)
        stop = min(len(frame_blur_scores), frame.index + BLUR_NEIGHBOR_RADIUS + 1)
        neighbors = (
            frame_blur_scores[start : frame.index]
            + frame_blur_scores[frame.index + 1 : stop]
        )
        if not neighbors:
            selected.append(frame)
            continue
        finite_neighbors = [score for score in neighbors if math.isfinite(score)]
        if not finite_neighbors:
            selected.append(frame)
            continue
        local_baseline = float(np.median(finite_neighbors))
        blur_score = frame_blur_scores[frame.index]
        if (
            math.isfinite(local_baseline)
            and math.isfinite(blur_score)
            and blur_score >= local_baseline * BLUR_RATIO_THRESHOLD
            and blur_score - local_baseline >= BLUR_DELTA_THRESHOLD
        ):
            continue
        selected.append(frame)
    return selected


@dataclass(frozen=True, slots=True)
class _ThumbnailStats:
    """Retain per-block moments once for each thumbnail reused during duplicate rejection."""

    mean: np.ndarray
    variance: np.ndarray
    blocks: np.ndarray

    @classmethod
    def from_image(cls, image: np.ndarray) -> _ThumbnailStats:
        usable_height = THUMBNAIL_HEIGHT - THUMBNAIL_HEIGHT % 8
        usable_width = THUMBNAIL_WIDTH - THUMBNAIL_WIDTH % 8
        blocks = (
            image[:usable_height, :usable_width]
            .reshape(usable_height // 8, 8, usable_width // 8, 8)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 64)
        )
        mean = blocks.mean(axis=1)
        return cls(mean=mean, variance=blocks.var(axis=1), blocks=blocks)

    def similarity(self, other: _ThumbnailStats) -> float:
        """Approximate FFmpeg SSIM over non-overlapping eight-pixel blocks from prepared moments."""
        covariance = (
            (self.blocks - self.mean[:, None]) * (other.blocks - other.mean[:, None])
        ).mean(axis=1)
        luminance_constant = (0.01 * 255) ** 2
        structure_constant = (0.03 * 255) ** 2
        similarity = (
            (2 * self.mean * other.mean + luminance_constant)
            * (2 * covariance + structure_constant)
            / (
                (self.mean**2 + other.mean**2 + luminance_constant)
                * (self.variance + other.variance + structure_constant)
            )
        )
        return float(similarity.mean())


def add_distinct_candidates(
    selected_frames: list[ScoredFrame],
    candidate_frames: list[ScoredFrame],
    thumbnails: np.ndarray,
    similarity_threshold: float,
) -> list[ScoredFrame]:
    """Return retained frames plus chronologically ordered distinct candidates."""
    combined = list(selected_frames)
    retained_thumbnails = [
        _ThumbnailStats.from_image(thumbnails[frame.index]) for frame in combined
    ]
    selected_indexes = {frame.index for frame in combined}
    for frame in candidate_frames:
        if frame.index in selected_indexes:
            continue
        thumbnail = _ThumbnailStats.from_image(thumbnails[frame.index])
        if any(
            thumbnail.similarity(kept) >= similarity_threshold
            for kept in retained_thumbnails
        ):
            continue
        combined.append(frame)
        retained_thumbnails.append(thumbnail)
        selected_indexes.add(frame.index)
    return sorted(combined, key=lambda frame: frame.timestamp_seconds)


def limit_frames(frames: list[ScoredFrame]) -> list[ScoredFrame]:
    """Thin oversized chronological output evenly without importance ranking."""
    if len(frames) <= MAX_OUTPUT_FRAMES:
        return frames
    step = (len(frames) - 1) / (MAX_OUTPUT_FRAMES - 1)
    return [frames[round(index * step)] for index in range(MAX_OUTPUT_FRAMES)]


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
    video_path: Path,
    frames: list[ScoredFrame],
    output_stem: Path,
    ffmpeg_path: str,
) -> list[Path]:
    """Decode selected frames once and assemble timestamped tiles one page at a time."""
    if not frames:
        return []
    decode_frames = sorted(
        {frame.index: frame for frame in frames}.values(), key=lambda frame: frame.index
    )
    selection = "+".join(f"eq(n,{frame.index})" for frame in decode_frames)
    output = run_ffmpeg(
        ffmpeg_path,
        [
            "-i",
            str(video_path),
            "-vf",
            (
                f"fps=1,select='{selection}',scale={FRAME_WIDTH}:{FRAME_HEIGHT}:"
                "force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
            ),
            "-fps_mode",
            "vfr",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
    )
    pixels = np.frombuffer(output, dtype=np.uint8)
    expected_size = len(decode_frames) * FRAME_HEIGHT * FRAME_WIDTH * 3
    if pixels.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} frame bytes, received {pixels.size}"
        )
    decoded = pixels.reshape(-1, FRAME_HEIGHT, FRAME_WIDTH, 3)
    decoded_by_index = {
        selected.index: frame
        for selected, frame in zip(decode_frames, decoded, strict=True)
    }
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
            tile = add_timestamp_footer(
                decoded_by_index[selected.index], selected.timestamp_seconds
            )
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


def run_ffmpeg(
    ffmpeg_path: str,
    arguments: list[str],
) -> bytes:
    """Run one bounded FFmpeg operation and preserve its diagnostic failure."""
    try:
        result = subprocess.run(  # noqa: S603 - executable is resolved from the local PATH and arguments are internally assembled.
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                str(FFMPEG_THREADS),
                "-y",
                *arguments,
            ],
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"FFmpeg vision processing timed out after {FFMPEG_TIMEOUT_SECONDS} seconds"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Could not start FFmpeg vision processing: {error}"
        ) from error
    if result.returncode:
        raise RuntimeError(
            result.stderr.decode().strip() or "FFmpeg vision processing failed"
        )
    return result.stdout
