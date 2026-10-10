"""Decode shortlisted intervals and choose scene representatives from image quality, stability, and diversity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
from cv2 import data as cv_data
import numpy as np
from scenedetect import FrameTimecode
from scenedetect.detectors import AdaptiveDetector

from ..config import DEFAULT_VISION_MAX_FRAMES
from .windows import TranscriptWindow

FFMPEG_TIMEOUT_SECONDS = 10 * 60
FFMPEG_THREADS = max(1, min(2, (os.process_cpu_count() or 1) // 4))
WINDOW_PADDING_BEFORE = 2
WINDOW_PADDING_AFTER = 5
FALLBACK_WINDOW_SECONDS = 30
DUPLICATE_SIMILARITY = 0.95
MIN_THUMBNAIL_CONTRAST = 3.0
LARGE_FACE_FRACTION = 0.04
MIN_ALTERNATIVE_SHOT_SECONDS = 3
THUMBNAIL_WIDTH = 160
THUMBNAIL_HEIGHT = 90
DETECTOR_WIDTH = 320
DETECTOR_HEIGHT = 180
FRAME_WIDTH = 640
FRAME_HEIGHT = 360


@dataclass(frozen=True, slots=True)
class VisualScores:
    """Raw image measurements, not confidence or semantic importance probabilities."""

    clarity: float
    neighbor_change: float
    shot_overlap_seconds: float
    face_area: float
    similarity_to_previous: float


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    """Keep one owned RGB image and the evidence explaining its selection."""

    timestamp_seconds: float
    image: np.ndarray = field(repr=False, compare=False)
    transcript: TranscriptWindow | None
    visual_scores: VisualScores

    def to_dict(self) -> dict[str, Any]:
        """Project selection evidence without exposing the in-memory image buffer."""
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "transcript": asdict(self.transcript) if self.transcript else None,
            "visual_scores": asdict(self.visual_scores),
        }


def select_frames(
    video_path: Path,
    windows: list[TranscriptWindow],
    ffmpeg_path: str,
    *,
    duration: float,
    max_frames: int = DEFAULT_VISION_MAX_FRAMES,
) -> list[SelectedFrame]:
    """Decode bounded windows and retain distinct representatives using image measurements alone."""
    detector = cv2.CascadeClassifier(
        str(Path(cv_data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if detector.empty():
        raise RuntimeError(
            "OpenCV's bundled face detector is missing; reinstall youtube-toolkit[vision]."
        )
    bounds: list[tuple[float, float, TranscriptWindow | None]] = [
        (window.start_seconds, window.end_seconds, window)
        for window in windows[:max_frames]
    ]
    if not bounds:
        count = min(max_frames, max(1, math.ceil(duration / FALLBACK_WINDOW_SECONDS)))
        for center in (np.arange(count) + 0.5) * duration / count:
            bounds.append(
                (
                    max(0.0, float(center) - FALLBACK_WINDOW_SECONDS / 2),
                    min(duration, float(center) + FALLBACK_WINDOW_SECONDS / 2),
                    None,
                )
            )
    selected: list[SelectedFrame] = []
    retained: list[_ThumbnailStats] = []
    for start, stop, passage in bounds:
        clip_start = max(0, math.floor(start - WINDOW_PADDING_BEFORE))
        clip_stop = min(duration, math.ceil(stop + WINDOW_PADDING_AFTER))
        images = decode_window(video_path, clip_start, clip_stop, ffmpeg_path)
        if not len(images):
            continue
        candidate = choose_frame(images, clip_start, start, stop, retained, detector)
        if candidate is None:
            continue
        timestamp = float(clip_start + candidate.index)
        if any(frame.timestamp_seconds == timestamp for frame in selected):
            continue
        retained.append(candidate.thumbnail)
        # A view would keep the entire decoded window alive for every selected screenshot.
        selected.append(
            SelectedFrame(
                timestamp, images[candidate.index].copy(), passage, candidate.scores
            )
        )
    if not selected:
        raise ValueError("FFmpeg produced no usable visual frames")
    return sorted(selected, key=lambda frame: frame.timestamp_seconds)


def decode_window(
    video_path: Path, start: float, stop: float, ffmpeg_path: str
) -> np.ndarray:
    """Seek before decoding so analysis cost depends on shortlisted intervals, not full video length."""
    output = run_ffmpeg(
        ffmpeg_path,
        [
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(stop - start),
            "-vf",
            f"fps=1,scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
    )
    pixels = np.frombuffer(output, dtype=np.uint8)
    frame_bytes = FRAME_HEIGHT * FRAME_WIDTH * 3
    if pixels.size % frame_bytes:
        raise ValueError(f"FFmpeg returned incomplete RGB frames ({pixels.size} bytes)")
    return pixels.reshape(-1, FRAME_HEIGHT, FRAME_WIDTH, 3)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One clear shot representative and its prepared duplicate-comparison statistics."""

    index: int
    scores: VisualScores
    thumbnail: _ThumbnailStats


def choose_frame(
    images: np.ndarray,
    clip_start: float,
    window_start: float,
    window_end: float,
    retained: list[_ThumbnailStats],
    face_detector: cv2.CascadeClassifier,
) -> _Candidate | None:
    """Prefer sustained, nonduplicate shots and clear stable representatives, with a portrait preference.

    Clarity is compared within each shot so diagrams do not compete with photographic texture.
    Large faces are downweighted only when another sustained shot exists; no text phrases enter this stage.
    """
    small = [
        cv2.resize(
            image, (DETECTOR_WIDTH, DETECTOR_HEIGHT), interpolation=cv2.INTER_AREA
        )
        for image in images
    ]
    gray = np.stack([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in small])
    detector = AdaptiveDetector(
        adaptive_threshold=3.0, min_scene_len=2, window_width=2, min_content_val=15.0
    )
    cuts = {0, len(images)}
    for index, image in enumerate(small):
        cuts.update(
            timecode.frame_num
            for timecode in detector.process_frame(
                FrameTimecode(index, fps=1.0), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
        )
    clarity = np.array([cv2.Laplacian(image, cv2.CV_64F).var() for image in gray])
    contrast = gray.std(axis=(1, 2))
    changes = [0.0]
    changes.extend(
        float(cv2.absdiff(left, right).mean() / 255) for left, right in pairwise(gray)
    )
    candidates = []
    for start, stop in pairwise(sorted(cuts)):
        support = max(
            0.0,
            min(clip_start + stop, window_end) - max(clip_start + start, window_start),
        )
        if stop - start < 2 or support <= 0:
            continue
        inset = int(stop - start > 4)
        options = [
            index
            for index in range(start + inset, stop - inset)
            if contrast[index] > MIN_THUMBNAIL_CONTRAST
        ]
        if not options:
            continue
        sharp_limit = float(np.quantile(clarity[options], 0.25))
        options = [index for index in options if clarity[index] >= sharp_limit]
        instability = {
            index: changes[index] + (changes[index + 1] if index + 1 < stop else 0.0)
            for index in options
        }
        stable_limit = float(np.median(list(instability.values())))
        options = [index for index in options if instability[index] <= stable_limit]
        typical = gray[start:stop].mean(axis=0)
        index = min(
            options,
            key=lambda item: (
                float(np.mean((gray[item] - typical) ** 2)),
                abs(item - (start + stop - 1) / 2),
            ),
        )
        candidates.append(
            _candidate(
                index,
                gray[index],
                float(clarity[index]),
                instability[index],
                support,
                retained,
                face_detector,
            )
        )
    if not candidates:
        index = min(
            len(images) - 1, max(0, round((window_start + window_end) / 2 - clip_start))
        )
        candidates = [
            _candidate(
                index,
                gray[index],
                float(clarity[index]),
                changes[index],
                0.0,
                retained,
                face_detector,
            )
        ]
    distinct = [
        candidate
        for candidate in candidates
        if candidate.scores.similarity_to_previous < DUPLICATE_SIMILARITY
    ]
    if not distinct:
        return None
    alternatives = [
        candidate
        for candidate in distinct
        if candidate.scores.face_area < LARGE_FACE_FRACTION
        and candidate.scores.shot_overlap_seconds >= MIN_ALTERNATIVE_SHOT_SECONDS
    ]
    return max(
        alternatives or distinct,
        key=lambda candidate: (
            candidate.scores.shot_overlap_seconds,
            -abs(clip_start + candidate.index - (window_start + window_end) / 2),
        ),
    )


def _candidate(
    index: int,
    image: np.ndarray,
    clarity: float,
    change: float,
    support: float,
    retained: list[_ThumbnailStats],
    face_detector: cv2.CascadeClassifier,
) -> _Candidate:
    """Attach raw quality and duplicate measurements to one representative."""
    thumbnail = _ThumbnailStats.from_image(
        cv2.resize(
            image,
            (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    )
    similarity = max(
        (thumbnail.similarity(previous) for previous in retained), default=0.0
    )
    faces = face_detector.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    face_area = max(
        (int(width) * int(height) / image.size for _, _, width, height in faces),
        default=0.0,
    )
    scores = VisualScores(
        clarity,
        float(change),
        float(support),
        float(face_area),
        float(similarity),
    )
    return _Candidate(index, scores, thumbnail)


@dataclass(frozen=True, slots=True)
class _ThumbnailStats:
    """Prepare block moments and centered pixels once for repeated duplicate comparisons."""

    mean: np.ndarray
    variance: np.ndarray
    centered_blocks: np.ndarray

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
        return cls(
            mean=mean,
            variance=blocks.var(axis=1),
            centered_blocks=blocks - mean[:, None],
        )

    def similarity(self, other: _ThumbnailStats) -> float:
        """Approximate FFmpeg SSIM over non-overlapping eight-pixel blocks from prepared moments."""
        covariance = (self.centered_blocks * other.centered_blocks).mean(axis=1)
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
                "-filter_threads",
                "1",
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
