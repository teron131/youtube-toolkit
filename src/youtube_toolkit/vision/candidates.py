"""Collect a video's keyframes as disk-backed candidates and, without Gemini ranking, spread the selection across them in equal time slots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..ffmpeg_utils import decode_keyframes

# On four graded videos, 10-15 s slots caught about 95% of key moments while 20 s slots caught 77%; 12 s keeps a margin before that drop.
SLOT_SECONDS = 12
REPEAT_SIMILARITY = 0.95
MIN_THUMBNAIL_CONTRAST = 3.0
THUMBNAIL_WIDTH = 160
THUMBNAIL_HEIGHT = 90
DETECTOR_WIDTH = 320
DETECTOR_HEIGHT = 180
DECODE_CHUNK_SECONDS = 60
DUPLICATE_LOOKBACK_SECONDS = 10
# Stricter than REPEAT_SIMILARITY because a candidate dropped here reaches neither selection; both selections drop the remaining duplicates later.
CANDIDATE_DUPLICATE_SIMILARITY = 0.98


@dataclass(frozen=True, slots=True)
class VisualScores:
    """Raw image measurements, not confidence or semantic importance probabilities."""

    clarity: float
    neighbor_change: float
    similarity_to_previous: float


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    timestamp_seconds: float
    image: np.ndarray = field(repr=False, compare=False)
    visual_scores: VisualScores

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "visual_scores": asdict(self.visual_scores),
        }


@dataclass(frozen=True, slots=True)
class StoredFrame:
    timestamp_seconds: float
    image_path: Path
    visual_scores: VisualScores

    def load(self) -> DecodedFrame:
        with Image.open(self.image_path) as image:
            pixels = np.array(image.convert("RGB"))
        return DecodedFrame(self.timestamp_seconds, pixels, self.visual_scores)


def collect_candidates(
    video_path: Path,
    ffmpeg_path: str,
    *,
    duration: float,
    directory: Path,
    excluded_intervals: list[tuple[float, float]],
) -> list[StoredFrame]:
    """Collect the video's keyframes as candidates independently of the final frame budget.

    YouTube encodes place a keyframe every three to seven seconds, so keyframes give an even grid without decoding the frames between them.
    Only nearby duplicates by pixel similarity are dropped; this appearance check does not guarantee semantic equivalence.
    """
    directory.mkdir(parents=True, exist_ok=True)
    candidates = []
    recent: list[tuple[float, _ThumbnailStats]] = []
    for start in range(0, math.ceil(duration), DECODE_CHUNK_SECONDS):
        times, images = decode_keyframes(
            video_path, start, min(duration, start + DECODE_CHUNK_SECONDS), ffmpeg_path
        )
        previous = None
        for index, timestamp in enumerate(times):
            gray = _gray(images[index])
            change = (
                0.0
                if previous is None
                else float(cv2.absdiff(previous, gray).mean() / 255)
            )
            previous = gray
            if gray.std() < MIN_THUMBNAIL_CONTRAST or any(
                left <= timestamp < right for left, right in excluded_intervals
            ):
                continue
            recent = [
                (time, thumbnail)
                for time, thumbnail in recent
                if timestamp - time <= DUPLICATE_LOOKBACK_SECONDS
            ]
            thumbnail = _ThumbnailStats.from_gray(gray)
            similarity = max(
                (thumbnail.similarity(other) for _, other in recent), default=0.0
            )
            if similarity >= CANDIDATE_DUPLICATE_SIMILARITY:
                continue
            scores = VisualScores(
                float(cv2.Laplacian(gray, cv2.CV_64F).var()), change, similarity
            )
            path = directory / f"frame-{round(timestamp * 1000):09d}.jpg"
            Image.fromarray(images[index]).save(path, "JPEG", quality=95)
            candidates.append(StoredFrame(timestamp, path, scores))
            recent.append((timestamp, thumbnail))
    return sorted(candidates, key=lambda frame: frame.timestamp_seconds)


def select_slot_frames(
    candidates: list[StoredFrame],
    *,
    duration: float,
    max_frames: int,
    min_gap_seconds: float,
) -> list[DecodedFrame]:
    """Split the video into equal slots of at least SLOT_SECONDS, never more than max_frames, and take the candidate nearest each slot's middle.

    A slot yields nothing when it has no candidates, when its pick falls within min_gap_seconds of the previous pick, or when its pick looks like the previous pick, so static stretches give fewer frames.
    """
    count = min(max_frames, math.ceil(duration / SLOT_SECONDS))
    width = duration / count
    picks: dict[int, StoredFrame] = {}
    for candidate in candidates:
        slot = min(count - 1, int(candidate.timestamp_seconds / width))
        middle = (slot + 0.5) * width
        best = picks.get(slot)
        if best is None or abs(candidate.timestamp_seconds - middle) < abs(
            best.timestamp_seconds - middle
        ):
            picks[slot] = candidate
    selected: list[DecodedFrame] = []
    previous: _ThumbnailStats | None = None
    for slot in sorted(picks):
        pick = picks[slot]
        if (
            selected
            and pick.timestamp_seconds - selected[-1].timestamp_seconds
            < min_gap_seconds
        ):
            continue
        frame = pick.load()
        thumbnail = _ThumbnailStats.from_gray(_gray(frame.image))
        if previous is not None and thumbnail.similarity(previous) >= REPEAT_SIMILARITY:
            continue
        selected.append(frame)
        previous = thumbnail
    return selected


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        cv2.resize(image, (DETECTOR_WIDTH, DETECTOR_HEIGHT)), cv2.COLOR_RGB2GRAY
    )


@dataclass(frozen=True, slots=True)
class _ThumbnailStats:
    """Prepare block moments and centered pixels once for repeated duplicate comparisons."""

    mean: np.ndarray
    variance: np.ndarray
    centered_blocks: np.ndarray

    @classmethod
    def from_gray(cls, gray: np.ndarray) -> _ThumbnailStats:
        image = cv2.resize(
            gray, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), interpolation=cv2.INTER_AREA
        )
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
