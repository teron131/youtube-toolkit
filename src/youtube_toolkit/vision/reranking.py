"""Rerank visual candidates by their similarity to text spoken around them, relative to the whole transcript, fuse scales with RRF, then apply output spacing, duplicate removal, and limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .text import TextTarget

RRF_CONSTANT = 60
# Visuals often appear just before they are discussed and stay up afterward, so a frame belongs to text from 2 s before it starts until 5 s after it ends.
PADDING_BEFORE_SECONDS = 2
PADDING_AFTER_SECONDS = 5
LEVELS = ("keyword", "sentence", "paragraph")


@dataclass(frozen=True, slots=True)
class TextMatch:
    target: TextTarget
    cosine: float
    relative_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class FrameRanking:
    index: int
    score: float
    matches: tuple[TextMatch, ...]


def rerank_frames(
    image_vectors: np.ndarray,
    text_vectors: np.ndarray,
    targets: list[TextTarget],
    timestamps: list[float],
    *,
    max_frames: int,
    min_similarity: float,
    min_gap_seconds: float,
    duplicate_cosine: float,
) -> list[FrameRanking]:
    """Consume normalized vectors aligned with targets/timestamps and return selection rank order.

    A frame matches only text whose padded span contains its timestamp, and eligibility uses that match's raw cosine.
    Ranking subtracts the frame's mean similarity to every unique text at that scale across the whole transcript before RRF.
    A frame is dropped when it lies within the minimum gap of a selected frame or duplicates one by image cosine of at least duplicate_cosine, so a screen recurring minutes apart is selected once.
    Empty eligibility stays empty, and an exact minimum timestamp gap is allowed.
    """
    if max_frames < 1 or not timestamps or not targets:
        return []
    if image_vectors.shape[0] != len(timestamps) or text_vectors.shape[0] != len(
        targets
    ):
        raise ValueError(
            "Embedding rows must align with frame timestamps and text targets"
        )
    times = np.asarray(timestamps, dtype=float)[:, None]
    fused = [0.0] * len(timestamps)
    matches: list[list[TextMatch]] = [[] for _ in timestamps]
    for level in LEVELS:
        indices = [
            index for index, target in enumerate(targets) if target.level == level
        ]
        if not indices:
            continue
        unique: dict[str, int] = {}
        for index in indices:
            unique.setdefault(targets[index].text, index)
        background = (
            (image_vectors @ text_vectors[list(unique.values())].T)
            .astype(float)
            .mean(axis=1)
        )
        starts = np.array([targets[index].start_seconds for index in indices])
        ends = np.array([targets[index].end_seconds for index in indices])
        nearby = (starts - PADDING_BEFORE_SECONDS <= times) & (
            times <= ends + PADDING_AFTER_SECONDS
        )
        cosines = np.where(nearby, image_vectors @ text_vectors[indices].T, -np.inf)
        best = cosines.argmax(axis=1)
        raw_scores = cosines[np.arange(len(timestamps)), best]
        relative_scores = raw_scores - background
        order = sorted(
            (
                index
                for index, score in enumerate(raw_scores)
                if score >= min_similarity
            ),
            key=lambda index: (-relative_scores[index], timestamps[index]),
        )
        rank = 0
        for position, index in enumerate(order):
            if (
                position == 0
                or relative_scores[index] != relative_scores[order[position - 1]]
            ):
                rank = position + 1
            matches[index].append(
                TextMatch(
                    targets[indices[int(best[index])]],
                    float(raw_scores[index]),
                    float(relative_scores[index]),
                    rank,
                )
            )
            fused[index] += 1 / (RRF_CONSTANT + rank)

    order = sorted(
        (index for index, score in enumerate(fused) if score > 0),
        key=lambda index: (
            -fused[index],
            -max(match.cosine for match in matches[index]),
            timestamps[index],
        ),
    )
    selected: list[FrameRanking] = []
    for index in order:
        if any(
            abs(timestamps[index] - timestamps[row.index]) < min_gap_seconds
            or image_vectors[index] @ image_vectors[row.index] >= duplicate_cosine
            for row in selected
        ):
            continue
        selected.append(FrameRanking(index, fused[index], tuple(matches[index])))
        if len(selected) == max_frames:
            break
    return selected
