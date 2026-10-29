"""Score candidate frames with a decision model: each frame, shown with the speech around it, is asked whether it adds information the narrator leaves out or shows the specific thing being described.

On four graded videos this score separated frames that add substantial information beyond the transcript from frames that add none at AUC 0.80, where image-embedding ranking scored 0.54.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import DecisionConfig
from ..decisions import (
    DecisionError,
    DecisionInput,
    DecisionResult,
    decide_each,
    image_asset,
)

if TYPE_CHECKING:
    from ..schemas import TranscriptDocument
    from .candidates import StoredFrame

CONTEXT_SECONDS = 30
MARKER = "[THE FRAME APPEARS HERE]"
FRAME_ASSET = "frame"
# Measured on the graded videos, in-sample: 0.4 and 0.6 ranked within 0.01 of each other, and dropping illustration hurt only real-world footage.
ILLUSTRATIVE_WEIGHT = 0.4
# Every frame graders said adds substantial information scored at least 0.32 on every graded video; 0.30 keeps a margin below that and drops most talking heads and transitions.
MIN_SCORE = 0.30
# Keyframes are 3 to 7 seconds apart and slides stay up across several, so a few unscored frames rarely lose content; more than this share means the host is failing and the ranking would have holes.
MAX_UNSCORED_SHARE = 0.1
QUESTIONS = {
    "has_information": {
        "type": "noul",
        "instructions": "Does the frame show specific information such as text, numbers, equations, tables, charts, labeled diagrams, code, or an interface?",
    },
    "unspoken": {
        "type": "noul",
        "instructions": f"Does the frame show information that the narrator does not say in the transcript around {MARKER}?",
    },
    "illustrative": {
        "type": "noul",
        "instructions": f"Does the frame show the specific object, demo, place, or process the narrator describes around {MARKER}, rather than generic footage or a person talking?",
    },
}


@dataclass(frozen=True, slots=True)
class FrameDecision:
    score: float
    has_information: float
    unspoken: float
    illustrative: float


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """Totals over the scored frames; each is None when the host did not report it for every one."""

    input_tokens: int | None
    cost_usd: float | None


def score_frames(
    frames: list[StoredFrame],
    document: TranscriptDocument,
    config: DecisionConfig,
) -> tuple[list[FrameDecision | None], DecisionUsage]:
    """Rows align with frames, None where the request failed after retries; more than MAX_UNSCORED_SHARE of them unscored raises instead."""
    rows = decide_each(
        config,
        questions=QUESTIONS,
        inputs=[
            DecisionInput(
                [_context(document, frame.timestamp_seconds), FRAME_ASSET],
                {FRAME_ASSET: image_asset(frame.image_path)},
            )
            for frame in frames
        ],
    )
    errors = [row for row in rows if isinstance(row, DecisionError)]
    if len(errors) > MAX_UNSCORED_SHARE * len(frames):
        raise DecisionError(
            f"{len(errors)} of {len(frames)} frame requests failed; first: {errors[0]}"
        ) from errors[0]
    results = [row for row in rows if isinstance(row, DecisionResult)]
    tokens = [result.input_tokens for result in results]
    costs = [result.cost_usd for result in results]
    decisions = [
        _decision(row.answers) if isinstance(row, DecisionResult) else None
        for row in rows
    ]
    return decisions, DecisionUsage(
        None if None in tokens else sum(token for token in tokens if token is not None),
        None if None in costs else sum(cost for cost in costs if cost is not None),
    )


def _context(document: TranscriptDocument, timestamp: float) -> str:
    """The marker sits between the lines that start before the frame and those that start after it."""
    lines = [
        chunk
        for chunk in document.chunks
        if chunk.start_seconds - CONTEXT_SECONDS
        <= timestamp
        <= chunk.end_seconds + CONTEXT_SECONDS
    ]
    spoken = [chunk.text for chunk in lines if chunk.start_seconds <= timestamp]
    spoken.append(MARKER)
    spoken += [chunk.text for chunk in lines if chunk.start_seconds > timestamp]
    chapter = next(
        (
            chapter.title
            for chapter in reversed(document.metadata.chapters)
            if chapter.start_seconds <= timestamp
        ),
        None,
    )
    header = f"Chapter: {chapter}\n" if chapter else ""
    return (
        f"{header}Transcript around a frame from a YouTube video:\n{' '.join(spoken)}"
    )


def _decision(answers: dict[str, dict[str, Any]]) -> FrameDecision:
    # A refused or missing answer carries no probability and counts as no.
    has_information, unspoken, illustrative = (
        float((answers.get(name) or {}).get("noul") or 0.0) for name in QUESTIONS
    )
    return FrameDecision(
        max(has_information * unspoken, ILLUSTRATIVE_WEIGHT * illustrative),
        has_information,
        unspoken,
        illustrative,
    )
