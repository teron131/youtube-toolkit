"""Score candidate frames with a decision model: each frame, shown with the speech around it or told there is none, is asked whether it shows specific information in full, shows the specific thing being described, or is a moment worth pausing on for what it shows, so finished slides outrank the frames that write them and wordless films rank by what they show.

On four graded narrated videos this score separated the graders' key frames from the rest of the final picks at AUC 0.76, and on a wordless illustrated short it separated a viewer's own picks from the other keyframes at 0.94.
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
from .text import SOUND_ANNOTATION

if TYPE_CHECKING:
    from ..schemas import TranscriptChunk, TranscriptDocument
    from .candidates import StoredFrame

CONTEXT_SECONDS = 30
MARKER = "[THE FRAME APPEARS HERE]"
NO_SPEECH = f"No speech within {CONTEXT_SECONDS} seconds of this frame."
FRAME_ASSET = "frame"
# Measured on the graded videos, in-sample: 0.4 and 0.6 ranked within 0.01 of each other, and dropping illustration hurt only real-world footage.
ILLUSTRATIVE_WEIGHT = 0.4
# Near speech, content worth seeing for itself ranks below shown information, so footage fills the picks after the slides and screens; on the graded videos, no talking head passed on visual worth alone, while 93% of the best slide and code frames scored 0.80 or more as information.
# With no speech near a frame, visual worth counts in full; at 0.4 there, text cards crowded the scenes out of a wordless film.
VISUAL_WEIGHT = 0.4
# On the graded videos 0.30 kept 635 of the 639 frames rated most useful and all 61 key frames, and dropped 26 of 30 talking heads; on wordless films it drops end screens and blank frames.
MIN_SCORE = 0.30
# Keyframes are 3 to 7 seconds apart and slides stay up across several, so a few unscored frames rarely lose content; more than this share means the host is failing and the ranking would have holes.
MAX_UNSCORED_SHARE = 0.1
# Each description lists by category what counts and what does not; against one-line questions this lifted key-frame AUC on the graded videos from 0.70 to 0.76 and spread the visual answers on a wordless short from a median of 0.93 to 0.63.
QUESTIONS = {
    # Excluding subtitles and name captions keeps talking heads with burned-in text from scoring as information, and asking for it in full ranks a finished slide above the frames that write it.
    "has_information": {
        "type": "noul",
        "instructions": "Does the frame show specific information a viewer could read or study, shown in full rather than still being written, typed, or drawn? Counts: slides and title cards with readable text; code, terminals, and app, web, or document screens; charts, tables, and figures with numbers; equations and labeled diagrams; posts, articles, and on-screen lists. Does not count: subtitles or captions of the speech, a speaker's name or channel caption, logos and watermarks, decorative or background text, and text too small, blurred, or partial to read.",
    },
    "illustrative": {
        "type": "noul",
        "instructions": f"Does the frame show the specific thing the narrator describes around {MARKER}? Counts: the actual object, product, device, place, experiment, demo, or process being talked about, or a close view of it. Does not count: generic or stock footage that only sets a mood, a person talking to the camera, or a shot unrelated to the words.",
    },
    "visual": {
        "type": "noul",
        "instructions": "Is the frame a moment worth pausing on for what it shows, where something new has fully appeared and the shot holds on it? Counts: a complete scene, illustration, or animation at its finished state; a reveal of a new place, object, or idea; a finished artwork, photograph, product shot, or generated result; a demo showing its outcome. Does not count: an earlier step still being drawn, written, built, or assembled; transitions such as cuts, fades, swirls, zooms, morphs, or motion blur; people as the main subject, such as talking heads, face close-ups, reactions, or someone walking or working; blank, dark, logo, end, or loading screens.",
    },
}


@dataclass(frozen=True, slots=True)
class FrameDecision:
    score: float
    has_information: float
    illustrative: float
    visual: float


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """Totals over the scored frames; each is None when the host did not report it for every one."""

    input_tokens: int | None
    cost_usd: float | None


def score_frames(
    frames: list[StoredFrame],
    document: TranscriptDocument | None,
    config: DecisionConfig,
) -> tuple[list[FrameDecision | None], DecisionUsage]:
    """Rows align with frames, None where the request failed after retries; more than MAX_UNSCORED_SHARE of them unscored raises instead.

    A None document means the video has no speech to show, and every frame is judged by its image alone.
    """
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
        _decision(row.answers, spoken=bool(_nearby(document, frame.timestamp_seconds)))
        if isinstance(row, DecisionResult)
        else None
        for frame, row in zip(frames, rows, strict=True)
    ]
    return decisions, DecisionUsage(
        None if None in tokens else sum(token for token in tokens if token is not None),
        None if None in costs else sum(cost for cost in costs if cost is not None),
    )


def _context(document: TranscriptDocument | None, timestamp: float) -> str:
    """The marker sits between the lines that start before the frame and those that start after it."""
    lines = _nearby(document, timestamp)
    chapters = document.metadata.chapters if document else []
    spoken = [chunk.text for chunk in lines if chunk.start_seconds <= timestamp]
    spoken.append(MARKER)
    spoken += [chunk.text for chunk in lines if chunk.start_seconds > timestamp]
    chapter = next(
        (
            chapter.title
            for chapter in reversed(chapters)
            if chapter.start_seconds <= timestamp
        ),
        None,
    )
    header = f"Chapter: {chapter}\n" if chapter else ""
    transcript = " ".join(spoken) if lines else NO_SPEECH
    return f"{header}Transcript around a frame from a YouTube video:\n{transcript}"


def _nearby(
    document: TranscriptDocument | None, timestamp: float
) -> list[TranscriptChunk]:
    return [
        chunk
        for chunk in (document.chunks if document else [])
        if chunk.start_seconds - CONTEXT_SECONDS
        <= timestamp
        <= chunk.end_seconds + CONTEXT_SECONDS
        and SOUND_ANNOTATION.sub("", chunk.text).strip()
    ]


def _decision(answers: dict[str, dict[str, Any]], *, spoken: bool) -> FrameDecision:
    # A refused or missing answer carries no probability and counts as no.
    has_information, illustrative, visual = (
        float((answers.get(name) or {}).get("noul") or 0.0) for name in QUESTIONS
    )
    score = max(
        has_information,
        ILLUSTRATIVE_WEIGHT * illustrative,
        (VISUAL_WEIGHT if spoken else 1.0) * visual,
    )
    return FrameDecision(score, has_information, illustrative, visual)
