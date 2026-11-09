"""Rank candidate frames with a decision model that judges each picture on its own and asks separately whether it shows what is being said: the picture questions see only the image, so naming a title card cannot raise its picture score, while the speech question lets the frame showing the thing described win over its neighbours, and neighbouring frames showing the same item collapse to one view.

On three blind-graded 10-minute clips, a creator talking over screens, a model review of 3D demos, and a documentary with real-world footage, this kept 54 different screenshots in 67 picks, against 50 in 66 when the speech only reordered items the picture had chosen and 50 in 64 when has_information also read the speech; the design was chosen after seeing those grades.
On graded explainers its score ranked key frames at AUC 0.761 against 0.754 for one speech-weighted score per frame.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
from typing import TYPE_CHECKING

from PIL import Image
from typesafe_sdk import Answer, Noul, NoulAnswer

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
    from pathlib import Path

    from ..schemas import TranscriptDocument, TranscriptSegment
    from .candidates import StoredFrame

CONTEXT_SECONDS = 30
MARKER = "[THE FRAME APPEARS HERE]"
FRAME_ASSET = "frame"
# Questions were measured on frames sent at this size, which bills 397 tokens per image against 687 at 960x540.
REQUEST_SIZE = (640, 360)
# Pillow decodes, shrinks, and encodes outside the GIL: 150 frames took 0.29 s on 4 threads against 0.86 s one by one.
ASSET_WORKERS = 4
# Applied to the score, which the speech can only raise; from the picture alone it dropped 70 of 1,334 keyframes on four narrated screen videos and none of the 461 graded picks.
MIN_SCORE = 0.30
# On consecutive graded picks from those videos, 0.6 joined 86% of the graders' repeats and 24% of different screens, mostly benign pairs such as a loading screen and its scene or a prompt and its result.
SAME_ITEM = 0.6
# On graded explainer frames, picture score plus 0.25 x illustrative ranked key frames at AUC 0.761, against 0.712 for the picture alone and 0.717 at a weight of 1.
CONTEXT_WEIGHT = 0.25
# Keyframes are 3 to 7 seconds apart and items span several, so a few failed requests rarely lose content; more than this share in any step means the host is failing and the ranking would have holes.
MAX_FAILED_SHARE = 0.1
# Each description lists by category what counts and what does not; against one-line questions this lifted key-frame AUC on the graded videos from 0.70 to 0.76 and spread the visual answers on a wordless short from a median of 0.93 to 0.63.
FRAME_QUESTIONS = {
    # Excluding subtitles and name captions keeps talking heads with burned-in text from scoring as information, and asking for it in full ranks a finished slide above the frames that write it.
    "has_information": Noul(
        instructions="Does the frame show specific information a viewer could read or study, shown in full rather than still being written, typed, or drawn? Counts: slides and title cards with readable text; code, terminals, and app, web, or document screens; charts, tables, and figures with numbers; equations and labeled diagrams; posts, articles, and on-screen lists. Does not count: subtitles or captions of the speech, a speaker's name or channel caption, logos and watermarks, decorative or background text, and text too small, blurred, or partial to read."
    ),
    "visual": Noul(
        instructions="Is the frame a moment worth pausing on for what it shows, where something new has fully appeared and the shot holds on it? Counts: a complete scene, illustration, or animation at its finished state; a reveal of a new place, object, or idea; a finished artwork, photograph, product shot, or generated result; a demo showing its outcome. Does not count: an earlier step still being drawn, written, built, or assembled; transitions such as cuts, fades, swirls, zooms, morphs, or motion blur; people as the main subject, such as talking heads, face close-ups, reactions, or someone walking or working; blank, dark, logo, end, or loading screens."
    ),
    # Judging the content rather than the frame keeps app windows and webcam insets from marking down every screen recording; it put clean views of a rendered scene at 0.61-0.73 and the same scene behind a settings panel at 0.43-0.56.
    "impressive": Noul(
        instructions="Looking at the main content and ignoring the app window around it and any small webcam inset, is it a visually impressive view: a striking, detailed render, scene, artwork, or result, shown fully and clearly? Does not count: main content partly covered by an open settings panel, menu, dialog, or tooltip; plain text, posts, tables, or code; blurry, dark, or half-loaded views."
    ),
}
# Any view of one item counts as the same, since a scrolled page, a rotated 3D scene, and a webcam toggled on change most pixels while showing nothing new.
SAME_ITEM_QUESTIONS = {
    "same_item": Noul(
        instructions="Do the FIRST and SECOND frames show the same item, so one screenshot of it would do? The item is the thing being shown: one post, article, page, chart, table, prompt and its result, comparison, product, or scene. Counts as the same item: any other view of it, such as scrolled, zoomed, panned, rotated, seen from another camera angle, highlighted, or with a panel, menu, stats bar, cursor, or webcam added or removed. Does not count: a different post, page, chart, prompt, result, or scene, even in the same app or layout."
    )
}
CONTEXT_QUESTIONS = {
    "illustrative": Noul(
        instructions=f"Does the frame show the specific thing the narrator describes around {MARKER}? Counts: the actual object, product, device, place, experiment, demo, or process being talked about, or a close view of it. Does not count: generic or stock footage that only sets a mood, a person talking to the camera, or a shot unrelated to the words."
    )
}


@dataclass(frozen=True, slots=True)
class FrameDecision:
    """score gates and ranks frames; score plus impressive chooses the representative within an item.

    illustrative is None when no speech is near the frame or its request failed, and the picture alone then scores it.
    """

    score: float
    has_information: float
    visual: float
    impressive: float
    illustrative: float | None


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """Totals over successful responses, so failed attempts are not counted; each is None when any successful response omitted it."""

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    # Requests run concurrently, so this sum is far longer than the ranking took.
    provider_seconds: float | None


@dataclass(frozen=True, slots=True)
class DecisionRanking:
    """One frame per item, best first, keyed by its index among the candidates; warning names requests that failed after retries and what that cost."""

    frames: list[tuple[int, FrameDecision]]
    usage: DecisionUsage
    warning: str | None


def rank_frames(
    frames: list[StoredFrame],
    document: TranscriptDocument | None,
    config: DecisionConfig,
) -> DecisionRanking:
    """Steps:

    1. Each frame alone: has_information, visual, and impressive.
    2. Each of those frames with speech within CONTEXT_SECONDS: illustrative, shown with that speech.
    3. Score: max(has_information, visual) + CONTEXT_WEIGHT x illustrative; frames under MIN_SCORE drop out.
    4. Each pair of neighbouring frames that both remain: same_item; neighbours at SAME_ITEM or more join one item, which keeps its frame with the highest score plus impressive.

    Items rank by score; a None document means the video has no speech to show, so step 2 asks nothing.
    A failed request leaves its frame out, scores it by the picture alone, or keeps its pair apart; more than MAX_FAILED_SHARE failing in any step raises.
    """
    with ThreadPoolExecutor(ASSET_WORKERS) as executor:
        assets = list(executor.map(_asset, [frame.image_path for frame in frames]))
    results: list[DecisionResult] = []
    warnings: list[str] = []

    def ask(
        questions: dict[str, Noul],
        inputs: list[DecisionInput],
        noun: str,
        consequence: str,
    ) -> list[dict[str, float] | None]:
        rows = decide_each(config, questions=questions, inputs=inputs) if inputs else []
        errors = [row for row in rows if isinstance(row, DecisionError)]
        if len(errors) > MAX_FAILED_SHARE * len(rows):
            raise DecisionError(
                f"{len(errors)} of {len(rows)} {noun} requests failed; first: {errors[0]}"
            ) from errors[0]
        if errors:
            warnings.append(
                f"{len(errors)} of {len(rows)} {noun} requests failed after repeated decision host errors, {consequence}."
            )
        results.extend(row for row in rows if isinstance(row, DecisionResult))
        return [
            _probabilities(row.answers, questions)
            if isinstance(row, DecisionResult)
            else None
            for row in rows
        ]

    pictures = ask(
        FRAME_QUESTIONS,
        [DecisionInput([FRAME_ASSET], {FRAME_ASSET: asset}) for asset in assets],
        "frame",
        "so those frames were left out of the ranking",
    )
    speech: dict[int, str] = {}
    if document is not None:
        spoken = [
            segment
            for segment in document.segments
            if SOUND_ANNOTATION.sub("", segment.text).strip()
        ]
        for index, picture in enumerate(pictures):
            timestamp = frames[index].timestamp_seconds
            if picture is not None and (lines := _nearby(spoken, timestamp)):
                speech[index] = _context(document, timestamp, lines)
    illustrations = ask(
        CONTEXT_QUESTIONS,
        [
            DecisionInput([text, FRAME_ASSET], {FRAME_ASSET: assets[index]})
            for index, text in speech.items()
        ],
        "speech",
        "so those frames were scored by their picture alone",
    )
    contexts = dict(zip(speech, illustrations, strict=True))
    decisions = {
        index: _decision(picture, contexts.get(index))
        for index, picture in enumerate(pictures)
        if picture is not None
    }
    passing = {
        index: decision
        for index, decision in decisions.items()
        if decision.score >= MIN_SCORE
    }
    pairs = [(index - 1, index) for index in passing if index - 1 in passing]
    same_item = ask(
        SAME_ITEM_QUESTIONS,
        [
            DecisionInput(
                ["FIRST frame:", "first", "SECOND frame:", "second"],
                {"first": assets[first], "second": assets[second]},
            )
            for first, second in pairs
        ],
        "same-item",
        "so those neighbours were kept as separate items",
    )
    joined = {
        second
        for (_, second), answer in zip(pairs, same_item, strict=True)
        if answer is not None and answer["same_item"] >= SAME_ITEM
    }
    items: list[list[int]] = []
    for index in passing:
        if index in joined:
            items[-1].append(index)
        else:
            items.append([index])
    kept = [
        max(item, key=lambda index: passing[index].score + passing[index].impressive)
        for item in items
    ]
    kept.sort(
        key=lambda index: (-passing[index].score, frames[index].timestamp_seconds)
    )
    provider_seconds = _total([result.provider_seconds for result in results])
    return DecisionRanking(
        [(index, passing[index]) for index in kept],
        DecisionUsage(
            _total([result.input_tokens for result in results]),
            _total([result.output_tokens for result in results]),
            _total([result.cost_usd for result in results]),
            None if provider_seconds is None else round(provider_seconds, 3),
        ),
        " ".join(warnings) or None,
    )


def _asset(path: Path) -> dict[str, str]:
    with Image.open(path) as image:
        small = image.convert("RGB")
    small.thumbnail(REQUEST_SIZE)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=85)
    return image_asset(buffer.getvalue())


def _decision(
    picture: dict[str, float], context: dict[str, float] | None
) -> FrameDecision:
    illustrative = None if context is None else context["illustrative"]
    return FrameDecision(
        max(picture["has_information"], picture["visual"])
        + CONTEXT_WEIGHT * (illustrative or 0.0),
        picture["has_information"],
        picture["visual"],
        picture["impressive"],
        illustrative,
    )


def _probabilities(
    answers: dict[str, Answer], questions: dict[str, Noul]
) -> dict[str, float]:
    # A missing answer counts as no.
    return {
        name: answer.noul
        if isinstance(answer := answers.get(name), NoulAnswer)
        else 0.0
        for name in questions
    }


def _context(
    document: TranscriptDocument, timestamp: float, lines: list[TranscriptSegment]
) -> str:
    """The marker sits between the lines that start before the frame and those that start after it."""
    spoken = [segment.text for segment in lines if segment.start_seconds <= timestamp]
    spoken.append(MARKER)
    spoken += [segment.text for segment in lines if segment.start_seconds > timestamp]
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


def _nearby(
    segments: list[TranscriptSegment], timestamp: float
) -> list[TranscriptSegment]:
    return [
        segment
        for segment in segments
        if segment.start_seconds - CONTEXT_SECONDS
        <= timestamp
        <= segment.end_seconds + CONTEXT_SECONDS
    ]


def _total[N: (int, float)](values: list[N | None]) -> N | None:
    present = [value for value in values if value is not None]
    return sum(present) if len(present) == len(values) else None
