"""The decision route's steps, which vision/workflow.py runs in order: the questions a decision model answers about candidate frames, how the answers score and keep frames, joining views of one item, and ranking the items.

The picture questions never see the speech, so a narrator naming a title card cannot raise its picture score.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
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
REQUEST_SIZE = (640, 360)
# Pillow releases the GIL while decoding and encoding, so threads speed this up.
ASSET_WORKERS = 4
# Applied to the combined score, so speech can lift a frame over it but never push one under.
MIN_SCORE = 0.30
# The model's own yes. A frame also needs one to some question, since several doubtful answers can add up past MIN_SCORE on a blank page.
YES = 0.5
SAME_ITEM = 0.6
# Items far apart are compared only because their colours match closest among many, which leans the model towards yes, and a wrong yes deletes a scene, so a repeat needs a surer answer than neighbours do.
REPEAT = 0.8
# Colour only shortlists the pairs same_item judges: more finds repeats whose colours changed more, at about one request per item each.
LOOKALIKES = 2
CONTEXT_WEIGHT = 0.25
# The share of its score an item loses at the same moment as a ranked one. Higher lets a weak frame far from every pick outrank a strong one near a pick.
NEARNESS_WEIGHT = 0.25
# Items span several keyframes, so a few failed requests rarely lose content. More than this share failing in any step means the host is failing and the ranking would have holes.
MAX_FAILED_SHARE = 0.1
FRAME_QUESTIONS = {
    "has_information": Noul(
        instructions="Does the frame show specific information a viewer could read or study, shown in full rather than still being written, typed, or drawn? Counts: slides and title cards with readable text; code, terminals, and app, web, or document screens; charts, tables, and figures with numbers; equations and labeled diagrams; posts, articles, and on-screen lists. Does not count: subtitles or captions of the speech, a speaker's name or channel caption, logos and watermarks, decorative or background text, a blank or loading page whose only text is the window, tabs, or toolbar around it, menus, pickers, and lists of titles or links for choosing what to open next, and text too small, blurred, or partial to read."
    ),
    "visual": Noul(
        instructions="Is the frame a moment worth pausing on for what it shows, where something new has fully appeared and the shot holds on it? Counts: a complete scene, illustration, or animation at its finished state; a reveal of a new place, object, or idea; a finished artwork, photograph, product shot, or generated result; a demo showing its outcome. Does not count: an earlier step still being drawn, written, built, or assembled; transitions such as cuts, fades, swirls, zooms, morphs, or motion blur; people as the main subject, such as talking heads, face close-ups, reactions, or someone walking or working; blank, dark, logo, end, or loading screens."
    ),
    "impressive": Noul(
        instructions="Looking at the main content and ignoring the app window around it and any small webcam inset, is it a visually impressive view: a striking, detailed render, scene, artwork, or result, shown fully and clearly? Does not count: main content partly covered by an open settings panel, menu, dialog, or tooltip; plain text, posts, tables, or code; blurry, dark, or half-loaded views."
    ),
    # The questions above judge what is shown, so a split view whose second pane never loaded scores like a finished one; only this one sees the missing pane.
    "loaded": Noul(
        instructions="Ignoring the app window around the main content, including its toolbars, sidebars, empty side columns, and any small webcam inset, has every pane of the main content finished loading? Counts: a single view, or a split or grid view where every pane shows its content, even when that finished content is dark, a night scene, or a sparse page. Does not count: a split or grid view where a pane is still blank, empty, a plain color, or a loading screen, so only some of the compared results show."
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
    """illustrative is None when no speech is near the frame or its request failed, and the picture alone then scores it."""

    score: float
    has_information: float
    visual: float
    impressive: float
    loaded: float
    illustrative: float | None


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """Totals over successful responses only, each None when any of them omitted it."""

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    # Requests run concurrently, so this sum is far longer than the ranking took.
    provider_seconds: float | None


class DecisionRequests:
    """Sends each step's requests and keeps every step's usage and failures.

    More than MAX_FAILED_SHARE of a step's requests failing raises DecisionError; fewer cost what that step's consequence says, and warning names them.
    """

    def __init__(self, config: DecisionConfig) -> None:
        self._config = config
        self._results: list[DecisionResult] = []
        self._warnings: list[str] = []

    def ask(
        self,
        questions: dict[str, Noul],
        inputs: list[DecisionInput],
        noun: str,
        consequence: str,
    ) -> list[dict[str, float] | None]:
        """Each input's answers as probabilities, None where its request failed."""
        rows = (
            decide_each(self._config, questions=questions, inputs=inputs)
            if inputs
            else []
        )
        errors = [row for row in rows if isinstance(row, DecisionError)]
        if len(errors) > MAX_FAILED_SHARE * len(rows):
            raise DecisionError(
                f"{len(errors)} of {len(rows)} {noun} requests failed; first: {errors[0]}"
            ) from errors[0]
        if errors:
            self._warnings.append(
                f"{len(errors)} of {len(rows)} {noun} requests failed after repeated decision host errors, {consequence}."
            )
        self._results.extend(row for row in rows if isinstance(row, DecisionResult))
        return [
            _probabilities(row.answers, questions)
            if isinstance(row, DecisionResult)
            else None
            for row in rows
        ]

    @property
    def usage(self) -> DecisionUsage:
        provider_seconds = _total([result.provider_seconds for result in self._results])
        return DecisionUsage(
            _total([result.input_tokens for result in self._results]),
            _total([result.output_tokens for result in self._results]),
            _total([result.cost_usd for result in self._results]),
            None if provider_seconds is None else round(provider_seconds, 3),
        )

    @property
    def warning(self) -> str | None:
        return " ".join(self._warnings) or None


def frame_assets(frames: list[StoredFrame]) -> list[dict[str, str]]:
    """Each frame shrunk to REQUEST_SIZE and encoded as a request image."""
    with ThreadPoolExecutor(ASSET_WORKERS) as executor:
        return list(executor.map(_asset, [frame.image_path for frame in frames]))


def judge_pictures(
    requests: DecisionRequests, assets: list[dict[str, str]]
) -> list[dict[str, float] | None]:
    """FRAME_QUESTIONS for each frame alone, None where its request failed, which leaves the frame out."""
    return requests.ask(
        FRAME_QUESTIONS,
        [DecisionInput([FRAME_ASSET], {FRAME_ASSET: asset}) for asset in assets],
        "frame",
        "so those frames were left out of the ranking",
    )


def judge_speech(
    requests: DecisionRequests,
    frames: list[StoredFrame],
    assets: list[dict[str, str]],
    pictures: list[dict[str, float] | None],
    document: TranscriptDocument | None,
) -> dict[int, dict[str, float] | None]:
    """illustrative for each frame with speech within CONTEXT_SECONDS that score_frames could still keep, shown with that speech and keyed by frame index.

    A None document means the video has no speech to show, so nothing is asked.
    A failed request leaves None, so the picture alone scores that frame.
    """
    speech: dict[int, str] = {}
    if document is not None:
        spoken = [
            segment
            for segment in document.segments
            if SOUND_ANNOTATION.sub("", segment.text).strip()
        ]
        for index, picture in enumerate(pictures):
            timestamp = frames[index].timestamp_seconds
            # A sure yes is the most speech can add.
            if (
                picture is not None
                and _passes(_decision(picture, {"illustrative": 1.0}))
                and (lines := _nearby(spoken, timestamp))
            ):
                speech[index] = _context(document, timestamp, lines)
    illustrations = requests.ask(
        CONTEXT_QUESTIONS,
        [
            DecisionInput([text, FRAME_ASSET], {FRAME_ASSET: assets[index]})
            for index, text in speech.items()
        ],
        "speech",
        "so those frames were scored by their picture alone",
    )
    return dict(zip(speech, illustrations, strict=True))


def score_frames(
    pictures: list[dict[str, float] | None],
    illustrations: dict[int, dict[str, float] | None],
) -> dict[int, FrameDecision]:
    """The frames that stay, keyed by index in time order: each scores max(has_information, visual) + CONTEXT_WEIGHT x illustrative and needs MIN_SCORE, a YES among those three answers, and a YES to loaded."""
    decisions = (
        (index, _decision(picture, illustrations.get(index)))
        for index, picture in enumerate(pictures)
        if picture is not None
    )
    return {index: decision for index, decision in decisions if _passes(decision)}


@dataclass(frozen=True, slots=True)
class Items:
    """Items in time order, each a run of neighbouring frames that show one thing.

    frames holds each item's chosen frame as a candidate index; ties[n] is the same_item answer between item n and item n + 1, None when their frames are not neighbours or the request failed.
    """

    frames: list[int]
    ties: list[float | None]


def join_neighbours(
    requests: DecisionRequests,
    assets: list[dict[str, str]],
    passing: dict[int, FrameDecision],
) -> Items:
    """same_item for each pair of neighbouring frames that both stay; neighbours at SAME_ITEM or more join one item, which keeps its frame with the highest score plus impressive, and a failed request keeps its pair apart."""
    pairs = [(index - 1, index) for index in passing if index - 1 in passing]
    answers = requests.ask(
        SAME_ITEM_QUESTIONS,
        [_pair(assets[first], assets[second]) for first, second in pairs],
        "same-item",
        "so those neighbours were kept as separate items",
    )
    ties = {
        pair: answer["same_item"]
        for pair, answer in zip(pairs, answers, strict=True)
        if answer is not None
    }
    items: list[list[int]] = []
    for index in passing:
        if ties.get((index - 1, index), 0.0) >= SAME_ITEM:
            items[-1].append(index)
        else:
            items.append([index])
    return Items(
        [
            max(
                item, key=lambda index: passing[index].score + passing[index].impressive
            )
            for item in items
        ],
        [ties.get((before[-1], after[0])) for before, after in pairwise(items)],
    )


def find_repeats(
    requests: DecisionRequests,
    frames: list[StoredFrame],
    assets: list[dict[str, str]],
    items: Items,
) -> frozenset[tuple[int, int]]:
    """Pairs of chosen frames, lower index first, that show one item again far apart: same_item at REPEAT or more for each item and its LOOKALIKES closest items in colour, unless join_neighbours compared them."""
    compared = {
        (number, number + 1) for number, tie in enumerate(items.ties) if tie is not None
    }
    lookalikes = _lookalikes(
        [frames[index].image_path for index in items.frames], compared
    )
    answers = requests.ask(
        SAME_ITEM_QUESTIONS,
        [
            _pair(assets[items.frames[first]], assets[items.frames[second]])
            for first, second in lookalikes
        ],
        "lookalike",
        "so those lookalikes were both kept",
    )
    return frozenset(
        (items.frames[first], items.frames[second])
        for (first, second), answer in zip(lookalikes, answers, strict=True)
        if answer is not None and answer["same_item"] >= REPEAT
    )


def rank_items(
    items: Items,
    passing: dict[int, FrameDecision],
    frames: list[StoredFrame],
    *,
    even_gap_seconds: float,
) -> list[int]:
    """The chosen frames' indexes, best first: items rank one at a time by score, lowered by how surely same_item ties them to an item already ranked and, by up to NEARNESS_WEIGHT, by how near in time they are to one, so things not yet shown and stretches not yet covered come first.

    even_gap_seconds is the gap that evenly spread picks would leave, and is how far a ranked item's nearness reaches.
    A tie across several items is the weakest one on the way, since a repeat needs every step between them to be one.
    The answer and the gap are the weights, so no cutoff decides what counts as a repeat or as too close.
    """
    scores = [passing[index].score for index in items.frames]
    ties = [tie or 0.0 for tie in items.ties]
    times = [frames[index].timestamp_seconds for index in items.frames]
    novelty = [1.0] * len(scores)
    spread = [1.0] * len(scores)
    order: list[int] = []
    unranked = set(range(len(scores)))

    def worth(item: int) -> tuple[float, int]:
        return scores[item] * novelty[item] * spread[item], -item

    while unranked:
        best = max(unranked, key=worth)
        unranked.remove(best)
        order.append(best)
        tie = 1.0
        for item in range(best + 1, len(scores)):
            tie = min(tie, ties[item - 1])
            novelty[item] = min(novelty[item], 1 - tie)
        tie = 1.0
        for item in range(best - 1, -1, -1):
            tie = min(tie, ties[item])
            novelty[item] = min(novelty[item], 1 - tie)
        for item in unranked:
            nearness = max(0.0, 1 - abs(times[item] - times[best]) / even_gap_seconds)
            spread[item] = min(spread[item], 1 - NEARNESS_WEIGHT * nearness)
    return [items.frames[item] for item in order]


def _asset(path: Path) -> dict[str, str]:
    with Image.open(path) as image:
        small = image.convert("RGB")
    small.thumbnail(REQUEST_SIZE)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=85)
    return image_asset(buffer.getvalue())


def _pair(first: dict[str, str], second: dict[str, str]) -> DecisionInput:
    return DecisionInput(
        ["FIRST frame:", "first", "SECOND frame:", "second"],
        {"first": first, "second": second},
    )


def _lookalikes(
    paths: list[Path], compared: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Pairs of positions in paths, lower first, joining each to its LOOKALIKES most similar in colour that compared does not hold."""
    if not paths:
        return []
    shares = np.sqrt(np.stack([_colours(path) for path in paths]))
    # Bhattacharyya overlap of the colour histograms, 1 for identical colours.
    similarity = shares @ shares.T
    pairs: set[tuple[int, int]] = set()
    for item, row in enumerate(similarity):
        closest = [
            (min(item, other), max(item, other))
            for other in map(int, np.argsort(-row, kind="stable"))
            if other != item and (min(item, other), max(item, other)) not in compared
        ]
        pairs.update(closest[:LOOKALIKES])
    return sorted(pairs)


def _colours(path: Path) -> np.ndarray:
    """The share of pixels in each of 512 colour bins."""
    with Image.open(path) as image:
        levels = np.asarray(image.convert("RGB").resize((160, 90))) // 32
    # Eight levels per channel make one bin: (255, 0, 64) -> 7 * 64 + 0 * 8 + 2 = 450.
    bins = levels[..., 0].astype(int) * 64 + levels[..., 1] * 8 + levels[..., 2]
    return np.bincount(bins.ravel(), minlength=512) / bins.size


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
        picture["loaded"],
        illustrative,
    )


def _passes(decision: FrameDecision) -> bool:
    return (
        decision.score >= MIN_SCORE
        and max(decision.has_information, decision.visual, decision.illustrative or 0.0)
        >= YES
        and decision.loaded >= YES
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
