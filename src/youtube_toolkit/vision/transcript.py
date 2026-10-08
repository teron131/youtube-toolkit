"""Rank timed transcript passages to locate visual search windows using chapter-local text statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
import re

from ..config import DEFAULT_VISION_MAX_FRAMES
from ..schemas import TranscriptChunk, TranscriptDocument

PASSAGE_SECONDS = 25
CAPTION_GAP_SECONDS = 5
REDUNDANCY_THRESHOLD = 0.92
# Function words contribute little to topical relevance; no visual-reference phrases are matched.
STOP_WORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her here him his how i if in into is it its just may me more my not of on
    or our she so some than that the their them then there these they this those
    to too us was we were what when where which who will with would you your
    """.split()  # noqa: SIM905 -- Keep the stop-word vocabulary compact and readable.
)

TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[^\W_]+", re.UNICODE)
SPONSOR_TITLE = re.compile(r"\b(?:sponsor(?:ed|ship)?|advertisement|ad break)\b", re.I)


@dataclass(frozen=True, slots=True)
class TranscriptWindow:
    """A real caption time range with chapter relevance, not semantic importance or confidence."""

    start_seconds: float
    end_seconds: float
    text: str
    text_score: float
    chapter: str | None


@dataclass(frozen=True, slots=True)
class _Section:
    start: float
    end: float
    title: str | None


def select_transcript_windows(
    document: TranscriptDocument, *, max_windows: int = DEFAULT_VISION_MAX_FRAMES
) -> list[TranscriptWindow]:
    """Select representative passages across chapters without inventing timing for untimed STT.

    Captions are grouped at chapter boundaries and long pauses before sublinear TF-IDF ranks them against their chapter centroid.
    Each chapter receives a first opportunity before additional, nonredundant passages fill the remaining allowance.
    Missing chapters use equal time sections to retain coverage without pretending to infer topics.
    """
    if max_windows < 1:
        return []
    duration = document.metadata.duration_seconds
    bound = (
        duration
        if duration is not None and math.isfinite(duration) and duration > 0
        else math.inf
    )
    chunks = sorted(
        (
            replace(chunk, end_seconds=min(chunk.end_seconds, bound))
            for chunk in document.chunks
            if math.isfinite(chunk.start_seconds)
            and math.isfinite(chunk.end_seconds)
            and 0 <= chunk.start_seconds < bound
            and chunk.end_seconds > chunk.start_seconds
            and chunk.text.strip()
        ),
        key=lambda chunk: chunk.start_seconds,
    )
    if not chunks:
        return []
    sections = _sections(document, chunks, max_windows)
    groups = [_passages(chunks, section) for section in sections]
    groups = [group for group in groups if group]
    if not groups:
        return []
    passages = [passage for group in groups for passage in group]
    vectors = _tfidf(passages)
    ranked: list[list[int]] = []
    offset = 0
    for group in groups:
        indices = list(range(offset, offset + len(group)))
        centroid: dict[str, float] = {}
        for index in indices:
            for token, weight in vectors[index].items():
                centroid[token] = centroid.get(token, 0.0) + weight
        center = _normalize(centroid)
        for index in indices:
            passages[index] = replace(
                passages[index], text_score=_similarity(vectors[index], center)
            )
        ranked.append(
            sorted(indices, key=lambda index: (-passages[index].text_score, index))
        )
        offset += len(group)

    # Choose evenly spaced chapter slots when the chapter count exceeds the allowance.
    if len(ranked) > max_windows:
        slots = (
            [len(ranked) // 2]
            if max_windows == 1
            else [
                round(slot * (len(ranked) - 1) / (max_windows - 1))
                for slot in range(max_windows)
            ]
        )
        ranked = [ranked[slot] for slot in slots]
    selected = [indices.pop(0) for indices in ranked]
    counts = [1] * len(ranked)
    while len(selected) < max_windows:
        eligible = [
            (group_number, index)
            for group_number, indices in enumerate(ranked)
            for index in indices
            if all(
                passages[index].end_seconds <= passages[other].start_seconds
                or passages[index].start_seconds >= passages[other].end_seconds
                for other in selected
            )
            and all(
                _similarity(vectors[index], vectors[other]) < REDUNDANCY_THRESHOLD
                for other in selected
            )
        ]
        if not eligible:
            break
        # Fill chapters with the fewest selections first, then use chapter-local relevance.
        group_number, index = min(
            eligible,
            key=lambda item: (
                counts[item[0]],
                -passages[item[1]].text_score,
                passages[item[1]].start_seconds,
            ),
        )
        selected.append(index)
        counts[group_number] += 1
        ranked[group_number].remove(index)
    return sorted(
        (passages[index] for index in selected), key=lambda item: item.start_seconds
    )


def _sections(
    document: TranscriptDocument, chunks: list[TranscriptChunk], max_windows: int
) -> list[_Section]:
    end = max(chunk.end_seconds for chunk in chunks)
    chapters = sorted(
        (
            chapter
            for chapter in document.metadata.chapters
            if math.isfinite(chapter.start_seconds) and 0 <= chapter.start_seconds < end
        ),
        key=lambda chapter: chapter.start_seconds,
    )
    if not chapters:
        start = chunks[0].start_seconds
        count = min(max_windows, max(1, math.ceil((end - start) / 60)))
        width = (end - start) / count
        return [
            _Section(start + slot * width, start + (slot + 1) * width, None)
            for slot in range(count)
        ]
    sections = []
    if chapters[0].start_seconds > chunks[0].start_seconds:
        sections.append(
            _Section(chunks[0].start_seconds, chapters[0].start_seconds, None)
        )
    for index, chapter in enumerate(chapters):
        next_start = (
            chapters[index + 1].start_seconds if index + 1 < len(chapters) else end
        )
        chapter_end = chapter.end_seconds
        boundary = (
            min(next_start, chapter_end)
            if chapter_end is not None
            and math.isfinite(chapter_end)
            and chapter_end > chapter.start_seconds
            else next_start
        )
        if not SPONSOR_TITLE.search(chapter.title) and boundary > chapter.start_seconds:
            sections.append(_Section(chapter.start_seconds, boundary, chapter.title))
    return sections


def _passages(
    chunks: list[TranscriptChunk], section: _Section
) -> list[TranscriptWindow]:
    passages = []
    pending: list[TranscriptChunk] = []
    for chunk in chunks:
        if not section.start <= chunk.start_seconds < section.end:
            continue
        if pending and (
            chunk.start_seconds - pending[-1].end_seconds > CAPTION_GAP_SECONDS
            or chunk.start_seconds - pending[0].start_seconds >= PASSAGE_SECONDS
        ):
            passages.append(_window(pending, section))
            pending = []
        pending.append(chunk)
    if pending:
        passages.append(_window(pending, section))
    return passages


def _window(chunks: list[TranscriptChunk], section: _Section) -> TranscriptWindow:
    return TranscriptWindow(
        start_seconds=chunks[0].start_seconds,
        end_seconds=min(section.end, max(chunk.end_seconds for chunk in chunks)),
        text=" ".join(chunk.text.strip() for chunk in chunks),
        text_score=0.0,
        chapter=section.title,
    )


def _tfidf(passages: list[TranscriptWindow]) -> list[dict[str, float]]:
    counts = [
        Counter(
            token
            for token in TOKEN_PATTERN.findall(passage.text.casefold())
            if token not in STOP_WORDS and not token.isdecimal()
        )
        for passage in passages
    ]
    frequency = Counter(token for count in counts for token in count)
    return [
        _normalize(
            {
                token: (1 + math.log(count))
                * (1 + math.log((1 + len(passages)) / (1 + frequency[token])))
                for token, count in passage.items()
            }
        )
        for passage in counts
    ]


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {token: value / norm for token, value in vector.items()} if norm else {}


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    return min(
        1.0,
        max(0.0, sum(value * right.get(token, 0.0) for token, value in left.items())),
    )
