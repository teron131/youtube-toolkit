"""Turn the full timed transcript into keyword, sentence, and paragraph targets that Gemini ranking matches frames against."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from typing import Literal

from yake import KeywordExtractor

from ..schemas import TranscriptChunk, TranscriptDocument

CAPTION_GAP_SECONDS = 5
SENTENCE_SECONDS = 15
PARAGRAPH_SECONDS = 45
SENTENCE_ENDINGS = (".", "?", "!", "\u3002", "\uff1f", "\uff01")
ENGLISH_HESITATIONS = frozenset({"uh", "um", "hmm"})


@dataclass(frozen=True, slots=True)
class TextTarget:
    level: Literal["keyword", "sentence", "paragraph"]
    start_seconds: float
    end_seconds: float
    text: str


def prepare_targets(
    document: TranscriptDocument,
    excluded_intervals: Sequence[tuple[float, float]],
) -> list[TextTarget]:
    """Retain whole caption text once per sentence/paragraph scale, with real timing clipped at video and section boundaries.

    Caption chunks are indivisible, so an unusually long chunk can exceed the nominal grouping duration.
    Excluded intervals are half-open and remove chunks by their start time.
    """
    segments = _segments(document, excluded_intervals)
    if not segments:
        return []
    sentences: list[TextTarget] = []
    paragraphs: list[TextTarget] = []
    for segment in segments:
        segment_sentences = _group_sentences(segment)
        sentences.extend(segment_sentences)
        paragraphs.extend(_group_paragraphs(segment_sentences))
    return [*_keywords(paragraphs, document.language), *sentences, *paragraphs]


def _segments(
    document: TranscriptDocument,
    excluded_intervals: Sequence[tuple[float, float]],
) -> list[list[TranscriptChunk]]:
    """Runs of timed chunks that share a chapter or excluded-interval section and have no caption gap over CAPTION_GAP_SECONDS, with ends clipped to that section."""
    duration = document.metadata.duration_seconds
    bound = (
        duration
        if duration is not None and math.isfinite(duration) and duration > 0
        else math.inf
    )
    boundaries = sorted(
        {
            boundary
            for chapter in document.metadata.chapters
            for boundary in (chapter.start_seconds, chapter.end_seconds)
            if boundary is not None and math.isfinite(boundary) and 0 < boundary < bound
        }
        | {
            boundary
            for interval in excluded_intervals
            for boundary in interval
            if math.isfinite(boundary) and 0 < boundary < bound
        }
    )
    chunks = sorted(
        (
            chunk
            for chunk in document.chunks
            if math.isfinite(chunk.start_seconds)
            and math.isfinite(chunk.end_seconds)
            and 0 <= chunk.start_seconds < bound
            and chunk.end_seconds > chunk.start_seconds
            and chunk.text.strip()
            and not any(
                start <= chunk.start_seconds < end for start, end in excluded_intervals
            )
        ),
        key=lambda chunk: chunk.start_seconds,
    )
    segments: list[list[TranscriptChunk]] = []
    previous_section = -1
    segment_end = 0.0
    for chunk in chunks:
        section = bisect_right(boundaries, chunk.start_seconds)
        end = min(
            chunk.end_seconds,
            boundaries[section] if section < len(boundaries) else bound,
        )
        if (
            not segments
            or section != previous_section
            or chunk.start_seconds - segment_end > CAPTION_GAP_SECONDS
        ):
            segments.append([])
            segment_end = end
        segments[-1].append(replace(chunk, end_seconds=end, text=chunk.text.strip()))
        segment_end = max(segment_end, end)
        previous_section = section
    return segments


def _group_sentences(segment: list[TranscriptChunk]) -> list[TextTarget]:
    sentences: list[TextTarget] = []
    group: list[TranscriptChunk] = []
    for chunk in segment:
        group.append(chunk)
        if (
            chunk.text.rstrip("\"')]}\u2019\u201d").endswith(SENTENCE_ENDINGS)
            or max(item.end_seconds for item in group) - group[0].start_seconds
            >= SENTENCE_SECONDS
        ):
            sentences.append(_sentence(group))
            group = []
    if group:
        sentences.append(_sentence(group))
    return sentences


def _group_paragraphs(sentences: list[TextTarget]) -> list[TextTarget]:
    paragraphs: list[TextTarget] = []
    group: list[TextTarget] = []
    for sentence in sentences:
        if (
            group
            and max(sentence.end_seconds, max(item.end_seconds for item in group))
            - group[0].start_seconds
            > PARAGRAPH_SECONDS
        ):
            paragraphs.append(_paragraph(group))
            group = []
        group.append(sentence)
    if group:
        paragraphs.append(_paragraph(group))
    return paragraphs


def _keywords(paragraphs: list[TextTarget], language: str) -> list[TextTarget]:
    extractor = _keyword_extractor(language)
    keywords = []
    for paragraph in paragraphs:
        seen: set[str] = set()
        for phrase, _score in extractor.extract_keywords(paragraph.text):
            phrase = " ".join(phrase.split())
            identity = phrase.casefold()
            if not phrase or identity in seen or identity in extractor.stopword_set:
                continue
            seen.add(identity)
            keywords.append(
                TextTarget(
                    "keyword", paragraph.start_seconds, paragraph.end_seconds, phrase
                )
            )
    return keywords


def _keyword_extractor(language: str) -> KeywordExtractor:
    language = language.partition("-")[0].lower()
    extractor = KeywordExtractor(lan=language, n=3, top=5)
    if language == "en":
        extractor.stopword_set = extractor.stopword_set | ENGLISH_HESITATIONS
    return extractor


def _sentence(chunks: list[TranscriptChunk]) -> TextTarget:
    return TextTarget(
        "sentence",
        chunks[0].start_seconds,
        max(chunk.end_seconds for chunk in chunks),
        " ".join(chunk.text for chunk in chunks),
    )


def _paragraph(sentences: list[TextTarget]) -> TextTarget:
    return TextTarget(
        "paragraph",
        sentences[0].start_seconds,
        max(sentence.end_seconds for sentence in sentences),
        " ".join(sentence.text for sentence in sentences),
    )
