"""Filter usable speech and turn the timed transcript into keyword, sentence, and paragraph targets for embedding ranking."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import re
from typing import Literal

from yake import KeywordExtractor

from ..schemas import TranscriptDocument, TranscriptSegment

CAPTION_GAP_SECONDS = 5
SENTENCE_SECONDS = 15
PARAGRAPH_SECONDS = 45
SENTENCE_ENDINGS = (".", "?", "!", "\u3002", "\uff1f", "\uff01")
ENGLISH_HESITATIONS = frozenset({"uh", "um", "hmm"})
# Captions mark sounds as [Music] or [Applause], music with note symbols, and speaker changes with >>.
SOUND_ANNOTATION = re.compile(r"\[[^\]]*\]|[\u266a\u266b]|>>")
CJK_RANGES = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
# Each CJK character counts as a word, since those scripts put no spaces between words, and other words stop where CJK text begins.
SPEECH_WORD = re.compile(rf"[{CJK_RANGES}]|[^\W_{CJK_RANGES}]+")
# Narration runs 100-200 words a minute; under 10 is automatic captioning hearing words in music, such as "Heat. Heat." over a wordless film, not speech to rank against.
MIN_SPEECH_WORDS_PER_MINUTE = 10


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

    Caption segments are indivisible, so an unusually long segment can exceed the nominal grouping duration.
    Excluded intervals are half-open and remove segments by their start time.
    Sound annotations are not speech, and a video of known duration with under MIN_SPEECH_WORDS_PER_MINUTE of speech has no targets at all.
    """
    runs = _runs(document, excluded_intervals)
    duration = document.metadata.duration_seconds
    words = sum(
        len(SPEECH_WORD.findall(segment.text)) for run in runs for segment in run
    )
    if not runs or (
        duration is not None
        and math.isfinite(duration)
        and words < MIN_SPEECH_WORDS_PER_MINUTE * duration / 60
    ):
        return []
    sentences: list[TextTarget] = []
    paragraphs: list[TextTarget] = []
    for run in runs:
        run_sentences = _group_sentences(run)
        sentences.extend(run_sentences)
        paragraphs.extend(_group_paragraphs(run_sentences))
    return [*_keywords(paragraphs, document.language), *sentences, *paragraphs]


def _runs(
    document: TranscriptDocument,
    excluded_intervals: Sequence[tuple[float, float]],
) -> list[list[TranscriptSegment]]:
    """Runs of timed segments that share a chapter or excluded-interval section and have no caption gap over CAPTION_GAP_SECONDS, with ends clipped to that section."""
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
    spoken = (
        replace(segment, text=" ".join(SOUND_ANNOTATION.sub(" ", segment.text).split()))
        for segment in document.segments
    )
    segments = sorted(
        (
            segment
            for segment in spoken
            if math.isfinite(segment.start_seconds)
            and math.isfinite(segment.end_seconds)
            and 0 <= segment.start_seconds < bound
            and segment.end_seconds > segment.start_seconds
            and segment.text
            and not any(
                start <= segment.start_seconds < end
                for start, end in excluded_intervals
            )
        ),
        key=lambda segment: segment.start_seconds,
    )
    runs: list[list[TranscriptSegment]] = []
    previous_section = -1
    run_end = 0.0
    for segment in segments:
        section = bisect_right(boundaries, segment.start_seconds)
        end = min(
            segment.end_seconds,
            boundaries[section] if section < len(boundaries) else bound,
        )
        if (
            not runs
            or section != previous_section
            or segment.start_seconds - run_end > CAPTION_GAP_SECONDS
        ):
            runs.append([])
            run_end = end
        runs[-1].append(replace(segment, end_seconds=end))
        run_end = max(run_end, end)
        previous_section = section
    return runs


def _group_sentences(run: list[TranscriptSegment]) -> list[TextTarget]:
    sentences: list[TextTarget] = []
    group: list[TranscriptSegment] = []
    for segment in run:
        group.append(segment)
        if (
            segment.text.rstrip("\"')]}\u2019\u201d").endswith(SENTENCE_ENDINGS)
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


def _sentence(segments: list[TranscriptSegment]) -> TextTarget:
    return TextTarget(
        "sentence",
        segments[0].start_seconds,
        max(segment.end_seconds for segment in segments),
        " ".join(segment.text for segment in segments),
    )


def _paragraph(sentences: list[TextTarget]) -> TextTarget:
    return TextTarget(
        "paragraph",
        sentences[0].start_seconds,
        max(sentence.end_seconds for sentence in sentences),
        " ".join(sentence.text for sentence in sentences),
    )
