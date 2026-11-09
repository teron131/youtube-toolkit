"""Vision text targets filter sound annotations and sparse speech while preserving usable caption timing and words at each matching scale."""

import math

import pytest

from youtube_toolkit.schemas import (
    TranscriptDocument,
    TranscriptSegment,
    VideoChapter,
    VideoMetadata,
)
from youtube_toolkit.vision import text as text_preparation
from youtube_toolkit.vision.text import TextTarget, prepare_targets


def document(segments, chapters=(), *, duration=None, language="en-orig"):
    return TranscriptDocument(
        VideoMetadata(
            "video", "url", duration_seconds=duration, chapters=list(chapters)
        ),
        language,
        "automatic_captions",
        segments,
    )


def test_targets_preserve_all_caption_words_once_per_passage_scale():
    segments = [
        TranscriptSegment(12, 18, "Then examine the lens."),
        TranscriptSegment(0, 6, "First inspect"),
        TranscriptSegment(5, 12, "the light source."),
        TranscriptSegment(18, 24, "Finally compare the images."),
    ]
    targets = prepare_targets(document(segments, duration=22), [])
    expected = "First inspect the light source. Then examine the lens. Finally compare the images."
    for level in ("sentence", "paragraph"):
        passages = [item for item in targets if item.level == level]
        assert " ".join(item.text for item in passages) == expected
        assert passages[0].start_seconds == 0
        assert passages[-1].end_seconds == 22
        assert all(
            0 <= item.start_seconds < item.end_seconds <= 22 for item in passages
        )
    paragraphs = [item for item in targets if item.level == "paragraph"]
    for keyword in (item for item in targets if item.level == "keyword"):
        assert any(
            (keyword.start_seconds, keyword.end_seconds)
            == (paragraph.start_seconds, paragraph.end_seconds)
            for paragraph in paragraphs
        )


def test_targets_break_at_chapters_caption_gaps_and_exclusions():
    segments = [
        TranscriptSegment(0, 8, "First topic"),
        TranscriptSegment(8, 14, "before chapter boundary"),
        TranscriptSegment(10, 15, "Second topic"),
        TranscriptSegment(20, 23, "Excluded advertisement"),
        TranscriptSegment(23, 26, "Still excluded"),
        TranscriptSegment(25, 28, "Return to discussion"),
        TranscriptSegment(40, 44, "After a long pause"),
    ]
    chapters = [VideoChapter("First", 0, 10), VideoChapter("Second", 10, 50)]
    targets = prepare_targets(document(segments, chapters, duration=50), [(20, 25)])
    paragraphs = [item for item in targets if item.level == "paragraph"]
    assert [(item.start_seconds, item.end_seconds) for item in paragraphs] == [
        (0, 10),
        (10, 15),
        (25, 28),
        (40, 44),
    ]
    assert paragraphs[0].text == "First topic before chapter boundary"
    assert all("excluded" not in item.text.lower() for item in targets)


def test_sentence_punctuation_and_duration_control_grouping():
    segments = [
        TranscriptSegment(0, 2, 'An English sentence."'),
        TranscriptSegment(2, 4, "中文句子。"),
        TranscriptSegment(4, 12, "Unfinished thought"),
        TranscriptSegment(12, 20, "continues without punctuation"),
        TranscriptSegment(20, 25, "new group"),
    ]
    targets = prepare_targets(document(segments), [])
    sentences = [item for item in targets if item.level == "sentence"]
    assert [item.text for item in sentences] == [
        'An English sentence."',
        "中文句子。",
        "Unfinished thought continues without punctuation",
        "new group",
    ]


def test_paragraphs_respect_duration_without_discarding_long_segments():
    segments = [
        TranscriptSegment(0, 20, "First."),
        TranscriptSegment(20, 40, "Second."),
        TranscriptSegment(40, 46, "Third."),
        TranscriptSegment(46, 110, "A single long timed caption."),
    ]
    paragraphs = [
        item
        for item in prepare_targets(document(segments), [])
        if item.level == "paragraph"
    ]
    assert [item.text for item in paragraphs] == [
        "First. Second.",
        "Third.",
        "A single long timed caption.",
    ]
    assert paragraphs[-1].end_seconds == 110


def test_untimed_invalid_and_excluded_captions_do_not_create_targets():
    segments = [
        TranscriptSegment(0, 0, "Untimed speech"),
        TranscriptSegment(-1, 4, "Negative start"),
        TranscriptSegment(math.nan, 4, "Invalid start"),
        TranscriptSegment(0, math.inf, "Invalid end"),
        TranscriptSegment(4, 5, "  "),
        TranscriptSegment(10, 12, "Beyond duration"),
        TranscriptSegment(2, 3, "Excluded"),
    ]
    assert prepare_targets(document(segments, duration=10), [(2, 3)]) == []


def test_sound_annotations_and_sparse_words_are_not_speech():
    music = [
        TranscriptSegment(2, 4, "[bell]"),
        TranscriptSegment(5, 7, ">> [music] ♪"),
        TranscriptSegment(36, 39, "Heat. Heat."),
    ]
    assert prepare_targets(document(music, duration=77), []) == []
    narrated = [
        TranscriptSegment(0, 4, "[Music] The lens bends light"),
        TranscriptSegment(4, 8, "into a sharp image on the sensor."),
    ]
    sentences = [
        target.text
        for target in prepare_targets(document(narrated, duration=60), [])
        if target.level == "sentence"
    ]
    assert sentences == ["The lens bends light into a sharp image on the sensor."]


@pytest.mark.parametrize(
    "text", ["AI讓我們看見更多細節", "讓我們看見更多細節AI", "AI 讓我們看見更多細節"]
)
def test_mixed_latin_and_cjk_speech_meets_the_minimum_without_spaces(text):
    segments = [TranscriptSegment(0, 60, text)]
    targets = prepare_targets(document(segments, duration=60, language="zh-Hant"), [])
    assert [target.text for target in targets if target.level == "sentence"] == [text]


def test_keyword_extraction_preserves_default_stopwords_and_parent_timing(monkeypatch):
    calls = []

    class Extractor:
        def __init__(self, **options):
            assert options == {"lan": "en", "n": 3, "top": 5}
            self.stopword_set = {"the", "and"}

        def extract_keywords(self, text):
            calls.append(text)
            assert self.stopword_set == {"the", "and", "uh", "um", "hmm"}
            return [("optical mirror", 0.1), ("Optical mirror", 0.2), ("uh", 0.3)]

    monkeypatch.setattr(text_preparation, "KeywordExtractor", Extractor)
    targets = prepare_targets(
        document([TranscriptSegment(7, 14, "The optical mirror reflects light.")]), []
    )
    keywords = [item for item in targets if item.level == "keyword"]
    assert keywords == [TextTarget("keyword", 7, 14, "optical mirror")]
    assert calls == ["The optical mirror reflects light."]


def test_keyword_stopwords_preserve_concepts_and_verbatim_passage_targets():
    segments = [
        TranscriptSegment(0, 5, "Um, the Internet of Things connects sensors."),
        TranscriptSegment(5, 10, "Uh, these things do not require a central server."),
        TranscriptSegment(10, 15, "Hmm, the Internet of Things connects devices."),
    ]
    targets = prepare_targets(document(segments), [])
    keywords = [target.text for target in targets if target.level == "keyword"]
    assert "Internet of Things" in keywords
    assert not {"uh", "um", "hmm"}.intersection(word.casefold() for word in keywords)
    expected = " ".join(segment.text for segment in segments)
    for level in ("sentence", "paragraph"):
        assert (
            " ".join(target.text for target in targets if target.level == level)
            == expected
        )


def test_targets_use_primary_language_tag_without_english_fillers(monkeypatch):
    class Extractor:
        def __init__(self, **options):
            assert options["lan"] == "zh"
            self.stopword_set = {"的"}

        def extract_keywords(self, _text):
            assert self.stopword_set == {"的"}
            return [("光學", 0.1)]

    monkeypatch.setattr(text_preparation, "KeywordExtractor", Extractor)
    targets = prepare_targets(
        document([TranscriptSegment(0, 5, "光學的原理。")], language="zh-Hant"), []
    )
    assert targets[0].text == "光學"
