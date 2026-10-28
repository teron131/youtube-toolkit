"""Vision text targets preserve caption timing and every caption word at each matching scale."""

import math

from youtube_toolkit.schemas import (
    TranscriptChunk,
    TranscriptDocument,
    VideoChapter,
    VideoMetadata,
)
from youtube_toolkit.vision import text as text_preparation
from youtube_toolkit.vision.text import TextTarget, prepare_targets


def document(chunks, chapters=(), *, duration=None, language="en-orig"):
    return TranscriptDocument(
        VideoMetadata(
            "video", "url", duration_seconds=duration, chapters=list(chapters)
        ),
        language,
        "captions",
        chunks,
    )


def test_targets_preserve_all_caption_words_once_per_passage_scale():
    chunks = [
        TranscriptChunk(12, 18, "Then examine the lens."),
        TranscriptChunk(0, 6, "First inspect"),
        TranscriptChunk(5, 12, "the light source."),
        TranscriptChunk(18, 24, "Finally compare the images."),
    ]
    targets = prepare_targets(document(chunks, duration=22), [])
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
    chunks = [
        TranscriptChunk(0, 8, "First topic"),
        TranscriptChunk(8, 14, "before chapter boundary"),
        TranscriptChunk(10, 15, "Second topic"),
        TranscriptChunk(20, 23, "Excluded advertisement"),
        TranscriptChunk(23, 26, "Still excluded"),
        TranscriptChunk(25, 28, "Return to discussion"),
        TranscriptChunk(40, 44, "After a long pause"),
    ]
    chapters = [VideoChapter("First", 0, 10), VideoChapter("Second", 10, 50)]
    targets = prepare_targets(document(chunks, chapters, duration=50), [(20, 25)])
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
    chunks = [
        TranscriptChunk(0, 2, 'An English sentence."'),
        TranscriptChunk(2, 4, "中文句子。"),
        TranscriptChunk(4, 12, "Unfinished thought"),
        TranscriptChunk(12, 20, "continues without punctuation"),
        TranscriptChunk(20, 25, "new group"),
    ]
    targets = prepare_targets(document(chunks), [])
    sentences = [item for item in targets if item.level == "sentence"]
    assert [item.text for item in sentences] == [
        'An English sentence."',
        "中文句子。",
        "Unfinished thought continues without punctuation",
        "new group",
    ]


def test_paragraphs_respect_duration_without_discarding_long_chunks():
    chunks = [
        TranscriptChunk(0, 20, "First."),
        TranscriptChunk(20, 40, "Second."),
        TranscriptChunk(40, 46, "Third."),
        TranscriptChunk(46, 110, "A single long timed caption."),
    ]
    paragraphs = [
        item
        for item in prepare_targets(document(chunks), [])
        if item.level == "paragraph"
    ]
    assert [item.text for item in paragraphs] == [
        "First. Second.",
        "Third.",
        "A single long timed caption.",
    ]
    assert paragraphs[-1].end_seconds == 110


def test_untimed_invalid_and_excluded_captions_do_not_create_targets():
    chunks = [
        TranscriptChunk(0, 0, "Untimed speech"),
        TranscriptChunk(-1, 4, "Negative start"),
        TranscriptChunk(math.nan, 4, "Invalid start"),
        TranscriptChunk(0, math.inf, "Invalid end"),
        TranscriptChunk(4, 5, "  "),
        TranscriptChunk(10, 12, "Beyond duration"),
        TranscriptChunk(2, 3, "Excluded"),
    ]
    assert prepare_targets(document(chunks, duration=10), [(2, 3)]) == []


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
        document([TranscriptChunk(7, 14, "The optical mirror reflects light.")]), []
    )
    keywords = [item for item in targets if item.level == "keyword"]
    assert keywords == [TextTarget("keyword", 7, 14, "optical mirror")]
    assert calls == ["The optical mirror reflects light."]


def test_keyword_stopwords_preserve_concepts_and_verbatim_passage_targets():
    chunks = [
        TranscriptChunk(0, 5, "Um, the Internet of Things connects sensors."),
        TranscriptChunk(5, 10, "Uh, these things do not require a central server."),
        TranscriptChunk(10, 15, "Hmm, the Internet of Things connects devices."),
    ]
    targets = prepare_targets(document(chunks), [])
    keywords = [target.text for target in targets if target.level == "keyword"]
    assert "Internet of Things" in keywords
    assert not {"uh", "um", "hmm"}.intersection(word.casefold() for word in keywords)
    expected = " ".join(chunk.text for chunk in chunks)
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
        document([TranscriptChunk(0, 5, "光學的原理。")], language="zh-Hant"), []
    )
    assert targets[0].text == "光學"
