"""Transcript-guided vision retains real timing and chapter coverage while ranking local topic relevance."""

from itertools import pairwise

import pytest

from youtube_toolkit.schemas import (
    TranscriptChunk,
    TranscriptDocument,
    VideoChapter,
    VideoMetadata,
)
from youtube_toolkit.vision.windows import select_transcript_windows


def document(chunks, chapters=(), *, duration=None):
    return TranscriptDocument(
        metadata=VideoMetadata(
            video_id="test",
            url="test",
            duration_seconds=duration,
            chapters=list(chapters),
        ),
        language="en",
        source="captions",
        chunks=chunks,
    )


def test_chapters_receive_coverage_and_real_caption_ranges():
    chapters = [
        VideoChapter(f"Topic {index}", index * 100, (index + 1) * 100)
        for index in range(3)
    ]
    chunks = [
        TranscriptChunk(index * 100 + 4, index * 100 + 19, f"Topic {index} explanation")
        for index in range(3)
    ]
    windows = select_transcript_windows(document(chunks, chapters), max_windows=3)
    assert [window.chapter for window in windows] == ["Topic 0", "Topic 1", "Topic 2"]
    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (4, 19),
        (104, 119),
        (204, 219),
    ]
    assert all(0 <= window.text_score <= 1 for window in windows)


def test_chapter_local_relevance_prefers_repeated_topic_over_function_word_filler():
    chunks = [
        TranscriptChunk(0, 20, "Well anyway yes okay welcome everyone"),
        TranscriptChunk(
            30,
            50,
            "Protein folding determines protein structure and molecular function",
        ),
        TranscriptChunk(
            60, 80, "Protein structure follows folding molecular interactions"
        ),
    ]
    windows = select_transcript_windows(document(chunks), max_windows=1)
    assert len(windows) == 1
    assert "Protein" in windows[0].text
    assert windows[0].text_score > 0


def test_many_chapters_use_evenly_distributed_slots_and_count_bound():
    chapters = [
        VideoChapter(str(index), index * 60, (index + 1) * 60) for index in range(12)
    ]
    chunks = [
        TranscriptChunk(index * 60, index * 60 + 20, f"Subject {index} explanation")
        for index in range(12)
    ]
    windows = select_transcript_windows(document(chunks, chapters), max_windows=3)
    assert [window.chapter for window in windows] == ["0", "6", "11"]


def test_missing_chapters_preserve_time_coverage():
    chunks = [
        TranscriptChunk(start, start + 20, f"Distinct explanation at {start}")
        for start in range(0, 300, 60)
    ]
    windows = select_transcript_windows(document(chunks), max_windows=3)
    assert len(windows) == 3
    assert windows[0].start_seconds < 100
    assert windows[1].start_seconds >= 100
    assert windows[2].start_seconds >= 2 * 260 / 3
    assert all(window.chapter is None for window in windows)


def test_untimed_stt_does_not_fabricate_windows():
    assert (
        select_transcript_windows(
            document([TranscriptChunk(0, 0, "Long untimed transcript")])
        )
        == []
    )


def test_short_video_returns_available_range_only():
    windows = select_transcript_windows(
        document([TranscriptChunk(2, 8, "A short demonstration")])
    )
    assert len(windows) == 1
    assert windows[0].start_seconds == 2
    assert windows[0].end_seconds == 8


def test_caption_gaps_are_not_joined_and_redundant_passages_do_not_fill_budget():
    chunks = [
        TranscriptChunk(0, 4, "Protein folding molecular structure"),
        TranscriptChunk(15, 19, "Protein folding molecular structure"),
    ]
    windows = select_transcript_windows(document(chunks), max_windows=6)
    assert len(windows) == 1
    assert windows[0].end_seconds - windows[0].start_seconds == 4


def test_sponsor_chapter_does_not_receive_a_window():
    chapters = [
        VideoChapter("Sponsor", 0, 30),
        VideoChapter("Main explanation", 30, 60),
    ]
    chunks = [
        TranscriptChunk(0, 20, "Buy product"),
        TranscriptChunk(30, 50, "Science explanation"),
    ]
    windows = select_transcript_windows(document(chunks, chapters))
    assert len(windows) == 1
    assert windows[0].chapter == "Main explanation"


@pytest.mark.parametrize("limit", [0, -1])
def test_empty_allowance_returns_no_windows(limit):
    assert (
        select_transcript_windows(
            document([TranscriptChunk(0, 10, "Explanation")]), max_windows=limit
        )
        == []
    )


def test_extra_passages_fill_allowance_without_overlapping():
    chunks = [
        TranscriptChunk(0, 20, "Protein folding creates molecular structure"),
        TranscriptChunk(30, 50, "DNA encodes genetic sequences"),
        TranscriptChunk(60, 80, "Cell membranes control chemical transport"),
    ]
    chapters = [VideoChapter("Biology", 0, 90)]
    windows = select_transcript_windows(document(chunks, chapters), max_windows=3)
    assert len(windows) == 3
    assert all(
        left.end_seconds <= right.start_seconds for left, right in pairwise(windows)
    )


def test_single_slot_selects_middle_chapter():
    chapters = [
        VideoChapter(str(index), index * 60, (index + 1) * 60) for index in range(5)
    ]
    chunks = [
        TranscriptChunk(index * 60, index * 60 + 20, f"Subject {index}")
        for index in range(5)
    ]
    windows = select_transcript_windows(document(chunks, chapters), max_windows=1)
    assert [window.chapter for window in windows] == ["2"]


def test_metadata_duration_bounds_caption_windows():
    chunks = [
        TranscriptChunk(5, 15, "Protein structure"),
        TranscriptChunk(30, 70, "DNA sequence"),
        TranscriptChunk(50, 60, "Outside real video"),
        TranscriptChunk(80, 90, "Outside later video"),
    ]
    chapters = [VideoChapter("Biology", 0, 100)]
    windows = select_transcript_windows(document(chunks, chapters, duration=50))
    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (5, 15),
        (30, 50),
    ]
    assert all("Outside" not in window.text for window in windows)


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf")])
def test_unusable_metadata_duration_keeps_real_caption_timing(duration):
    windows = select_transcript_windows(
        document([TranscriptChunk(2, 8, "Explanation")], duration=duration)
    )
    assert windows[0].end_seconds == 8
