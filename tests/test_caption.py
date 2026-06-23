import pytest

from youtube_toolkit.schemas import TranscriptChunk
from youtube_toolkit.storage import get_connection, upsert_video
from youtube_toolkit.ytdlp.caption import DEFAULT_TRANSCRIPT_LANGUAGES, TranscriptOptions, fetch_transcript, remove_sponsorblock_chunks, select_caption_track


def test_remove_sponsorblock_chunks_drops_overlapping_chunks() -> None:
    chunks = [
        TranscriptChunk(start_seconds=0, end_seconds=5, text="main"),
        TranscriptChunk(start_seconds=5, end_seconds=10, text="sponsor"),
        TranscriptChunk(start_seconds=10, end_seconds=15, text="more"),
    ]

    assert remove_sponsorblock_chunks(chunks, [(5, 10)]) == [
        TranscriptChunk(start_seconds=0, end_seconds=5, text="main"),
        TranscriptChunk(start_seconds=10, end_seconds=15, text="more"),
    ]


def test_fetch_transcript_returns_stored_transcript_without_client(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Stored title",
                "upload_date": "20260621",
                "view_count": 123,
            },
            transcript="Stored transcript.",
        )

    class FailingClient:
        def metadata(self, video):
            raise AssertionError("metadata should not be fetched")

    document = fetch_transcript(
        "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        client=FailingClient(),
    )

    assert document.metadata.video_id == "JpSHyEIZ_bo"
    assert document.metadata.title == "Stored title"
    assert document.language == "stored"
    assert document.source == "database"
    assert document.text == "Stored transcript."
    assert document.sections[0].text == "Stored transcript."


def test_fetch_transcript_does_not_use_stored_transcript_for_custom_options(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Stored title",
            },
            transcript="Stored transcript.",
        )

    class FailingClient:
        def metadata(self, video):
            raise AssertionError("metadata fetched")

    with pytest.raises(AssertionError, match="metadata fetched"):
        fetch_transcript(
            "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            options=TranscriptOptions(languages=("zh",)),
            client=FailingClient(),
        )


def test_any_caption_source_falls_back_to_manual_subtitles() -> None:
    """Default caption selection can use manual subtitles when automatic captions are absent."""
    metadata = {
        "automatic_captions": {},
        "subtitles": {
            "en": [
                {
                    "ext": "json3",
                    "url": "https://www.youtube.com/api/timedtext?v=example",
                }
            ]
        },
    }

    assert select_caption_track(metadata, ["en"], "any") == (
        "https://www.youtube.com/api/timedtext?v=example",
        "en",
        "subtitles",
    )


def test_default_languages_include_chinese_caption_fallbacks() -> None:
    """Default transcript language selection includes common Chinese caption codes."""
    metadata = {
        "automatic_captions": {},
        "subtitles": {
            "zh-Hant": [
                {
                    "ext": "json3",
                    "url": "https://www.youtube.com/api/timedtext?v=example&lang=zh-Hant",
                }
            ]
        },
    }

    assert select_caption_track(metadata, list(DEFAULT_TRANSCRIPT_LANGUAGES), "any") == (
        "https://www.youtube.com/api/timedtext?v=example&lang=zh-Hant",
        "zh-Hant",
        "subtitles",
    )
