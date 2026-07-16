"""Caption selection, transcript cache, and STT fallback tests."""

from pathlib import Path
import urllib.error

import pytest

from youtube_toolkit.schemas import TranscriptChunk
from youtube_toolkit.storage import get_connection, get_stored_video, upsert_video
from youtube_toolkit.stt import SpeechTranscription
from youtube_toolkit.youtube.caption import (
    DEFAULT_TRANSCRIPT_LANGUAGES,
    TranscriptOptions,
    fetch_caption_payload,
    fetch_live_transcript,
    fetch_transcript,
    remove_sponsorblock_chunks,
    select_caption_track,
)


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
            },
            transcript="Stored transcript.",
        )

    class FailingClient:
        def metadata(self, video_ref: str) -> dict:
            raise AssertionError("metadata should not be fetched")

    document = fetch_transcript(
        "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        client=FailingClient(),
        auto_index=False,
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
        def metadata(self, video_ref: str) -> dict:
            raise AssertionError("metadata fetched")

    with pytest.raises(AssertionError, match="metadata fetched"):
        fetch_transcript(
            "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            options=TranscriptOptions(languages=("zh",)),
            client=FailingClient(),
            auto_index=False,
        )


def test_fetch_transcript_falls_back_to_stt_when_captions_are_missing(
    monkeypatch,
    tmp_path,
) -> None:
    """Normal transcript acquisition should use STT when captions are unavailable."""
    audio_path = tmp_path / "JpSHyEIZ_bo.m4a"
    audio_path.write_bytes(b"audio")

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            assert video_ref == "JpSHyEIZ_bo"
            return {
                "id": "JpSHyEIZ_bo",
                "title": "No captions",
                "automatic_captions": {},
                "subtitles": {},
            }

        def download_transcription_audio(self, video_id: str) -> Path:
            assert video_id == "JpSHyEIZ_bo"
            return audio_path

    def fake_transcribe_audio_details(path: Path) -> SpeechTranscription:
        assert path == audio_path
        return SpeechTranscription(
            text="数据库",
            language="zh",
        )

    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("youtube_toolkit.youtube.caption.transcribe_audio_details", fake_transcribe_audio_details)

    document = fetch_transcript(
        "JpSHyEIZ_bo",
        client=FakeClient(),
        auto_index=False,
    )

    assert document.metadata.video_id == "JpSHyEIZ_bo"
    assert document.metadata.title == "No captions"
    assert document.source == "speech-to-text"
    assert document.language == "speech-to-text"
    assert document.text == "數據庫"
    assert not audio_path.exists()

    with get_connection() as connection:
        stored_video = get_stored_video(connection, "JpSHyEIZ_bo")
    assert stored_video is not None
    assert stored_video["title"] == "No captions"
    assert stored_video["transcript"] == "數據庫"


def test_fetch_live_transcript_does_not_open_storage(monkeypatch) -> None:
    """Stateless caption retrieval should complete even when every storage access would fail."""

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            return {
                "id": video_ref,
                "title": "Frontier model discussion",
                "automatic_captions": {
                    "en": [
                        {
                            "ext": "json3",
                            "url": "https://www.youtube.com/api/timedtext?v=example",
                        }
                    ]
                },
            }

    def deny_connection():
        raise AssertionError("storage should not be opened")

    monkeypatch.setattr("youtube_toolkit.youtube.caption.get_connection", deny_connection)
    monkeypatch.setattr(
        "youtube_toolkit.youtube.caption.fetch_caption_payload",
        lambda _url: {
            "events": [
                {
                    "tStartMs": 0,
                    "dDurationMs": 1000,
                    "segs": [{"utf8": "GPT-5.6 model analysis"}],
                }
            ]
        },
    )

    document = fetch_live_transcript("JpSHyEIZ_bo", client=FakeClient())

    assert document.metadata.video_id == "JpSHyEIZ_bo"
    assert document.text == "GPT-5.6 model analysis"


def test_stt_fallback_removes_downloaded_audio_after_failure(
    monkeypatch,
    tmp_path,
) -> None:
    """A failed stateless STT fallback must not leave downloaded audio behind."""
    audio_path = tmp_path / "JpSHyEIZ_bo.m4a"
    audio_path.write_bytes(b"audio")

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            return {
                "id": video_ref,
                "automatic_captions": {},
                "subtitles": {},
            }

        def download_transcription_audio(self, _video_id: str) -> Path:
            return audio_path

    def fail_transcription(_path: Path) -> SpeechTranscription:
        raise RuntimeError("STT unavailable")

    monkeypatch.setattr(
        "youtube_toolkit.youtube.caption.transcribe_audio_details",
        fail_transcription,
    )

    with pytest.raises(RuntimeError, match="STT unavailable"):
        fetch_live_transcript("JpSHyEIZ_bo", client=FakeClient())

    assert not audio_path.exists()


def test_fetch_caption_payload_retries_transient_network_failures(monkeypatch) -> None:
    """Caption downloads should retry transient failures and preserve bounded backoff."""
    attempts = 0
    delays: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"events": []}'

    def fake_urlopen(_request, *, timeout: int):
        nonlocal attempts
        assert timeout == 30
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()

    monkeypatch.setattr("youtube_toolkit.youtube.caption.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("youtube_toolkit.youtube.caption.time.sleep", delays.append)

    assert fetch_caption_payload("https://www.youtube.com/api/timedtext?v=example") == {"events": []}
    assert attempts == 3
    assert delays == [0.25, 0.5]


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


def test_default_languages_prefer_hong_kong_and_cantonese_before_simplified() -> None:
    """Default transcript language selection prefers Traditional/HK keys before Simplified Chinese."""
    metadata = {
        "automatic_captions": {},
        "subtitles": {
            "zh-Hans": [
                {
                    "ext": "json3",
                    "url": "https://www.youtube.com/api/timedtext?v=example&lang=zh-Hans",
                }
            ],
            "yue-HK": [
                {
                    "ext": "json3",
                    "url": "https://www.youtube.com/api/timedtext?v=example&lang=yue-HK",
                }
            ],
        },
    }

    assert select_caption_track(metadata, list(DEFAULT_TRANSCRIPT_LANGUAGES), "any") == (
        "https://www.youtube.com/api/timedtext?v=example&lang=yue-HK",
        "yue-HK",
        "subtitles",
    )


def test_default_languages_prefer_manual_english_before_automatic_chinese() -> None:
    """Default transcript language selection prefers English even across caption sources."""
    metadata = {
        "automatic_captions": {
            "zh-Hant": [
                {
                    "ext": "json3",
                    "url": "https://www.youtube.com/api/timedtext?v=example&lang=zh-Hant",
                }
            ]
        },
        "subtitles": {
            "en": [
                {
                    "ext": "json3",
                    "url": "https://www.youtube.com/api/timedtext?v=example&lang=en",
                }
            ]
        },
    }

    assert select_caption_track(metadata, list(DEFAULT_TRANSCRIPT_LANGUAGES), "any") == (
        "https://www.youtube.com/api/timedtext?v=example&lang=en",
        "en",
        "subtitles",
    )
