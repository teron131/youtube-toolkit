"""Caption selection, transcript archives, and STT fallback tests."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from youtube_toolkit.artifacts import write_artifact
from youtube_toolkit.schemas import (
    SpeechTranscription,
    TranscriptChunk,
    TranscriptDocument,
    VideoMetadata,
)
from youtube_toolkit.youtube.caption import (
    CaptionTranscriptUnavailable,
    fetch_caption_payload,
    remove_sponsorblock_chunks,
    select_caption_track,
)
from youtube_toolkit.transcript import (
    DEFAULT_TRANSCRIPT_LANGUAGES,
    TranscriptOptions,
    fetch_transcript,
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

    def fake_transcribe_in_cloud(path: Path) -> SpeechTranscription:
        assert path == audio_path
        return SpeechTranscription(
            text="数据库",
            language="zh",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        fake_transcribe_in_cloud,
    )

    document = fetch_transcript(
        "JpSHyEIZ_bo",
        client=FakeClient(),
    )

    assert document.metadata.video_id == "JpSHyEIZ_bo"
    assert document.metadata.title == "No captions"
    assert document.source == "speech-to-text"
    assert document.language == "zh"
    assert document.text == "數據庫"
    assert not audio_path.exists()


def test_fetch_transcript_fetches_and_archives_captions(monkeypatch, tmp_path) -> None:
    """Public transcript retrieval saves captions in the current directory archive."""

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

        def download_transcription_audio(self, _video_id: str) -> Path:
            pytest.fail("caption retrieval should not download STT audio")

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
    monkeypatch.chdir(tmp_path)

    document = fetch_transcript("JpSHyEIZ_bo", client=FakeClient())

    assert document.metadata.video_id == "JpSHyEIZ_bo"
    assert document.text == "GPT-5.6 model analysis"


@pytest.mark.parametrize(
    ("language", "caption", "expected"),
    [("zh-Hans", "数据库里", "數據庫裡"), ("en", "Database", "Database")],
)
def test_caption_text_is_converted_to_taiwan_traditional(
    monkeypatch, tmp_path, language, caption, expected
) -> None:
    """Captions follow the same script policy as speech-to-text: Taiwan Traditional characters, original wording."""

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            return {
                "id": video_ref,
                "subtitles": {language: [caption_track(language)]},
            }

        def download_transcription_audio(self, _video_id: str) -> Path:
            pytest.fail("caption retrieval should not download STT audio")

    monkeypatch.setattr(
        "youtube_toolkit.youtube.caption.fetch_caption_payload",
        lambda _url: {
            "events": [
                {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": caption}]}
            ]
        },
    )
    monkeypatch.chdir(tmp_path)

    document = fetch_transcript(
        "JpSHyEIZ_bo",
        options=TranscriptOptions(languages=(language,)),
        client=FakeClient(),
    )

    assert document.language == language
    assert document.text == expected


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

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        fail_transcription,
    )

    with pytest.raises(RuntimeError, match="STT unavailable"):
        fetch_transcript("JpSHyEIZ_bo", client=FakeClient())

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

    monkeypatch.setattr(
        "youtube_toolkit.youtube.caption.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)

    assert fetch_caption_payload("https://www.youtube.com/api/timedtext?v=example") == {
        "events": []
    }
    assert attempts == 3
    assert delays == [1.0, 2.0]


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

    assert select_caption_track(
        metadata, list(DEFAULT_TRANSCRIPT_LANGUAGES), "any"
    ) == (
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

    assert select_caption_track(
        metadata, list(DEFAULT_TRANSCRIPT_LANGUAGES), "any"
    ) == (
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

    assert select_caption_track(
        metadata, list(DEFAULT_TRANSCRIPT_LANGUAGES), "any"
    ) == (
        "https://www.youtube.com/api/timedtext?v=example&lang=en",
        "en",
        "subtitles",
    )


def caption_track(lang: str, translated_to: str | None = None) -> dict:
    url = f"https://www.youtube.com/api/timedtext?v=example&lang={lang}&fmt=json3"
    return {
        "ext": "json3",
        "url": url + (f"&tlang={translated_to}" if translated_to else ""),
    }


@pytest.mark.parametrize(
    ("language", "expected_key", "expected_lang"),
    [("ja", "ja-orig", "ja"), ("en-US", "en-US", "en-US")],
)
def test_default_languages_prefer_original_captions_over_machine_translations(
    language, expected_key, expected_lang
) -> None:
    """Without English or Chinese originals, the spoken language wins over YouTube's machine translations, even on dubbed videos."""
    metadata = {
        "language": language,
        "automatic_captions": {
            "en": [caption_track(expected_lang, translated_to="en")],
            "zh-Hant": [caption_track(expected_lang, translated_to="zh-Hant")],
            "ja-orig": [caption_track("ja")],
            "ja": [caption_track("ja")],
            "en-US": [caption_track("en-US")],
        },
        "subtitles": {},
    }
    assert select_caption_track(metadata, DEFAULT_TRANSCRIPT_LANGUAGES, "any") == (
        caption_track(expected_lang)["url"],
        expected_key,
        "automatic_captions",
    )


@pytest.mark.parametrize(
    ("language", "manual", "expected"),
    [
        ("en", ["en-US", "zh-Hant"], ("en-US", "subtitles")),
        (None, ["en", "zh-Hant"], ("zh-Hant", "subtitles")),
        ("ja", ["en", "zh-Hant"], ("en", "subtitles")),
        ("ja", ["fr"], ("ja-orig", "automatic_captions")),
    ],
)
def test_manual_subtitles_follow_spoken_language_before_automatic_captions(
    language, manual, expected
) -> None:
    """Manual English or Chinese comes first, led by the spoken language; a video without automatic captions counts as Chinese."""
    spoken = language or "zh"
    metadata = {
        "language": language,
        "automatic_captions": (
            {f"{spoken}-orig": [caption_track(spoken)]} if language else {}
        ),
        "subtitles": {key: [caption_track(key)] for key in manual},
    }

    assert select_caption_track(metadata, DEFAULT_TRANSCRIPT_LANGUAGES, "any") == (
        caption_track(expected[0].removesuffix("-orig"))["url"],
        *expected,
    )


def test_metadata_drops_machine_translated_captions(monkeypatch, tmp_path) -> None:
    """Translations are removed before metadata is returned or archived, keeping the original tracks."""
    import json

    from youtube_toolkit.youtube.client import YtDlpClient

    monkeypatch.chdir(tmp_path)
    response = {
        "id": "JpSHyEIZ_bo",
        "automatic_captions": {
            "en-orig": [caption_track("en")],
            "en": [caption_track("en"), caption_track("ja", translated_to="en")],
            "fr": [caption_track("en", translated_to="fr")],
        },
    }
    monkeypatch.setattr(
        "youtube_toolkit.youtube.client.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(response)
        ),
    )

    metadata = YtDlpClient().metadata("JpSHyEIZ_bo")

    originals = {"en-orig": [caption_track("en")], "en": [caption_track("en")]}
    assert metadata["automatic_captions"] == originals
    archived = json.loads(
        next((tmp_path / ".cache/youtube-toolkit/metadata").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert archived["payload"]["automatic_captions"] == originals


@pytest.mark.parametrize(
    "options",
    [
        TranscriptOptions(languages=("zh",)),
        TranscriptOptions(source="manual"),
    ],
)
def test_transcript_archives_preserve_timing_and_retrieval_options(
    monkeypatch, tmp_path, options
) -> None:
    """Equivalent video URLs fetch fresh documents and preserve each retrieval policy."""
    from youtube_toolkit.schemas import TranscriptDocument, VideoChapter, VideoMetadata

    monkeypatch.chdir(tmp_path)
    calls = []
    expected = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo",
            url="https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            title="Example",
            description="A demonstration",
            thumbnail="https://example.com/thumbnail.jpg",
            chapters=[
                VideoChapter(title="Opening", start_seconds=0.0, end_seconds=4.0)
            ],
        ),
        language="en",
        source="manual",
        chunks=[
            TranscriptChunk(start_seconds=1.5, end_seconds=4.0, text="An example.")
        ],
    )

    class FakeClient:
        def metadata(self, video_ref):
            calls.append(video_ref)
            return {"id": "JpSHyEIZ_bo"}

    monkeypatch.setattr("youtube_toolkit.transcript.YtDlpClient", FakeClient)
    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript",
        lambda _metadata, **_policy: expected,
    )
    assert fetch_transcript("JpSHyEIZ_bo") == expected
    assert fetch_transcript("https://youtu.be/JpSHyEIZ_bo") == expected
    assert len(calls) == 2
    assert fetch_transcript("JpSHyEIZ_bo", options=options) == expected
    assert len(calls) == 3
    import json

    paths = list((tmp_path / ".cache/youtube-toolkit/transcripts").glob("*.json"))
    assert len(paths) == 3
    assert all("JpSHyEIZ_bo" in path.name for path in paths)
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert all(entry["payload"] == asdict(expected) for entry in entries)
    assert any(
        entry["request"]["languages"] == list(options.languages)
        and entry["request"]["source"] == options.source
        for entry in entries
    )
    previous = {path: path.read_bytes() for path in paths}
    old_artifact = paths[0].parent / "old-copy.json"
    old_artifact.write_text("broken JSON", encoding="utf-8")
    assert fetch_transcript("JpSHyEIZ_bo") == expected
    assert len(calls) == 4
    assert all(path.read_bytes() == content for path, content in previous.items())
    assert old_artifact.read_text(encoding="utf-8") == "broken JSON"

    other = tmp_path / "other-task"
    other.mkdir()
    monkeypatch.chdir(other)
    assert fetch_transcript("JpSHyEIZ_bo") == expected
    assert len(calls) == 5
    assert len(list((other / ".cache/youtube-toolkit/transcripts").glob("*.json"))) == 1


@pytest.mark.parametrize("archived_stt", [False, True])
def test_caption_only_policy_never_uses_stt(
    monkeypatch, tmp_path, archived_stt
) -> None:
    """Caption-only retrieval ignores saved speech transcripts and never triggers paid audio work."""
    video_id = "JpSHyEIZ_bo"
    monkeypatch.chdir(tmp_path)
    if archived_stt:
        document = TranscriptDocument(
            VideoMetadata(video_id=video_id, url="https://youtu.be/" + video_id),
            "speech-to-text",
            "speech-to-text",
            [TranscriptChunk(0, 0, "Untimed speech")],
        )
        write_artifact(
            "transcripts",
            video_id,
            {
                "video_id": video_id,
                "languages": list(DEFAULT_TRANSCRIPT_LANGUAGES),
                "source": "any",
            },
            asdict(document),
        )

    class FakeClient:
        def metadata(self, _ref):
            return {"id": video_id, "automatic_captions": {}, "subtitles": {}}

        def download_transcription_audio(self, _ref):
            pytest.fail("caption-only policy must not download STT audio")

    with pytest.raises(CaptionTranscriptUnavailable):
        fetch_transcript(
            video_id, options=TranscriptOptions(allow_stt=False), client=FakeClient()
        )


def test_stt_without_any_route_keeps_caption_failure_without_audio_work(
    monkeypatch, tmp_path
) -> None:
    """With no cloud key and no local libraries installed retrieval is captions-only: the caption failure surfaces and no audio is downloaded."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("youtube_toolkit.transcript.local_stt_installed", lambda: False)

    class FakeClient:
        def metadata(self, _ref):
            return {"id": "JpSHyEIZ_bo", "automatic_captions": {}, "subtitles": {}}

        def download_transcription_audio(self, _ref):
            pytest.fail("an unconfigured STT fallback must not download audio")

    with pytest.raises(CaptionTranscriptUnavailable):
        fetch_transcript("JpSHyEIZ_bo", client=FakeClient())


@pytest.mark.parametrize(
    ("cloud_key", "expected_text", "expected_language"),
    [(None, "local text", "de"), ("test-key", "cloud text", "en")],
)
def test_stt_prefers_cloud_and_falls_back_to_the_local_model(
    monkeypatch, tmp_path, cloud_key, expected_text, expected_language
) -> None:
    """The cloud provider wins when its key is set; otherwise the local model transcribes and the video's reported language labels the result."""
    audio_path = tmp_path / "JpSHyEIZ_bo.m4a"
    audio_path.write_bytes(b"audio")
    monkeypatch.chdir(tmp_path)
    if cloud_key is None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("XAI_API_KEY", cloud_key)
    monkeypatch.setattr("youtube_toolkit.transcript.local_stt_installed", lambda: True)
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_locally",
        lambda _path: SpeechTranscription(text="local text"),
    )
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        lambda _path: SpeechTranscription(text="cloud text", language="en"),
    )

    class FakeClient:
        def metadata(self, _ref):
            return {
                "id": "JpSHyEIZ_bo",
                "language": "de",
                "automatic_captions": {},
                "subtitles": {},
            }

        def download_transcription_audio(self, _ref):
            return audio_path

    document = fetch_transcript("JpSHyEIZ_bo", client=FakeClient())

    assert document.source == "speech-to-text"
    assert document.text == expected_text
    assert document.language == expected_language
    assert not audio_path.exists()


def test_supplied_transcript_metadata_must_match_video(monkeypatch, tmp_path) -> None:
    """A caller cannot archive another video's captions under the requested video ID."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="Supplied metadata does not match requested video"
    ):
        fetch_transcript("JpSHyEIZ_bo", metadata={"id": "abcdefghijk"})
    assert not (tmp_path / ".cache/youtube-toolkit/transcripts").exists()
