"""Caption selection, saved transcripts, and speech-to-text fallback."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from youtube_toolkit.schemas import (
    SpeechTranscription,
    TranscriptDocument,
    TranscriptSegment,
    VideoMetadata,
)
from youtube_toolkit.transcript import (
    SUBTITLE_LANGUAGES,
    fetch_transcript,
)
from youtube_toolkit.youtube.captions import (
    CaptionTranscriptUnavailable,
    fetch_caption_payload,
    remove_sponsorblock_segments,
    select_caption_track,
)


def test_remove_sponsorblock_segments_drops_overlapping_segments() -> None:
    segments = [
        TranscriptSegment(start_seconds=0, end_seconds=5, text="main"),
        TranscriptSegment(start_seconds=5, end_seconds=10, text="sponsor"),
        TranscriptSegment(start_seconds=10, end_seconds=15, text="more"),
    ]

    assert remove_sponsorblock_segments(segments, [(5, 10)]) == [
        TranscriptSegment(start_seconds=0, end_seconds=5, text="main"),
        TranscriptSegment(start_seconds=10, end_seconds=15, text="more"),
    ]


def test_fetch_transcript_falls_back_to_stt_when_captions_are_missing(
    monkeypatch,
    tmp_path,
) -> None:
    downloads: list[Path] = []

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            assert video_ref == "JpSHyEIZ_bo"
            return {
                "id": "JpSHyEIZ_bo",
                "title": "No captions",
                "description": "Comparing Fable with GPT-6, and Fable again: https://youtu.be/VRFfAeW30qE",
                "automatic_captions": {},
                "subtitles": {},
            }

        def download_transcription_audio(self, metadata, directory: Path) -> Path:
            assert metadata["id"] == "JpSHyEIZ_bo"
            return save_audio(directory, downloads)

    def fake_transcribe_in_cloud(
        path: Path, vocabulary: list[str]
    ) -> SpeechTranscription:
        assert path == downloads[0]
        assert vocabulary == ["Fable", "GPT-6"]
        return SpeechTranscription(
            text="数据库", segments=[], model="cloud-model", language="zh"
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        fake_transcribe_in_cloud,
    )

    document = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False).value

    assert document.metadata.video_id == "JpSHyEIZ_bo"
    assert document.metadata.title == "No captions"
    assert document.source == "speech-to-text"
    assert document.language == "zh"
    assert document.text == "數據庫"
    assert document.segments == [TranscriptSegment(0, 0, "數據庫")]
    assert not downloads[0].parent.exists()


def test_timed_speech_to_text_segments_become_document_segments(
    monkeypatch, tmp_path
) -> None:
    segments = [
        TranscriptSegment(0.2, 1.0, "Hello world."),
        TranscriptSegment(2.5, 3.2, "Next line"),
    ]

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            return {"id": video_ref, "automatic_captions": {}, "subtitles": {}}

        def download_transcription_audio(self, _metadata, directory: Path) -> Path:
            return save_audio(directory, [])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        lambda _path, _vocabulary: SpeechTranscription(
            text="Hello world. Next line",
            segments=segments,
            model="cloud-model",
            language="en",
        ),
    )

    document = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False).value

    assert document.source == "speech-to-text"
    assert document.segments == segments
    assert document.text == "Hello world. Next line"


def test_fetch_transcript_saves_captions_in_the_video_folder(
    monkeypatch, tmp_path
) -> None:
    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            return {
                "id": video_ref,
                "title": "Frontier model [4K] discussion (Q&A)?",
                "automatic_captions": {
                    "en": [
                        {
                            "ext": "json3",
                            "url": "https://www.youtube.com/api/timedtext?v=example",
                        }
                    ]
                },
            }

        def download_transcription_audio(self, _metadata, _directory) -> Path:
            pytest.fail("caption retrieval should not download STT audio")

    monkeypatch.setattr(
        "youtube_toolkit.youtube.captions.fetch_caption_payload",
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

    transcript = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False)

    assert transcript.value.metadata.video_id == "JpSHyEIZ_bo"
    assert transcript.value.text == "GPT-5.6 model analysis"
    assert not transcript.reused
    assert transcript.path.parent == (
        tmp_path
        / ".youtube-toolkit/videos/Frontier_model_4K_discussion_Q_A_JpSHyEIZ_bo/transcripts"
    )
    assert transcript.path.name.startswith("original_fallback-")
    assert (tmp_path / ".youtube-toolkit/.gitignore").read_text() == "*\n"


@pytest.mark.parametrize(
    ("language", "captions", "expected"),
    [
        ("zh-Hans", ["数据库里"], "數據庫裡"),
        ("en", ["Database"], "Database"),
        # Japanese shares kanji with Simplified Chinese, and its lines without kana are still Japanese.
        ("ja", ["動画編集", "の争いです"], "動画編集 の争いです"),
    ],
)
def test_chinese_caption_text_is_converted_to_taiwan_traditional(
    monkeypatch, tmp_path, language, captions, expected
) -> None:
    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            # Japanese is taken from automatic captions, as only English and Chinese subtitles are preferred.
            tracks = {language: [caption_track(language)]}
            if language == "ja":
                return {"id": video_ref, "language": "ja", "automatic_captions": tracks}
            return {"id": video_ref, "subtitles": tracks}

        def download_transcription_audio(self, _metadata, _directory) -> Path:
            pytest.fail("caption retrieval should not download STT audio")

    monkeypatch.setattr(
        "youtube_toolkit.youtube.captions.fetch_caption_payload",
        lambda _url: {
            "events": [
                {
                    "tStartMs": index * 1000,
                    "dDurationMs": 1000,
                    "segs": [{"utf8": text}],
                }
                for index, text in enumerate(captions)
            ]
        },
    )
    monkeypatch.chdir(tmp_path)

    document = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False).value

    assert document.language == language
    assert document.text == expected


def test_stt_fallback_removes_downloaded_audio_after_failure(
    monkeypatch,
    tmp_path,
) -> None:
    downloads: list[Path] = []

    class FakeClient:
        def metadata(self, video_ref: str) -> dict:
            return {
                "id": video_ref,
                "automatic_captions": {},
                "subtitles": {},
            }

        def download_transcription_audio(self, _metadata, directory: Path) -> Path:
            return save_audio(directory, downloads)

    def fail_transcription(_path: Path, _vocabulary: list[str]) -> SpeechTranscription:
        raise RuntimeError("STT unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        fail_transcription,
    )

    with pytest.raises(RuntimeError, match="STT unavailable"):
        fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False)

    assert len(downloads) == 1
    assert not downloads[0].parent.exists()
    assert list((tmp_path / ".youtube-toolkit/tmp").iterdir()) == []


def test_fetch_caption_payload_retries_transient_network_failures(monkeypatch) -> None:
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
        "youtube_toolkit.youtube.captions.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)

    assert fetch_caption_payload("https://www.youtube.com/api/timedtext?v=example") == {
        "events": []
    }
    assert attempts == 3
    assert delays == [1.0, 2.0]


def test_any_caption_source_falls_back_to_manual_subtitles() -> None:
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

    assert select_caption_track(metadata, ["en"]) == (
        "https://www.youtube.com/api/timedtext?v=example",
        "en",
        "subtitles",
    )


def test_language_without_json3_track_counts_as_unavailable() -> None:
    metadata = {
        "automatic_captions": {},
        "subtitles": {
            "en": [
                {
                    "ext": "vtt",
                    "url": "https://www.youtube.com/api/timedtext?v=example&fmt=vtt",
                }
            ]
        },
    }

    with pytest.raises(CaptionTranscriptUnavailable):
        select_caption_track(metadata, ["en"])


def test_default_languages_include_chinese_caption_fallbacks() -> None:
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

    assert select_caption_track(metadata, list(SUBTITLE_LANGUAGES)) == (
        "https://www.youtube.com/api/timedtext?v=example&lang=zh-Hant",
        "zh-Hant",
        "subtitles",
    )


def test_default_languages_prefer_hong_kong_and_cantonese_before_simplified() -> None:
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

    assert select_caption_track(metadata, list(SUBTITLE_LANGUAGES)) == (
        "https://www.youtube.com/api/timedtext?v=example&lang=yue-HK",
        "yue-HK",
        "subtitles",
    )


def test_default_languages_prefer_manual_english_before_automatic_chinese() -> None:
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

    assert select_caption_track(metadata, list(SUBTITLE_LANGUAGES)) == (
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
    assert select_caption_track(metadata, SUBTITLE_LANGUAGES) == (
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
    spoken = language or "zh"
    metadata = {
        "language": language,
        "automatic_captions": (
            {f"{spoken}-orig": [caption_track(spoken)]} if language else {}
        ),
        "subtitles": {key: [caption_track(key)] for key in manual},
    }

    assert select_caption_track(metadata, SUBTITLE_LANGUAGES) == (
        caption_track(expected[0].removesuffix("-orig"))["url"],
        *expected,
    )


def test_metadata_drops_machine_translated_captions(monkeypatch, tmp_path) -> None:
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
    recorded = json.loads(
        next((tmp_path / ".youtube-toolkit/records/metadata").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert recorded["payload"]["automatic_captions"] == originals


def test_saved_transcripts_are_reused_per_policy_until_fresh(
    monkeypatch, tmp_path
) -> None:
    import json

    from youtube_toolkit.schemas import TranscriptDocument, VideoChapter, VideoMetadata

    monkeypatch.chdir(tmp_path)
    metadata_calls = []
    caption_calls = []
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
        source="subtitles",
        segments=[
            TranscriptSegment(start_seconds=1.5, end_seconds=4.0, text="An example.")
        ],
    )

    class FakeClient:
        def metadata(self, video_ref):
            metadata_calls.append(video_ref)
            return {"id": "JpSHyEIZ_bo", "title": "Example"}

    def fetch_captions(metadata, **_policy):
        caption_calls.append(metadata["id"])
        return expected

    monkeypatch.setattr("youtube_toolkit.transcript.YtDlpClient", FakeClient)
    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript", fetch_captions
    )

    first = fetch_transcript("JpSHyEIZ_bo", fresh=False)
    assert (first.value, first.reused) == (expected, False)
    again = fetch_transcript("https://youtu.be/JpSHyEIZ_bo", fresh=False)
    assert (again.value, again.reused, again.path) == (expected, True, first.path)
    assert len(metadata_calls) == 2
    assert len(caption_calls) == 1
    saved = json.loads(first.path.read_text(encoding="utf-8"))
    assert saved["data"] == asdict(expected)
    assert saved["key"]["stt"] == "fallback"

    caption_only = fetch_transcript("JpSHyEIZ_bo", stt="never", fresh=False)
    assert (caption_only.value, caption_only.reused) == (expected, False)
    assert caption_only.path != first.path
    assert caption_only.path.name.startswith("original_never-")
    assert len(caption_calls) == 2

    refetched = fetch_transcript("JpSHyEIZ_bo", fresh=True)
    assert (refetched.value, refetched.reused, refetched.path) == (
        expected,
        False,
        first.path,
    )
    assert len(caption_calls) == 3

    other = tmp_path / "other-task"
    other.mkdir()
    monkeypatch.chdir(other)
    elsewhere = fetch_transcript("JpSHyEIZ_bo", fresh=False)
    assert (elsewhere.value, elsewhere.reused) == (expected, False)
    assert elsewhere.path.is_relative_to(other / ".youtube-toolkit/videos")
    assert len(caption_calls) == 4


def test_a_renamed_video_keeps_its_folder(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    titles = iter(["First title", "Second title"])

    class FakeClient:
        def metadata(self, _ref):
            return {"id": "JpSHyEIZ_bo", "title": next(titles)}

        def download_transcription_audio(self, _metadata, _directory):
            raise AssertionError("captions exist, so no audio is downloaded")

    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript",
        lambda metadata, **_policy: TranscriptDocument(
            VideoMetadata.from_yt_dlp(metadata), "en", "subtitles", []
        ),
    )

    first = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False)
    second = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False)

    assert first.path.parent.parent.name == "First_title_JpSHyEIZ_bo"
    assert (second.path, second.reused) == (first.path, True)


@pytest.mark.parametrize("saved_stt", [False, True])
def test_caption_only_policy_never_uses_stt(monkeypatch, tmp_path, saved_stt) -> None:
    video_id = "JpSHyEIZ_bo"
    monkeypatch.chdir(tmp_path)

    class FakeClient:
        def metadata(self, _ref):
            return {"id": video_id, "automatic_captions": {}, "subtitles": {}}

        def download_transcription_audio(self, _metadata, directory):
            if not saved_stt:
                pytest.fail("caption-only policy must not download STT audio")
            return save_audio(directory, [])

    if saved_stt:
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr(
            "youtube_toolkit.transcript.transcribe_in_cloud",
            lambda _path, _vocabulary: SpeechTranscription(
                text="Untimed speech", segments=[], model="cloud-model"
            ),
        )
        fetch_transcript(video_id, client=FakeClient(), fresh=False)

    with pytest.raises(CaptionTranscriptUnavailable):
        fetch_transcript(video_id, stt="never", client=FakeClient(), fresh=False)


def test_stt_without_any_route_keeps_caption_failure_without_audio_work(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("youtube_toolkit.transcript.local_stt_installed", lambda: False)

    class FakeClient:
        def metadata(self, _ref):
            return {"id": "JpSHyEIZ_bo", "automatic_captions": {}, "subtitles": {}}

        def download_transcription_audio(self, _metadata, _directory):
            pytest.fail("an unconfigured STT fallback must not download audio")

    with pytest.raises(CaptionTranscriptUnavailable):
        fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False)
    with pytest.raises(RuntimeError, match="Speech-to-text is not set up"):
        fetch_transcript("JpSHyEIZ_bo", stt="always", client=FakeClient(), fresh=False)


def test_always_policy_transcribes_even_when_captions_exist(
    monkeypatch, tmp_path
) -> None:
    downloads: list[Path] = []

    class FakeClient:
        def metadata(self, _ref):
            return {"id": "JpSHyEIZ_bo", "subtitles": {"en": [{"ext": "json3"}]}}

        def download_transcription_audio(self, _metadata, directory):
            return save_audio(directory, downloads)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript",
        lambda *_args, **_kwargs: pytest.fail("always must skip captions"),
    )
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        lambda _path, _vocabulary: SpeechTranscription(
            text="spoken", segments=[], model="cloud-model", language="en"
        ),
    )

    document = fetch_transcript(
        "JpSHyEIZ_bo", stt="always", client=FakeClient(), fresh=False
    ).value

    assert document.source == "speech-to-text"
    assert document.text == "spoken"
    assert not downloads[0].parent.exists()


@pytest.mark.parametrize(
    ("cloud_key", "expected_text", "expected_language", "expected_model"),
    [
        (None, "local text", "de", "local-model"),
        ("test-key", "cloud text", "en", "cloud-model"),
    ],
)
def test_stt_prefers_cloud_and_falls_back_to_the_local_model(
    monkeypatch, tmp_path, cloud_key, expected_text, expected_language, expected_model
) -> None:
    downloads: list[Path] = []
    monkeypatch.chdir(tmp_path)
    if cloud_key is None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("XAI_API_KEY", cloud_key)
    monkeypatch.setattr("youtube_toolkit.transcript.local_stt_installed", lambda: True)
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_locally",
        lambda _path: SpeechTranscription(
            text="local text", segments=[], model="local-model"
        ),
    )
    monkeypatch.setattr(
        "youtube_toolkit.transcript.transcribe_in_cloud",
        lambda _path, _vocabulary: SpeechTranscription(
            text="cloud text", segments=[], model="cloud-model", language="en"
        ),
    )

    class FakeClient:
        def metadata(self, _ref):
            return {
                "id": "JpSHyEIZ_bo",
                "language": "de",
                "automatic_captions": {},
                "subtitles": {},
            }

        def download_transcription_audio(self, _metadata, directory):
            return save_audio(directory, downloads)

    document = fetch_transcript("JpSHyEIZ_bo", client=FakeClient(), fresh=False).value

    assert document.source == "speech-to-text"
    assert document.text == expected_text
    assert document.language == expected_language
    assert document.stt_model == expected_model
    assert not downloads[0].parent.exists()


def test_supplied_transcript_metadata_must_match_video(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="Supplied metadata does not match requested video"
    ):
        fetch_transcript("JpSHyEIZ_bo", metadata={"id": "abcdefghijk"}, fresh=False)
    assert not (tmp_path / ".youtube-toolkit/videos").exists()


def save_audio(directory: Path, downloads: list[Path]) -> Path:
    audio_path = directory / "audio.m4a"
    audio_path.write_bytes(b"audio")
    downloads.append(audio_path)
    return audio_path
