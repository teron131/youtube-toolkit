"""Speech-to-text provider requests and response parsing, retry policy, and the local route's audio splitting and model loading, with network calls and models faked."""

import base64
from email.message import Message
import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from youtube_toolkit import ffmpeg_utils
from youtube_toolkit.schemas import SpeechTranscription, TranscriptChunk
from youtube_toolkit.stt import cloud, local
from youtube_toolkit.stt.cloud import (
    PROVIDERS,
    encode_multipart_form,
    parse_transcription,
    run_json_request,
    transcribe_in_cloud,
)


def test_xai_word_timings_become_caption_like_lines() -> None:
    words = [
        ("Hello", 0.2, 0.5),
        ("world.", 0.6, 1.0),
        ("After", 1.1, 1.4),
        ("a", 1.5, 1.6),
        ("pause", 1.7, 2.0),
        ("next", 3.2, 3.5),
        ("line", 3.6, 3.9),
    ]
    result = parse_transcription(
        {
            "text": "Hello world. After a pause next line",
            "language": "en",
            "duration": 4.0,
            "words": [
                {"text": text, "start": start, "end": end} for text, start, end in words
            ],
        },
        PROVIDERS["xai"],
    )

    assert result == SpeechTranscription(
        text="Hello world. After a pause next line",
        segments=[
            TranscriptChunk(0.2, 1.0, "Hello world."),
            TranscriptChunk(1.1, 2.0, "After a pause"),
            TranscriptChunk(3.2, 3.9, "next line"),
        ],
        language="en",
    )


def test_long_unpunctuated_speech_breaks_into_ten_second_lines_and_cjk_joins_without_spaces() -> (
    None
):
    words = [
        {"text": "數據", "start": float(second), "end": second + 0.9}
        for second in range(12)
    ]
    result = parse_transcription(
        {"text": "數據" * 12, "language": "zh", "words": words}, PROVIDERS["xai"]
    )

    assert result is not None
    assert [(line.start_seconds, line.end_seconds) for line in result.segments] == [
        (0, 10.9),
        (11, 11.9),
    ]
    assert result.segments[1].text == "數據"
    assert result.segments[0].text == "數據" * 11


def test_elevenlabs_keeps_only_timed_words_and_multichannel_stays_untimed() -> None:
    single = parse_transcription(
        {
            "text": "transcript text",
            "language_code": "en",
            "words": [
                {"text": "transcript", "type": "word", "start": 0.0, "end": 0.6},
                {"text": " ", "type": "spacing", "start": 0.6, "end": 0.7},
                {"text": "(laughs)", "type": "audio_event", "start": 0.7, "end": 1.2},
                {"text": "text", "type": "word", "start": 1.2, "end": 1.5},
                {"text": "lost", "type": "word", "start": None, "end": None},
            ],
        },
        PROVIDERS["elevenlabs"],
    )
    multichannel = parse_transcription(
        {"transcripts": [{"text": "left"}, {"text": "right"}], "language_code": "en"},
        PROVIDERS["elevenlabs"],
    )

    assert single == SpeechTranscription(
        text="transcript text",
        segments=[TranscriptChunk(0.0, 1.5, "transcript text")],
        language="en",
    )
    assert multichannel == SpeechTranscription(
        text="left right", segments=[], language="en"
    )


@pytest.mark.parametrize(
    ("provider", "key_env", "auth_header", "auth_value", "response", "sent_model"),
    [
        (
            None,
            "XAI_API_KEY",
            "Authorization",
            "Bearer test-key",
            {"text": "数据库", "language": "zh"},
            b"grok-voice-transcribe-2.0",
        ),
        (
            "elevenlabs",
            "ELEVENLABS_API_KEY",
            "Xi-api-key",
            "test-key",
            {"text": "数据库", "language_code": "zh"},
            b"scribe_v2",
        ),
        (
            "openai",
            "OPENAI_API_KEY",
            "Authorization",
            "Bearer test-key",
            {"text": "数据库", "languages": [{"code": "zh"}]},
            b"gpt-transcribe",
        ),
    ],
)
def test_cloud_transcription_follows_the_configured_provider(
    tmp_path,
    monkeypatch,
    provider,
    key_env,
    auth_header,
    auth_value,
    response,
    sent_model,
) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    requests = []

    def fake_run_json_request(request, *, provider_name):
        requests.append(request)
        return response

    if provider is None:
        monkeypatch.delenv("STT_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("STT_PROVIDER", provider)
    monkeypatch.setenv(key_env, "test-key")
    monkeypatch.setattr(cloud, "run_json_request", fake_run_json_request)

    assert transcribe_in_cloud(audio_path) == SpeechTranscription(
        text="数据库", segments=[], language="zh"
    )
    assert requests[0].full_url == PROVIDERS[provider or "xai"].url
    assert requests[0].get_header(auth_header) == auth_value
    assert sent_model in requests[0].data


def test_vercel_gateway_takes_base64_audio_and_the_model_in_a_header(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")
    requests = []

    def fake_run_json_request(request, *, provider_name):
        requests.append(request)
        return {
            "text": "hello",
            "language": "en",
            "segments": [{"text": "hello", "startSecond": 0.5, "endSecond": 1.25}],
        }

    monkeypatch.setenv("STT_PROVIDER", "vercel")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr(cloud, "run_json_request", fake_run_json_request)

    assert transcribe_in_cloud(audio_path) == SpeechTranscription(
        text="hello", segments=[TranscriptChunk(0.5, 1.25, "hello")], language="en"
    )
    (request,) = requests
    assert request.full_url == PROVIDERS["vercel"].url
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Ai-model-id") == "spacexai/grok-stt"
    assert request.get_header("Ai-gateway-protocol-version") == "0.0.1"
    assert json.loads(request.data) == {
        "audio": base64.b64encode(b"audio").decode(),
        "mediaType": "audio/x-m4a",
    }


def test_gemini_uploads_transcribes_by_reference_and_deletes_the_upload(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")
    requests = []
    answer = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"audioTranscription": {"text": "first part"}},
                        {"audioTranscription": {"text": "second part"}},
                    ]
                }
            }
        ]
    }

    def fake_run_json_request(request, *, provider_name):
        requests.append(request)
        if request.get_method() == "DELETE":
            return {}
        if "upload" in request.full_url:
            return {
                "file": {
                    "name": "files/abc",
                    "uri": "https://files.example/abc",
                    "mimeType": "audio/x-m4a",
                }
            }
        return answer

    monkeypatch.setenv("STT_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(cloud, "run_json_request", fake_run_json_request)

    assert transcribe_in_cloud(audio_path) == SpeechTranscription(
        text="first part second part", segments=[]
    )
    upload, generate, delete = requests
    assert upload.data == b"audio"
    assert upload.get_header("X-goog-api-key") == "test-key"
    assert "gemini-3.5-transcribe:generateContent" in generate.full_url
    assert json.loads(generate.data)["contents"][0]["parts"][0]["fileData"] == {
        "fileUri": "https://files.example/abc",
        "mimeType": "audio/x-m4a",
    }
    assert delete.full_url.endswith("/v1beta/files/abc")

    requests.clear()
    answer["candidates"] = [{"finishReason": "STOP"}]
    with pytest.raises(ValueError, match="did not include transcript text"):
        transcribe_in_cloud(audio_path)
    assert [request.get_method() for request in requests] == ["POST", "POST", "DELETE"]


def test_cloud_check_uploads_a_test_tone_and_status_sends_nothing(monkeypatch) -> None:
    requests = []

    def fake_run_json_request(request, *, provider_name):
        requests.append(request)
        return {"text": ""}

    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(cloud, "run_json_request", fake_run_json_request)

    status = {"provider": "xAI", "model": "grok-voice-transcribe-2.0", "key_set": True}
    assert cloud.cloud_stt_status() == status
    assert requests == []

    assert cloud.check_cloud_stt() == {**status, "working": True}
    assert b"grok-voice-transcribe-2.0" in requests[0].data
    assert b'filename="test-tone.wav"' in requests[0].data
    assert b"RIFF" in requests[0].data

    monkeypatch.delenv("XAI_API_KEY")
    assert cloud.cloud_stt_status()["key_set"] is False
    with pytest.raises(ValueError, match="XAI_API_KEY is required"):
        cloud.check_cloud_stt()


def test_unknown_provider_falls_back_to_the_default_with_a_warning(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("STT_PROVIDER", "nobody")
    cloud._known_provider.cache_clear()

    assert cloud.cloud_provider() is PROVIDERS["xai"]
    assert cloud.cloud_provider() is PROVIDERS["xai"]
    assert caplog.text.count("STT_PROVIDER=nobody") == 1
    assert "falling back to the default xai" in caplog.text


def test_cloud_transcription_requires_the_provider_key(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="XAI_API_KEY is required"):
        transcribe_in_cloud(audio_path)


def test_encode_multipart_form_handles_ytdlp_default_containers(tmp_path) -> None:
    webm_path = tmp_path / "audio.webm"
    m4a_path = tmp_path / "audio.m4a"
    webm_path.write_bytes(b"webm-audio")
    m4a_path.write_bytes(b"m4a-audio")

    webm_body, _webm_content_type = encode_multipart_form(
        fields={},
        file_path=webm_path,
    )
    m4a_body, _m4a_content_type = encode_multipart_form(
        fields={},
        file_path=m4a_path,
    )

    assert b"Content-Type: audio/webm" in webm_body
    assert b"Content-Type: audio/x-m4a" in m4a_body


def test_stt_request_retries_transient_network_failures(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"text": "transcript", "language": "en"}'

    def fake_urlopen(_request, *, timeout: float):
        nonlocal attempts
        assert timeout == cloud.REQUEST_TIMEOUT_SECONDS
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()

    monkeypatch.setattr(
        "youtube_toolkit.stt.cloud.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)

    result = run_json_request(
        urllib.request.Request("https://api.example.test/stt"),
        provider_name="Test",
    )

    assert result == {"text": "transcript", "language": "en"}
    assert attempts == 3
    assert delays == [1.0, 2.0]


def test_stt_request_does_not_retry_permanent_client_failure(monkeypatch) -> None:
    attempts = 0
    headers = Message()

    def fake_urlopen(_request, *, timeout: float):
        nonlocal attempts
        assert timeout == cloud.REQUEST_TIMEOUT_SECONDS
        attempts += 1
        raise urllib.error.HTTPError(
            "https://api.example.test/stt",
            401,
            "Unauthorized",
            headers,
            io.BytesIO(b"invalid API key"),
        )

    monkeypatch.setattr(
        "youtube_toolkit.stt.cloud.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "youtube_toolkit.execution.time.sleep",
        lambda _delay: pytest.fail("permanent failures must not back off and retry"),
    )

    with pytest.raises(RuntimeError, match=r"401.*invalid API key"):
        run_json_request(
            urllib.request.Request("https://api.example.test/stt"),
            provider_name="Test",
        )

    assert attempts == 1


def test_local_pieces_become_timed_segments_at_their_offsets(
    monkeypatch, tmp_path
) -> None:
    rate = ffmpeg_utils.SAMPLE_RATE
    pieces = [np.zeros(seconds * rate, dtype=np.float32) for seconds in (28, 10, 5)]
    monkeypatch.setattr(local, "decode_audio", lambda _path: np.concatenate(pieces))
    monkeypatch.setattr(local, "split_at_quiet_points", lambda _samples: pieces)
    monkeypatch.setattr(
        local,
        "load_recognizer",
        lambda: lambda _inputs: [{"text": " first "}, {"text": " "}, {"text": "third"}],
    )

    assert local.transcribe_locally(tmp_path / "audio.m4a") == SpeechTranscription(
        text="first third",
        segments=[TranscriptChunk(0, 28, "first"), TranscriptChunk(38, 43, "third")],
    )


def test_long_audio_is_cut_in_the_quiet_gap_before_the_limit() -> None:
    rate = ffmpeg_utils.SAMPLE_RATE
    samples = np.ones(25 * rate, dtype=np.float32)
    samples[8 * rate : 9 * rate] = 0.0
    samples[17 * rate : 18 * rate] = 0.0

    pieces = local.split_at_quiet_points(samples, max_seconds=10)

    assert [len(piece) <= 10 * rate for piece in pieces] == [True, True, True]
    assert np.array_equal(np.concatenate(pieces), samples)
    assert 8 * rate < len(pieces[0]) < 9 * rate
    assert 17 * rate < len(pieces[0]) + len(pieces[1]) < 18 * rate


def test_model_check_runs_a_test_tone_and_reports_where_it_ran(monkeypatch) -> None:
    heard = []

    class FakeRecognizer:
        device = "cpu"
        model = type(
            "Model",
            (),
            {"dtype": "torch.float32", "name_or_path": "example/speech-model"},
        )()

        def __call__(self, audio):
            heard.append(audio)
            return {"text": ""}

    decoded = []

    def fake_decode_audio(path):
        decoded.append((path.name, path.read_bytes()[:4]))
        return np.zeros(8, dtype=np.float32)

    monkeypatch.setenv("STT_LOCAL_MODEL", "example/speech-model")
    monkeypatch.setattr(local, "load_recognizer", FakeRecognizer)
    monkeypatch.setattr(local, "decode_audio", fake_decode_audio)

    assert local.check_local_stt() == {
        "model": "example/speech-model",
        "device": "cpu",
        "precision": "float32",
        "working": True,
    }
    assert decoded == [("test-tone.wav", b"RIFF")]
    assert len(heard[0]["raw"]) == 8

    def broken():
        raise OSError("model cannot be loaded")

    monkeypatch.setattr(local, "load_recognizer", broken)
    with pytest.raises(OSError, match="cannot be loaded"):
        local.check_local_stt()


def test_unloadable_local_model_falls_back_to_the_default_with_a_warning(
    monkeypatch, caplog
) -> None:
    built = []

    def fake_build(model):
        built.append(model)
        if model != "nvidia/parakeet-tdt-0.6b-v3":
            raise OSError(f"{model} is not a valid model identifier\nmore detail")
        return "default recognizer"

    monkeypatch.setenv("STT_LOCAL_MODEL", "example/typo")
    monkeypatch.setattr(local, "_build_recognizer", fake_build)

    assert local.load_recognizer() == "default recognizer"
    assert built == ["example/typo", "nvidia/parakeet-tdt-0.6b-v3"]
    assert "STT_LOCAL_MODEL=example/typo could not be loaded" in caplog.text
    assert "more detail" not in caplog.text

    def broken(model):
        raise OSError("no network")

    monkeypatch.delenv("STT_LOCAL_MODEL")
    monkeypatch.setattr(local, "_build_recognizer", broken)
    with pytest.raises(OSError, match="no network"):
        local.load_recognizer()
