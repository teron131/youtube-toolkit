import base64
from email.message import Message
import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from youtube_toolkit.schemas import SpeechTranscription
from youtube_toolkit.stt import cloud, local
from youtube_toolkit.stt.cloud import (
    PROVIDERS,
    encode_multipart_form,
    parse_transcription,
    run_json_request,
    transcribe_in_cloud,
)


def test_xai_stt_response_normalizes_to_plain_contract() -> None:
    result = parse_transcription(
        {
            "text": "transcript text",
            "language": "en",
            "duration": 42.5,
            "words": [{"word": "transcript"}, {"word": "text"}],
        },
        PROVIDERS["xai"],
    )

    assert result == SpeechTranscription(text="transcript text", language="en")


def test_elevenlabs_stt_response_normalizes_to_plain_contract() -> None:
    """ElevenLabs names its language field differently and may split text per channel."""
    single = parse_transcription(
        {"text": "transcript text", "language_code": "en"}, PROVIDERS["elevenlabs"]
    )
    multichannel = parse_transcription(
        {"transcripts": [{"text": "left"}, {"text": "right"}], "language_code": "en"},
        PROVIDERS["elevenlabs"],
    )

    assert single == SpeechTranscription(text="transcript text", language="en")
    assert multichannel == SpeechTranscription(text="left right", language="en")


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
    """STT_PROVIDER picks the endpoint, credentials, and model; xAI is the default."""
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
        text="数据库", language="zh"
    )
    assert requests[0].full_url == PROVIDERS[provider or "xai"].url
    assert requests[0].get_header(auth_header) == auth_value
    assert sent_model in requests[0].data


def test_vercel_gateway_takes_base64_audio_and_the_model_in_a_header(
    tmp_path, monkeypatch
) -> None:
    """The gateway accepts no file upload: audio travels as base64 in JSON and the model as a header."""
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")
    requests = []

    def fake_run_json_request(request, *, provider_name):
        requests.append(request)
        return {"text": "hello", "language": "en", "segments": []}

    monkeypatch.setenv("STT_PROVIDER", "vercel")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr(cloud, "run_json_request", fake_run_json_request)

    assert transcribe_in_cloud(audio_path) == SpeechTranscription(
        text="hello", language="en"
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
    """Gemini reads audio from its file store, so one transcript is three requests, and the stored copy is removed even with no speech."""
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
        text="first part second part"
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
    """The billed check proves the provider accepts the key and model; the status alone never makes a request."""
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
    """STT uploads include useful MIME hints for common yt-dlp audio outputs."""

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
    """Provider-neutral STT requests should make three total attempts on transient failures."""
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
    """Authentication and other permanent client errors should not duplicate paid STT work."""
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


def test_long_audio_is_cut_in_the_quiet_gap_before_the_limit() -> None:
    """Pieces stay within the limit, lose no samples, and split inside the silence instead of inside speech."""
    rate = local.SAMPLE_RATE
    samples = np.ones(25 * rate, dtype=np.float32)
    samples[8 * rate : 9 * rate] = 0.0
    samples[17 * rate : 18 * rate] = 0.0

    pieces = local.split_at_quiet_points(samples, max_seconds=10)

    assert [len(piece) <= 10 * rate for piece in pieces] == [True, True, True]
    assert np.array_equal(np.concatenate(pieces), samples)
    assert 8 * rate < len(pieces[0]) < 9 * rate
    assert 17 * rate < len(pieces[0]) + len(pieces[1]) < 18 * rate


def test_model_check_runs_a_test_tone_and_reports_where_it_ran(monkeypatch) -> None:
    """The setup check decodes the shared test tone, runs the loaded model on it, and names its device and precision; a failing model raises."""
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
    """A wrong STT_LOCAL_MODEL still yields a working recognizer, built from the default model, and says so; a broken default raises."""
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
