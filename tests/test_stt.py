from dataclasses import fields
from email.message import Message
import io
import urllib.error
import urllib.request

import pytest

from youtube_toolkit.stt import (
    SpeechToTextOptions,
    SpeechTranscription,
    transcribe_audio,
    transcribe_audio_details,
)
from youtube_toolkit.stt.transport import encode_multipart_form, run_json_request


def test_xai_stt_response_normalizes_to_plain_contract() -> None:
    result = SpeechTranscription.from_xai_stt(
        {
            "text": "transcript text",
            "language": "en",
            "duration": 42.5,
            "words": [{"word": "transcript"}, {"word": "text"}],
        }
    )

    assert result == SpeechTranscription(
        text="transcript text",
        language="en",
    )


def test_elevenlabs_stt_response_normalizes_to_plain_contract() -> None:
    result = SpeechTranscription.from_elevenlabs_stt(
        {
            "text": "transcript text",
            "language_code": "en",
            "words": [
                {"text": "transcript", "start": 0, "end": 0.5, "type": "word"},
                {"text": "text", "start": 0.5, "end": 1.0, "type": "word"},
            ],
        }
    )

    assert result == SpeechTranscription(
        text="transcript text",
        language="en",
    )


def test_transcribe_audio_returns_plain_provider_text(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")

    def fake_xai_stt(
        *_args,
        **_kwargs,
    ):
        return {"text": "数据库", "language": "zh"}

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.stt.xai.xai_stt",
        fake_xai_stt,
    )
    result = transcribe_audio(audio_path)

    assert "opencc_config" not in {field.name for field in fields(SpeechToTextOptions)}
    assert result == "数据库"


def test_transcribe_audio_details_returns_plain_contract(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")

    def fake_xai_stt(
        *_args,
        **_kwargs,
    ):
        return {
            "text": "数据库",
            "language": "zh",
            "duration": 12.5,
            "words": [{"word": "数据库"}],
        }

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.stt.xai.xai_stt",
        fake_xai_stt,
    )

    assert transcribe_audio_details(audio_path) == SpeechTranscription(
        text="数据库",
        language="zh",
    )


def test_encode_multipart_form_handles_ytdlp_default_containers(tmp_path) -> None:
    """STT uploads include useful MIME hints for common yt-dlp audio outputs."""

    webm_path = tmp_path / "audio.webm"
    m4a_path = tmp_path / "audio.m4a"
    webm_path.write_bytes(b"webm-audio")
    m4a_path.write_bytes(b"m4a-audio")

    webm_body, _webm_content_type = encode_multipart_form(
        fields={},
        file_field="file",
        file_path=webm_path,
    )
    m4a_body, _m4a_content_type = encode_multipart_form(
        fields={},
        file_field="file",
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
        assert timeout == 30
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()

    monkeypatch.setattr(
        "youtube_toolkit.stt.transport.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)

    result = run_json_request(
        urllib.request.Request("https://api.example.test/stt"),
        timeout_seconds=30,
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
        assert timeout == 30
        attempts += 1
        raise urllib.error.HTTPError(
            "https://api.example.test/stt",
            401,
            "Unauthorized",
            headers,
            io.BytesIO(b"invalid API key"),
        )

    monkeypatch.setattr(
        "youtube_toolkit.stt.transport.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr(
        "youtube_toolkit.execution.time.sleep",
        lambda _delay: pytest.fail("permanent failures must not back off and retry"),
    )

    with pytest.raises(RuntimeError, match=r"401.*invalid API key"):
        run_json_request(
            urllib.request.Request("https://api.example.test/stt"),
            timeout_seconds=30,
            provider_name="Test",
        )

    assert attempts == 1
