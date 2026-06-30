from dataclasses import fields

from youtube_toolkit.stt import (
    SpeechToTextOptions,
    SpeechTranscription,
    transcribe_audio,
    transcribe_audio_details,
)
from youtube_toolkit.stt.common import encode_multipart_form


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
        duration_seconds=42.5,
        word_count=2,
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
        duration_seconds=1.0,
        word_count=2,
    )


def test_transcribe_audio_returns_plain_provider_text(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")

    def fake_request_speech_to_text(
        *_args,
        **_kwargs,
    ):
        return {"text": "数据库", "language": "zh"}

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.stt.xai.request_speech_to_text",
        fake_request_speech_to_text,
    )
    result = transcribe_audio(audio_path)

    assert "opencc_config" not in {field.name for field in fields(SpeechToTextOptions)}
    assert result == "数据库"


def test_transcribe_audio_details_preserves_provider_fields(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"audio")

    def fake_request_speech_to_text(
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
        "youtube_toolkit.stt.xai.request_speech_to_text",
        fake_request_speech_to_text,
    )

    assert transcribe_audio_details(audio_path) == SpeechTranscription(
        text="数据库",
        language="zh",
        duration_seconds=12.5,
        word_count=1,
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
