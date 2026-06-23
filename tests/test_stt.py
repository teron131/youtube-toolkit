import sqlite3
from dataclasses import fields

from youtube_toolkit.storage import get_connection, upsert_video
from youtube_toolkit.stt import (
    SpeechToTextOptions,
    SpeechTranscription,
    transcribe_audio_file,
    transcribe_audio_to_db,
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


def test_stt_transcription_is_hardcoded_to_s2hk(tmp_path, monkeypatch) -> None:
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
    result = transcribe_audio_file(audio_path)

    assert "opencc_config" not in {field.name for field in fields(SpeechToTextOptions)}
    assert result.text == "數據庫"


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


def test_transcribe_audio_to_db_stores_first_then_returns_json_payload(
    tmp_path,
    monkeypatch,
) -> None:
    storage_home = tmp_path / "state"
    audio_path = tmp_path / "JpSHyEIZ_bo.mp3"
    audio_path.write_bytes(b"audio")

    def fake_request_speech_to_text(
        *_args,
        **_kwargs,
    ):
        return {
            "text": "数据库",
            "language": "zh",
            "duration": 12.5,
            "words": [{"word": "数据"}, {"word": "库"}],
        }

    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "youtube_toolkit.stt.xai.request_speech_to_text",
        fake_request_speech_to_text,
    )

    payload = transcribe_audio_to_db(audio_path)

    assert payload == {
        "video_id": "JpSHyEIZ_bo",
        "database_row_id": 1,
        "title": None,
        "upload_date": None,
        "view_count": None,
        "channel_id": None,
        "channel_handle": None,
        "transcript": "數據庫",
        "language": "zh",
        "duration_seconds": 12.5,
        "word_count": 2,
    }
    connection = sqlite3.connect(storage_home / "youtube.sqlite3")
    try:
        row = connection.execute(
            "SELECT transcript FROM videos WHERE youtube_video_id = ?",
            ("JpSHyEIZ_bo",),
        ).fetchone()
    finally:
        connection.close()
    assert row == ("數據庫",)
    assert not audio_path.exists()


def test_transcribe_audio_to_db_returns_existing_transcript_before_stt(
    tmp_path,
    monkeypatch,
) -> None:
    storage_home = tmp_path / "state"
    audio_path = tmp_path / "JpSHyEIZ_bo.mp3"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with get_connection() as connection:
        upsert_video(
            connection,
            {"id": "JpSHyEIZ_bo"},
            transcript="既有逐字稿",
        )

    payload = transcribe_audio_to_db(audio_path)

    assert payload == {
        "video_id": "JpSHyEIZ_bo",
        "database_row_id": 1,
        "title": None,
        "upload_date": None,
        "view_count": None,
        "channel_id": None,
        "channel_handle": None,
        "transcript": "既有逐字稿",
        "language": None,
        "duration_seconds": None,
        "word_count": None,
    }
    assert not audio_path.exists()
