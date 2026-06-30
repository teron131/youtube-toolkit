"""Local audio transcription workflow tests."""

import sqlite3

from youtube_toolkit.audio_transcription import transcribe_audio_to_storage
from youtube_toolkit.storage import get_connection, upsert_video
from youtube_toolkit.stt import SpeechTranscription


def test_transcribe_audio_to_storage_stores_first_then_returns_json_payload(
    tmp_path,
    monkeypatch,
) -> None:
    storage_home = tmp_path / "state"
    audio_path = tmp_path / "JpSHyEIZ_bo.mp3"
    audio_path.write_bytes(b"audio")

    def fake_transcribe_audio_details(*_args, **_kwargs):
        return SpeechTranscription(
            text="数据库",
            language="zh",
            duration_seconds=12.5,
            word_count=2,
        )

    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.setattr("youtube_toolkit.audio_transcription.transcribe_audio_details", fake_transcribe_audio_details)

    payload = transcribe_audio_to_storage(audio_path)

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


def test_transcribe_audio_to_storage_returns_existing_transcript_before_stt(
    tmp_path,
    monkeypatch,
) -> None:
    storage_home = tmp_path / "state"
    audio_path = tmp_path / "JpSHyEIZ_bo.mp3"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    def fail_transcribe_audio_details(*_args, **_kwargs):
        raise AssertionError("STT should not run when the transcript is already stored")

    monkeypatch.setattr("youtube_toolkit.audio_transcription.transcribe_audio_details", fail_transcribe_audio_details)

    with get_connection() as connection:
        upsert_video(
            connection,
            {"id": "JpSHyEIZ_bo"},
            transcript="既有逐字稿",
        )

    payload = transcribe_audio_to_storage(audio_path)

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
