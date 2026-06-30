"""Local audio transcription workflow tests."""

from youtube_toolkit.audio_transcription import store_audio_transcript
from youtube_toolkit.storage import get_connection, get_stored_video
from youtube_toolkit.stt import SpeechTranscription


def test_store_audio_transcript_stores_transcript_and_returns_json_payload(
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
        )

    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.setattr("youtube_toolkit.audio_transcription.transcribe_audio_details", fake_transcribe_audio_details)

    payload = store_audio_transcript(audio_path)

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
    }
    with get_connection() as connection:
        stored_video = get_stored_video(connection, "JpSHyEIZ_bo")
    assert stored_video is not None
    assert stored_video["transcript"] == "數據庫"
    assert not audio_path.exists()
