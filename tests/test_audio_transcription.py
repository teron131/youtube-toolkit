"""Local audio transcription workflow tests."""

from youtube_toolkit.audio_transcription import resolve_audio_video_id, store_audio_transcript
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

    assert {
        key: payload[key]
        for key in (
            "video_id",
            "title",
            "upload_date",
            "duration",
            "view_count",
            "like_count",
            "channel_id",
            "channel_handle",
            "transcript",
            "language",
        )
    } == {
        "video_id": "JpSHyEIZ_bo",
        "title": None,
        "upload_date": None,
        "duration": None,
        "view_count": None,
        "like_count": None,
        "channel_id": None,
        "channel_handle": None,
        "transcript": "數據庫",
        "language": "zh",
    }
    assert payload["metadata_fetched_at"] is None
    assert payload["transcript_fetched_at"] is not None
    with get_connection() as connection:
        stored_video = get_stored_video(connection, "JpSHyEIZ_bo")
    assert stored_video is not None
    assert stored_video["transcript"] == "數據庫"
    assert not audio_path.exists()


def test_resolve_audio_video_id_accepts_unique_download_stem(tmp_path) -> None:
    audio_path = tmp_path / "JpSHyEIZ_bo-6d6d4debe080414eb01bc9a76d88be2c.m4a"

    assert resolve_audio_video_id(audio_path, {}) == "JpSHyEIZ_bo"
