"""Local audio transcription workflows that persist YouTube transcript state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .convert_zh import convert_zh
from .storage import get_connection, get_stored_video, upsert_video
from .stt import SpeechTranscription, transcribe_audio_details
from .url import extract_video_id


def transcribe_audio_to_storage(audio_path: Path) -> dict[str, Any]:
    """Transcribe a YouTube-named local audio file and persist the transcript."""
    video_id = extract_video_id(audio_path.stem)
    if video_id is None:
        raise ValueError("audio filename must include a YouTube video ID so the transcript can be stored")

    with get_connection() as connection:
        stored_video = get_stored_video(
            connection,
            video_id,
            require_transcript=True,
        )
    if stored_video is not None:
        audio_path.unlink(missing_ok=True)
        return stored_transcription_payload(stored_video)

    transcription = transcribe_audio_details(audio_path)
    transcript = convert_zh(transcription.text, "s2hk")
    with get_connection() as connection:
        stored_video = upsert_video(
            connection,
            {"id": video_id},
            transcript=transcript,
        )
    if stored_video is None or not stored_video["transcript"]:
        raise RuntimeError(f"Stored transcript could not be found after writing video {video_id}")
    audio_path.unlink()
    return stored_transcription_payload(
        stored_video,
        transcription=transcription,
    )


def stored_transcription_payload(
    stored_video: dict[str, Any],
    *,
    transcription: SpeechTranscription | None = None,
) -> dict[str, Any]:
    return {
        **stored_video,
        "language": transcription.language if transcription is not None else None,
        "duration_seconds": transcription.duration_seconds if transcription is not None else None,
        "word_count": transcription.word_count if transcription is not None else None,
    }
