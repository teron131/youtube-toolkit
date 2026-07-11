"""Local audio transcription persists DB-backed YouTube transcript state for caption fallbacks and agent calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .convert_zh import convert_zh
from .identity import extract_video_id
from .storage import get_connection, upsert_video
from .stt import transcribe_audio_details


def store_audio_transcript(
    audio_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transcribe local audio, persist the transcript, and return the stored payload."""
    video_metadata = metadata or {}
    video_id = resolve_audio_video_id(audio_path, video_metadata)
    transcription = transcribe_audio_details(audio_path)
    transcript = convert_zh(transcription.text, "s2hk")
    with get_connection() as connection:
        stored_video = upsert_video(
            connection,
            {**video_metadata, "id": video_id},
            transcript=transcript,
        )
    if stored_video is None or not stored_video["transcript"]:
        raise RuntimeError(f"Stored transcript could not be found after writing video {video_id}")
    audio_path.unlink()
    return {
        **stored_video,
        "language": transcription.language,
    }


def resolve_audio_video_id(
    audio_path: Path,
    metadata: dict[str, Any],
) -> str:
    """Resolve the storage video ID from trusted metadata first, then from the downloaded audio filename."""
    metadata_video_id = metadata.get("id")
    if isinstance(metadata_video_id, str) and metadata_video_id.strip():
        return metadata_video_id.strip()

    video_id = extract_video_id(audio_path.stem)
    if video_id is None and len(audio_path.stem) > 11 and audio_path.stem[11] == "-":
        video_id = extract_video_id(audio_path.stem[:11])
    if video_id is None:
        raise ValueError("audio filename must include a YouTube video ID so the transcript can be stored")
    return video_id
