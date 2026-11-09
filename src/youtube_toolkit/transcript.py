"""Transcript retrieval policy: prefer captions, fall back to cloud and then local speech-to-text when allowed and set up, which listens for the names written in the title and description, convert Chinese text to Taiwan Traditional, and archive each fresh document."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from .artifacts import write_artifact
from .convert_zh import convert_zh, is_japanese
from .identity import VideoRef, require_video_id
from .names import metadata_names
from .schemas import (
    SpeechTranscription,
    TranscriptDocument,
    TranscriptSegment,
    VideoMetadata,
    normalize_text,
)
from .stt import (
    cloud_stt_configured,
    local_stt_installed,
    transcribe_in_cloud,
    transcribe_locally,
)
from .youtube import (
    CaptionTranscriptUnavailable,
    YtDlpClient,
    fetch_caption_transcript,
)

# Ranks manual subtitles only; automatic captions are taken in the video's spoken language, whatever it is.
SUBTITLE_LANGUAGES = (
    "en",
    "zh-Hant",
    "zh-HK",
    "yue-HK",
    "yue",
    "zh-TW",
    "zh-Hans",
    "zh-CN",
    "zh",
)


class TranscriptClient(Protocol):
    def metadata(self, video_ref: VideoRef, /) -> dict[str, Any]: ...

    def download_transcription_audio(self, video_ref: VideoRef, /) -> Path:
        """The caller deletes the returned file."""
        ...


def fetch_transcript(
    video_ref: VideoRef,
    *,
    stt: Literal["never", "fallback", "always"] = "fallback",
    client: TranscriptClient | None = None,
    metadata: dict[str, Any] | None = None,
) -> TranscriptDocument:
    """With "fallback", speech-to-text runs only when captions are missing, and when it is not set up the caption failure stands without any audio download. "always" skips captions and fails when speech-to-text is not set up.

    Supplied metadata must be a response the current operation fetched for this video; otherwise metadata is fetched fresh.
    """
    video_id = require_video_id(video_ref)
    if metadata is not None and metadata.get("id") != video_id:
        raise ValueError("Supplied metadata does not match requested video")
    client = client or YtDlpClient()
    if metadata is None:
        metadata = client.metadata(video_ref)
    if stt == "always":
        transcribe = _stt_transcriber()
        if transcribe is None:
            raise RuntimeError(
                "Speech-to-text is not set up; `youtube doctor` names the next step"
            )
        document = _fetch_stt_transcript(metadata, client=client, transcribe=transcribe)
    else:
        try:
            document = fetch_caption_transcript(metadata, languages=SUBTITLE_LANGUAGES)
        except CaptionTranscriptUnavailable:
            transcribe = None if stt == "never" else _stt_transcriber()
            if transcribe is None:
                raise
            document = _fetch_stt_transcript(
                metadata, client=client, transcribe=transcribe
            )
    if not is_japanese(document.text):
        document = replace(
            document,
            segments=[
                replace(segment, text=convert_zh(segment.text))
                for segment in document.segments
            ],
        )
    request = {
        "video_id": video_id,
        "languages": list(SUBTITLE_LANGUAGES),
        "stt": stt,
    }
    write_artifact("transcripts", video_id, request, asdict(document))
    return document


def _stt_transcriber() -> Callable[[Path, list[str]], SpeechTranscription] | None:
    """The transcriber takes the audio and a vocabulary of names to listen for, which local models ignore."""
    if cloud_stt_configured():
        return transcribe_in_cloud
    if local_stt_installed():
        return lambda audio_path, _: transcribe_locally(audio_path)
    return None


def _fetch_stt_transcript(
    metadata_payload: dict[str, Any],
    *,
    client: TranscriptClient,
    transcribe: Callable[[Path, list[str]], SpeechTranscription],
) -> TranscriptDocument:
    metadata = VideoMetadata.from_yt_dlp(metadata_payload)
    audio_path = client.download_transcription_audio(metadata.video_id)
    try:
        transcription = transcribe(audio_path, metadata_names(metadata))
    finally:
        audio_path.unlink(missing_ok=True)
    return TranscriptDocument(
        metadata=metadata,
        language=transcription.language
        or normalize_text(metadata_payload.get("language"))
        or "speech-to-text",
        source="speech-to-text",
        segments=transcription.segments
        or [TranscriptSegment(start_seconds=0, end_seconds=0, text=transcription.text)],
    )
