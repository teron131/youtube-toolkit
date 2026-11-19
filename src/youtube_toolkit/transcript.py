"""Transcript retrieval policy: captions first, then speech-to-text when allowed and set up.

Chinese text is converted to Taiwan Traditional, and each document is saved in the video's folder, where a later fetch with the same speech-to-text choice reuses it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from .artifacts import Cached, reuse, temporary_directory, video_directory
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

    def download_transcription_audio(
        self, metadata: dict[str, Any], directory: Path, /
    ) -> Path: ...


def fetch_transcript(
    video_ref: VideoRef,
    *,
    stt: Literal["never", "fallback", "always"] = "fallback",
    client: TranscriptClient | None = None,
    metadata: dict[str, Any] | None = None,
    fresh: bool,
) -> Cached[TranscriptDocument]:
    """With "fallback", speech-to-text runs only when captions are missing, and when it is not set up the caption failure stands without any audio download. "always" skips captions and fails when speech-to-text is not set up.

    A document saved by an earlier fetch with the same `stt` comes back without fetching unless `fresh`, even when the video has gained captions since or the speech-to-text settings changed.
    Supplied metadata must be a response the current operation fetched for this video, and one for another video raises ValueError.
    """
    if metadata is not None and metadata.get("id") != require_video_id(video_ref):
        raise ValueError("Supplied metadata does not match requested video")
    fetcher = client or YtDlpClient()
    payload = fetcher.metadata(video_ref) if metadata is None else metadata
    video = VideoMetadata.from_yt_dlp(payload)
    entry = reuse(
        video_directory(video.video_id, video.title) / "transcripts",
        f"original_{stt}",
        {"stt": stt, "subtitle_languages": list(SUBTITLE_LANGUAGES)},
        fresh=fresh,
        compute=lambda: asdict(_fetch_document(payload, stt=stt, client=fetcher)),
    )
    return Cached(TranscriptDocument.from_dict(entry.value), entry.reused, entry.path)


def _fetch_document(
    metadata: dict[str, Any],
    *,
    stt: Literal["never", "fallback", "always"],
    client: TranscriptClient,
) -> TranscriptDocument:
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
    if is_japanese(document.text):
        return document
    return replace(
        document,
        segments=[
            replace(segment, text=convert_zh(segment.text))
            for segment in document.segments
        ],
    )


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
    with temporary_directory("audio-") as directory:
        audio_path = client.download_transcription_audio(metadata_payload, directory)
        transcription = transcribe(audio_path, metadata_names(metadata))
    return TranscriptDocument(
        metadata=metadata,
        language=transcription.language
        or normalize_text(metadata_payload.get("language"))
        or "speech-to-text",
        source="speech-to-text",
        segments=transcription.segments
        or [TranscriptSegment(start_seconds=0, end_seconds=0, text=transcription.text)],
        stt_model=transcription.model,
    )
