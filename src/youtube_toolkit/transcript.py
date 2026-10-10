"""Transcript retrieval policy: prefer captions, fall back to speech-to-text when allowed, archive each fresh document, and export readable text with overwrite protection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from typing import Any, Protocol, TypedDict
import unicodedata

from .artifacts import write_artifact
from .convert_zh import convert_zh
from .identity import VideoRef, require_video_id
from .schemas import TranscriptChunk, TranscriptDocument, VideoMetadata
from .stt import transcribe_audio_details
from .youtube.caption import CaptionTranscriptUnavailable, fetch_caption_transcript
from .youtube.client import YtDlpClient

DEFAULT_TRANSCRIPT_LANGUAGES = (
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
TRANSCRIPT_LANGUAGE_LABELS = {
    "en-orig": "English",
    "en": "English",
    "zh-Hant": "Chinese (Traditional)",
    "zh-HK": "Chinese (Hong Kong)",
    "yue-HK": "Cantonese",
    "yue": "Cantonese",
    "zh-TW": "Chinese (Traditional)",
    "zh-Hans": "Chinese (Simplified)",
    "zh-CN": "Chinese (Simplified)",
    "zh": "Chinese",
}
TRANSCRIPT_SOURCE_LABELS = {
    "automatic_captions": "automatic captions",
    "subtitles": "manual captions",
    "auto": "automatic captions",
    "manual": "manual captions",
}
UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_WHITESPACE = re.compile(r"\s+")
REPEATED_UNDERSCORES = re.compile(r"_+")


@dataclass(frozen=True, slots=True)
class TranscriptOptions:
    """Caller policy for caption language, source, and permission to fall back to untimed STT.

    Languages rank manual subtitles; automatic captions are used only in the spoken language of the video, never as machine translations.
    """

    languages: tuple[str, ...] = DEFAULT_TRANSCRIPT_LANGUAGES
    source: str = "any"
    allow_stt: bool = True


class TranscriptClient(Protocol):
    """yt-dlp surface for caption metadata and the speech-to-text audio fallback."""

    def metadata(self, video_ref: VideoRef, /) -> dict[str, Any]:
        """Fetch raw video metadata containing caption URLs and fallback STT identity."""
        ...

    def download_transcription_audio(self, video_ref: VideoRef, /) -> Path:
        """Download one temporary audio file for the transcript STT fallback."""
        ...


class TranscriptExportResult(TypedDict):
    """File created by an explicit transcript export request."""

    video_id: str
    path: str


def fetch_transcript(
    video_ref: VideoRef,
    *,
    options: TranscriptOptions | None = None,
    client: TranscriptClient | None = None,
    metadata: dict[str, Any] | None = None,
) -> TranscriptDocument:
    """Prefer captions, fall back to STT when allowed, and archive the new document.

    Text from either source is returned in Taiwan Traditional script.

    Supplied metadata must be a response the current operation fetched for this video; otherwise metadata is fetched fresh.
    """
    options = options or TranscriptOptions()
    video_id = require_video_id(video_ref)
    if metadata is not None and metadata.get("id") != video_id:
        raise ValueError("Supplied metadata does not match requested video")
    client = client or YtDlpClient()
    if metadata is None:
        metadata = client.metadata(video_ref)
    try:
        document = fetch_caption_transcript(
            metadata, languages=options.languages, source=options.source
        )
    except CaptionTranscriptUnavailable:
        if not options.allow_stt:
            raise
        document = _fetch_stt_transcript(metadata, client=client)
    document = replace(
        document,
        chunks=[
            replace(chunk, text=convert_zh(chunk.text)) for chunk in document.chunks
        ],
    )
    request = {
        "video_id": video_id,
        "languages": list(options.languages),
        "source": options.source,
        "allow_stt": options.allow_stt,
    }
    write_artifact("transcripts", video_id, request, asdict(document))
    return document


def write_transcript_export(
    document: TranscriptDocument,
    *,
    timestamps: bool,
    force: bool,
) -> TranscriptExportResult:
    """Write one deterministic UTF-8 transcript file without silently replacing an existing export."""
    path = Path.cwd() / _export_filename(document.metadata)
    try:
        with path.open(
            "w" if force else "x", encoding="utf-8", newline="\n"
        ) as export_file:
            export_file.write(_render_export(document, timestamps=timestamps))
    except FileExistsError as error:
        raise ValueError(
            f"Transcript export already exists: {path}; use --force to replace it"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Could not write transcript export {path}: {error}"
        ) from error
    return {
        "video_id": document.metadata.video_id,
        "path": str(path),
    }


def _fetch_stt_transcript(
    metadata_payload: dict[str, Any],
    *,
    client: TranscriptClient,
) -> TranscriptDocument:
    """Download temporary audio, run STT, and return its untimed text as one transcript chunk in the detected language."""
    metadata = VideoMetadata.from_yt_dlp(metadata_payload)
    audio_path = client.download_transcription_audio(metadata.video_id)
    try:
        transcription = transcribe_audio_details(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
    return TranscriptDocument(
        metadata=metadata,
        language=transcription.language or "speech-to-text",
        source="speech-to-text",
        chunks=[
            TranscriptChunk(
                start_seconds=0,
                end_seconds=0,
                text=transcription.text,
            )
        ],
    )


def _export_filename(metadata: VideoMetadata) -> str:
    """Build a portable title-and-channel filename without exposing the video ID."""
    title = _normalize_filename_component(metadata.title or "") or "transcript"
    channel = _normalize_filename_component(metadata.channel or "")
    stem = "_".join(part for part in (title, channel) if part)
    return f"{stem}.txt"


def _normalize_filename_component(value: str) -> str:
    """Normalize human-readable metadata for a portable filename component."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = UNSAFE_FILENAME_CHARACTERS.sub("", normalized)
    normalized = FILENAME_WHITESPACE.sub("_", normalized)
    normalized = REPEATED_UNDERSCORES.sub("_", normalized)
    return normalized.strip(" ._")


def _render_export(
    document: TranscriptDocument,
    *,
    timestamps: bool,
) -> str:
    """Render normalized metadata followed by plain or timestamped transcript text."""
    metadata = document.metadata
    lines = [f"Title: {metadata.title or metadata.video_id}"]
    if metadata.channel:
        lines.append(f"Author: {metadata.channel}")
    channel = metadata.channel_handle or metadata.channel_id
    if channel:
        lines.append(f"Channel: {channel}")
    if metadata.upload_date:
        lines.append(f"Published: {metadata.upload_date}")
    if metadata.duration_seconds is not None:
        lines.append(f"Duration: {_format_timestamp(metadata.duration_seconds)}")
    if metadata.view_count is not None:
        lines.append(f"Views: {metadata.view_count:,}")
    if metadata.like_count is not None:
        lines.append(f"Likes: {metadata.like_count:,}")
    lines.append(f"URL: {metadata.url}")
    language = TRANSCRIPT_LANGUAGE_LABELS.get(document.language, document.language)
    if language == "speech-to-text":
        language = None
    if language:
        lines.append(f"Language: {language}")
    source = TRANSCRIPT_SOURCE_LABELS.get(document.source, document.source)
    if source:
        lines.append(f"Transcript source: {source}")

    transcript = document.text
    if timestamps:
        transcript = "\n".join(
            f"[{_format_timestamp(chunk.start_seconds)}] {chunk.text}"
            for chunk in document.chunks
        )
    return "\n".join([*lines, "", transcript]).rstrip() + "\n"


def _format_timestamp(seconds: float) -> str:
    """Format a non-negative second offset as a stable HH:MM:SS timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
