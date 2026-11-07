"""Transcript retrieval policy: prefer captions, fall back to cloud and then local speech-to-text when allowed and set up, convert Chinese text to Taiwan Traditional, archive each fresh document, and export readable text with overwrite protection, or as Markdown into a given directory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
import re
from typing import Any, Literal, Protocol, TypedDict
import unicodedata

from .artifacts import write_artifact
from .convert_zh import convert_zh
from .identity import VideoRef, require_video_id
from .schemas import (
    SpeechTranscription,
    TranscriptChunk,
    TranscriptDocument,
    TranscriptSource,
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
TRANSCRIPT_SOURCE_LABELS: dict[TranscriptSource, str] = {
    "subtitles": "manual captions",
    "automatic_captions": "automatic captions",
    "speech-to-text": "speech-to-text",
}
UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_WHITESPACE = re.compile(r"\s+")
REPEATED_UNDERSCORES = re.compile(r"_+")


class TranscriptClient(Protocol):
    def metadata(self, video_ref: VideoRef, /) -> dict[str, Any]: ...

    def download_transcription_audio(self, video_ref: VideoRef, /) -> Path:
        """The caller deletes the returned file."""
        ...


class TranscriptExportResult(TypedDict):
    video_id: str
    path: str


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
    document = replace(
        document,
        chunks=[
            replace(chunk, text=convert_zh(chunk.text)) for chunk in document.chunks
        ],
    )
    request = {
        "video_id": video_id,
        "languages": list(SUBTITLE_LANGUAGES),
        "stt": stt,
    }
    write_artifact("transcripts", video_id, request, asdict(document))
    return document


def write_transcript_export(
    document: TranscriptDocument,
    *,
    timestamps: bool,
    force: bool,
) -> TranscriptExportResult:
    path = Path.cwd() / _export_filename(document)
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


def write_markdown_export(
    document: TranscriptDocument, directory: Path, *, timestamps: bool
) -> TranscriptExportResult:
    path = directory / Path(_export_filename(document)).with_suffix(".md")
    path.write_text(_render_markdown(document, timestamps=timestamps), encoding="utf-8")
    return {"video_id": document.metadata.video_id, "path": str(path)}


def _stt_transcriber() -> Callable[[Path], SpeechTranscription] | None:
    if cloud_stt_configured():
        return transcribe_in_cloud
    if local_stt_installed():
        return transcribe_locally
    return None


def _fetch_stt_transcript(
    metadata_payload: dict[str, Any],
    *,
    client: TranscriptClient,
    transcribe: Callable[[Path], SpeechTranscription],
) -> TranscriptDocument:
    metadata = VideoMetadata.from_yt_dlp(metadata_payload)
    audio_path = client.download_transcription_audio(metadata.video_id)
    try:
        transcription = transcribe(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
    return TranscriptDocument(
        metadata=metadata,
        language=transcription.language
        or normalize_text(metadata_payload.get("language"))
        or "speech-to-text",
        source="speech-to-text",
        chunks=transcription.segments
        or [TranscriptChunk(start_seconds=0, end_seconds=0, text=transcription.text)],
    )


def _export_filename(document: TranscriptDocument) -> str:
    """`TITLE_CHANNEL.txt`, or `TITLE_CHANNEL_LANGUAGE.txt` for a translation."""
    metadata = document.metadata
    title = _normalize_filename_component(metadata.title or "") or "transcript"
    channel = _normalize_filename_component(metadata.channel or "")
    language = (
        _normalize_filename_component(document.language)
        if document.translated_by
        else ""
    )
    stem = "_".join(part for part in (title, channel, language) if part)
    return f"{stem}.txt"


def _normalize_filename_component(value: str) -> str:
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
    lines = [f"{name}: {value}" for name, value in _header_fields(document)]
    transcript = document.text
    if timestamps:
        transcript = "\n".join(_timed_line(chunk) for chunk in document.chunks)
    return "\n".join([*lines, "", transcript]).rstrip() + "\n"


def _render_markdown(document: TranscriptDocument, *, timestamps: bool) -> str:
    (_, title), *fields = _header_fields(document)
    header = "\n".join(f"- {name}: {value}" for name, value in fields)
    # Timed lines are separate paragraphs, since Markdown joins adjacent lines into one.
    body = (
        [_timed_line(chunk) for chunk in document.chunks]
        if timestamps
        else [document.text]
    )
    return "\n\n".join([f"# {title}", header, *body]) + "\n"


def _header_fields(document: TranscriptDocument) -> list[tuple[str, str]]:
    metadata = document.metadata
    fields = [("Title", metadata.title or metadata.video_id)]
    if metadata.channel:
        fields.append(("Author", metadata.channel))
    channel = metadata.channel_handle or metadata.channel_id
    if channel:
        fields.append(("Channel", channel))
    if metadata.upload_date:
        fields.append(("Published", metadata.upload_date))
    if metadata.duration_seconds is not None:
        fields.append(("Duration", _format_timestamp(metadata.duration_seconds)))
    fields.append(("URL", metadata.url))
    fields.append(("Transcript source", TRANSCRIPT_SOURCE_LABELS[document.source]))
    if document.translated_by:
        fields.append(
            ("Translation", f"{document.language} by {document.translated_by}")
        )
    return fields


def _timed_line(chunk: TranscriptChunk) -> str:
    return f"[{_format_timestamp(chunk.start_seconds)}] {chunk.text}"


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
