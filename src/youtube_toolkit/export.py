"""Transcript exports named for the video's title and channel: readable text in the working directory, or Markdown in a given directory."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TypedDict
import unicodedata

from .schemas import TranscriptDocument, TranscriptSegment, TranscriptSource

TRANSCRIPT_SOURCE_LABELS: dict[TranscriptSource, str] = {
    "subtitles": "manual captions",
    "automatic_captions": "automatic captions",
    "speech-to-text": "speech-to-text",
}
UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_WHITESPACE = re.compile(r"\s+")
REPEATED_UNDERSCORES = re.compile(r"_+")


class TranscriptExportResult(TypedDict):
    video_id: str
    path: str


def write_text_export(
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
            export_file.write(_render_text(document, timestamps=timestamps))
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


def _export_filename(document: TranscriptDocument) -> str:
    """`TITLE_CHANNEL.txt`, `TITLE_CHANNEL_LANGUAGE.txt` for a translation, or `TITLE_CHANNEL_refined.txt` for a cleaned-up copy."""
    metadata = document.metadata
    title = _normalize_filename_component(metadata.title or "") or "transcript"
    channel = _normalize_filename_component(metadata.channel or "")
    rewrite = document.rewrite
    suffix = ""
    if rewrite is not None:
        suffix = (
            _normalize_filename_component(document.language)
            if rewrite.translated
            else "refined"
        )
    stem = "_".join(part for part in (title, channel, suffix) if part)
    return f"{stem}.txt"


def _normalize_filename_component(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = UNSAFE_FILENAME_CHARACTERS.sub("", normalized)
    normalized = FILENAME_WHITESPACE.sub("_", normalized)
    normalized = REPEATED_UNDERSCORES.sub("_", normalized)
    return normalized.strip(" ._")


def _render_text(
    document: TranscriptDocument,
    *,
    timestamps: bool,
) -> str:
    lines = [f"{name}: {value}" for name, value in _header_fields(document)]
    transcript = document.text
    if timestamps:
        transcript = "\n".join(_timed_line(segment) for segment in document.segments)
    return "\n".join([*lines, "", transcript]).rstrip() + "\n"


def _render_markdown(document: TranscriptDocument, *, timestamps: bool) -> str:
    (_, title), *fields = _header_fields(document)
    header = "\n".join(f"- {name}: {value}" for name, value in fields)
    # Timed lines are separate paragraphs, since Markdown joins adjacent lines into one.
    body = (
        [_timed_line(segment) for segment in document.segments]
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
    if (rewrite := document.rewrite) is not None:
        fields.append(
            ("Translation", f"{document.language} by {rewrite.model}")
            if rewrite.translated
            else ("Refined by", rewrite.model)
        )
    return fields


def _timed_line(segment: TranscriptSegment) -> str:
    return f"[{_format_timestamp(segment.start_seconds)}] {segment.text}"


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
