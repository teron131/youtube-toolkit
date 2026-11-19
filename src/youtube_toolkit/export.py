"""Files a video gives its user, named for its title and channel: transcript exports, as readable text in the working directory or Markdown in a given directory, and the video or its sound, whole or one time range, downloaded to the working directory."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import re
import shutil
from typing import Literal, TypedDict
import unicodedata

from .schemas import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptSource,
    VideoMetadata,
)
from .youtube import YtDlpClient

type MediaKind = Literal["video", "audio"]
type DownloadQuality = Literal["360p", "480p", "720p", "1080p"]

TRANSCRIPT_SOURCE_LABELS: dict[TranscriptSource, str] = {
    "subtitles": "manual captions",
    "automatic_captions": "automatic captions",
    "speech-to-text": "speech-to-text",
}
UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
FILENAME_WHITESPACE = re.compile(r"\s+")
REPEATED_UNDERSCORES = re.compile(r"_+")
# YouTube's own stream heights; 4K is left out because its files run to gigabytes for little an agent can use.
DOWNLOAD_HEIGHTS: dict[DownloadQuality, int] = {
    "360p": 360,
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}
DEFAULT_DOWNLOAD_QUALITY: DownloadQuality = "1080p"
# A download counts as stuck after twice its span's playing time, and never sooner than vision's ten minutes: at the rate vision's download is sized for, a two-hour 720p video in ten minutes, even 1080p with sound arrives in well under half its playing time.
MIN_DOWNLOAD_TIMEOUT_SECONDS = 10 * 60


class TranscriptExportResult(TypedDict):
    video_id: str
    path: str


class DownloadResult(TypedDict):
    path: str
    message: str | None


def write_text_export(
    document: TranscriptDocument,
    *,
    timestamps: bool,
    fresh: bool,
) -> TranscriptExportResult:
    path = Path.cwd() / _export_filename(document)
    try:
        with path.open(
            "w" if fresh else "x", encoding="utf-8", newline="\n"
        ) as export_file:
            export_file.write(_render_text(document, timestamps=timestamps))
    except FileExistsError as error:
        raise ValueError(
            f"Transcript export already exists: {path}; rerun with --fresh to replace it"
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


def download_media(
    metadata: VideoMetadata,
    kind: MediaKind,
    *,
    quality: DownloadQuality,
    start: float | None,
    end: float | None,
    fresh: bool,
    client: YtDlpClient,
) -> DownloadResult:
    """`start` and `end` in seconds keep only that span, which opens at the keyframe at or before `start`, so a video can begin a few seconds early; an `end` past the video stops at its end.

    The file name holds the kind, the video's quality, and the range, so a file already there is that same download and comes back without downloading, with a message saying so, unless `fresh` downloads it again.
    The quality names the requested cap, so a video smaller than it is downloaded again under each larger cap.
    A failed download can leave yt-dlp's .part files beside the target, which a rerun resumes, or for a range overwrites.
    """
    duration = metadata.duration_seconds
    if duration is None:
        raise ValueError(
            f"Missing duration for {metadata.video_id}; a live stream has none until it ends"
        )
    first = 0.0 if start is None else start
    last = duration if end is None else min(end, duration)
    if not 0 <= first < last:
        raise ValueError(
            "A download range must start before it ends and before the video ends"
        )
    section = None if start is None and end is None else (first, last)
    labels: list[str] = [] if kind == "audio" else [quality]
    if section is not None:
        labels.append(f"{_time_label(first)}_{_time_label(last)}")
    extension = "m4a" if kind == "audio" else "mp4"
    path = Path.cwd() / f"{_filename_stem(metadata, '_'.join(labels))}.{extension}"
    if path.exists() and not fresh:
        return {
            "path": str(path),
            "message": "Already downloaded, so this is the existing file; rerun with --fresh to replace it with a fresh download",
        }
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Downloads need ffmpeg, which yt-dlp uses to join the picture with its sound and to cut time ranges"
        )
    client.download_media(
        metadata.video_id,
        path,
        height=None if kind == "audio" else DOWNLOAD_HEIGHTS[quality],
        section=section,
        overwrite=fresh,
        timeout_seconds=max(MIN_DOWNLOAD_TIMEOUT_SECONDS, ceil(2 * (last - first))),
    )
    return {"path": str(path), "message": None}


def _export_filename(document: TranscriptDocument) -> str:
    """`TITLE_CHANNEL.txt`, `TITLE_CHANNEL_LANGUAGE.txt` for a translation, or `TITLE_CHANNEL_refined.txt` for a cleaned-up copy."""
    rewrite = document.rewrite
    label = ""
    if rewrite is not None:
        label = document.language if rewrite.translated else "refined"
    return f"{_filename_stem(document.metadata, label)}.txt"


def _filename_stem(metadata: VideoMetadata, label: str) -> str:
    """`TITLE_CHANNEL_LABEL`, leaving out empty parts; a video without a title goes by its ID."""
    parts = (
        _normalize_filename_component(metadata.title or "") or metadata.video_id,
        _normalize_filename_component(metadata.channel or ""),
        _normalize_filename_component(label),
    )
    return "_".join(part for part in parts if part)


def _normalize_filename_component(value: str) -> str:
    """Characters a filename cannot hold are removed and each run of whitespace becomes one underscore: `  TITLE: "SUBTITLE"  PART? ` → `TITLE_SUBTITLE_PART`."""
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
    if document.stt_model is not None:
        fields.append(("Transcribed by", document.stt_model))
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
    """Always HH:MM:SS, rounded down: 3725.9 → `01:02:05`."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _time_label(seconds: float) -> str:
    """Whole seconds as minutes and seconds, with hours only from an hour on: 5 → `0m5s`, 3725.9 → `1h2m5s`."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, rest = divmod(remainder, 60)
    return f"{hours}h{minutes}m{rest}s" if hours else f"{minutes}m{rest}s"
