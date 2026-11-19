"""Reading copies of transcripts and summaries: the video's facts as a Markdown header over the text, laid out so line-based tools such as grep and sed find their way through it."""

from __future__ import annotations

from typing import Literal

from .schemas import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptSource,
    VideoMetadata,
)
from .sentences import split_sentences

TRANSCRIPT_SOURCE_LABELS: dict[TranscriptSource, str] = {
    "subtitles": "manual captions",
    "automatic_captions": "automatic captions",
    "speech-to-text": "speech-to-text",
}

# "timed" gives each caption line its start time, "sentences" gives each sentence its own line, and "article" keeps a rewrite's paragraphs and chapter headings.
type TranscriptLayout = Literal["timed", "sentences", "article"]


def transcript_markdown(document: TranscriptDocument, layout: TranscriptLayout) -> str:
    fields = [
        *_video_fields(document.metadata),
        ("Transcript source", TRANSCRIPT_SOURCE_LABELS[document.source]),
    ]
    if document.stt_model is not None:
        fields.append(("Transcribed by", document.stt_model))
    if (rewrite := document.rewrite) is not None:
        fields.append(
            ("Translation", f"{document.language} by {rewrite.model}")
            if rewrite.translated
            else ("Refined by", rewrite.model)
        )
    match layout:
        case "timed":
            # Timed lines are separate paragraphs, since Markdown joins adjacent lines into one.
            body = "\n\n".join(_timed_line(segment) for segment in document.segments)
        case "sentences":
            body = "\n".join(
                sentence.text for sentence in split_sentences(document.segments)
            )
        case "article":
            body = document.text
    return _page(document.metadata, fields, body)


def summary_markdown(metadata: VideoMetadata, summary: str, *, model: str) -> str:
    return _page(
        metadata, [*_video_fields(metadata), ("Summarized by", model)], summary
    )


def _page(metadata: VideoMetadata, fields: list[tuple[str, str]], body: str) -> str:
    header = "\n".join(f"- {name}: {value}" for name, value in fields)
    return f"# {metadata.title or metadata.video_id}\n\n{header}\n\n{body}\n"


def _video_fields(metadata: VideoMetadata) -> list[tuple[str, str]]:
    fields = []
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
    return fields


def _timed_line(segment: TranscriptSegment) -> str:
    return f"[{_format_timestamp(segment.start_seconds)}] {segment.text}"


def _format_timestamp(seconds: float) -> str:
    """Always HH:MM:SS, rounded down: 3725.9 → `01:02:05`."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
