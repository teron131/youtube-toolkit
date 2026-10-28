"""External YouTube payloads are normalized here before search and transcript workflows depend on them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from .convert_zh import convert_zh
from .identity import extract_channel_identity

TranscriptSource = Literal["subtitles", "automatic_captions", "speech-to-text"]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    channel_handle: str | None = None
    upload_date: str | None = None
    duration_seconds: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    description: str | None = None
    thumbnail: str | None = None
    chapters: list[VideoChapter] = field(default_factory=list)

    @classmethod
    def from_yt_dlp(cls, metadata: dict[str, Any]) -> VideoMetadata:
        video_id = str(metadata.get("id") or "").strip()
        if not video_id:
            raise ValueError("yt-dlp metadata did not include a video id")

        channel_identity = extract_channel_identity(metadata)
        return cls(
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=normalize_title(metadata.get("title")),
            channel=normalize_text(metadata.get("uploader") or metadata.get("channel")),
            channel_id=channel_identity.youtube_channel_id,
            channel_handle=channel_identity.channel_handle,
            upload_date=normalize_upload_date(metadata.get("upload_date")),
            duration_seconds=normalize_number(metadata.get("duration")),
            view_count=normalize_count(metadata.get("view_count")),
            like_count=normalize_count(metadata.get("like_count")),
            description=normalize_text(metadata.get("description")),
            thumbnail=normalize_text(metadata.get("thumbnail")),
            chapters=[
                VideoChapter.from_yt_dlp(chapter)
                for chapter in metadata.get("chapters") or []
            ],
        )


@dataclass(frozen=True, slots=True)
class VideoChapter:
    title: str
    start_seconds: float
    end_seconds: float | None

    @classmethod
    def from_yt_dlp(cls, chapter: dict[str, Any]) -> VideoChapter:
        return cls(
            title=str(chapter.get("title") or "Untitled chapter"),
            start_seconds=float(chapter.get("start_time") or 0),
            end_seconds=float(chapter["end_time"])
            if chapter.get("end_time") is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class SpeechTranscription:
    """`segments` is empty when the route reports no timing, and `language` is None when it reports no detected language."""

    text: str
    segments: list[TranscriptChunk]
    language: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptDocument:
    """A speech-to-text document holds the route's timed segments, or one untimed chunk at 0 seconds when the route reports no timing; its `language` is `speech-to-text` when no language was detected or reported."""

    metadata: VideoMetadata
    language: str
    source: TranscriptSource
    chunks: list[TranscriptChunk]

    @property
    def text(self) -> str:
        return " ".join(chunk.text for chunk in self.chunks)


def parse_upload_date(value: Any) -> date | None:
    """yt-dlp reports YYYYMMDD; ISO dates parse too, and anything else is None."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def normalize_upload_date(value: Any) -> str | None:
    upload_date = parse_upload_date(value)
    return upload_date.isoformat() if upload_date is not None else None


def normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    return stripped_value or None


def normalize_title(value: Any) -> str | None:
    normalized_text = normalize_text(value)
    if normalized_text is None:
        return None
    return convert_zh(normalized_text)


def normalize_count(count: Any) -> int | None:
    if isinstance(count, bool):
        return None
    if isinstance(count, int):
        return count
    return None


def normalize_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
