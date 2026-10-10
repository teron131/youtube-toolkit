"""External YouTube payloads are normalized here before search and transcript workflows depend on them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .convert_zh import convert_zh
from .identity import extract_channel_identity


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Normalized metadata shape for a YouTube video for search and transcript documents."""

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
        """Translate yt-dlp video metadata into the toolkit's normalized video metadata contract."""
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
    """Source chapter titles and time boundaries remain available independently of transcript text."""

    title: str
    start_seconds: float
    end_seconds: float | None

    @classmethod
    def from_yt_dlp(cls, chapter: dict[str, Any]) -> VideoChapter:
        """Normalize source chapter times while retaining chapters without a known end."""
        return cls(
            title=str(chapter.get("title") or "Untitled chapter"),
            start_seconds=float(chapter.get("start_time") or 0),
            end_seconds=float(chapter["end_time"])
            if chapter.get("end_time") is not None
            else None,
        )


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """Timed transcript text span used for document rendering and semantic chunking."""

    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptDocument:
    """Transcript document boundary returned by caption and STT workflows."""

    metadata: VideoMetadata
    language: str
    source: str
    chunks: list[TranscriptChunk]

    @property
    def text(self) -> str:
        return " ".join(chunk.text for chunk in self.chunks)


def normalize_upload_date(upload_date: Any) -> str | None:
    """Normalize yt-dlp YYYYMMDD dates to ISO dates while preserving already textual dates."""
    if not isinstance(upload_date, str):
        return None
    stripped_upload_date = upload_date.strip()
    if len(stripped_upload_date) != 8 or not stripped_upload_date.isdigit():
        return stripped_upload_date or None
    return f"{stripped_upload_date[:4]}-{stripped_upload_date[4:6]}-{stripped_upload_date[6:8]}"


def normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    return stripped_value or None


def normalize_title(value: Any) -> str | None:
    """Normalize title text and convert Chinese script to the toolkit's Taiwan Traditional policy."""
    normalized_text = normalize_text(value)
    if normalized_text is None:
        return None
    return convert_zh(normalized_text)


def normalize_count(count: Any) -> int | None:
    """Accept integer counters from external metadata while rejecting booleans."""
    if isinstance(count, bool):
        return None
    if isinstance(count, int):
        return count
    return None


def normalize_number(value: Any) -> float | None:
    """Accept numeric metadata values as floats while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
