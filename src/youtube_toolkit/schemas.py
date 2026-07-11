"""External YouTube payloads are normalized here before storage, search, or transcript workflows depend on them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict
from urllib.parse import urlparse

from .convert_zh import convert_zh


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Normalized metadata shape for a YouTube video before it enters storage or transcript documents."""

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

    @classmethod
    def from_yt_dlp(cls, metadata: dict[str, Any]) -> VideoMetadata:
        """Translate yt-dlp video metadata into the toolkit's normalized video metadata contract."""
        video_id = str(metadata.get("id") or "").strip()
        if not video_id:
            raise ValueError("yt-dlp metadata did not include a video id")

        channel_handle = normalize_handle(metadata.get("channel_handle") or metadata.get("uploader_id") or metadata.get("channel_url") or metadata.get("uploader_url"))
        return cls(
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=normalize_title(metadata.get("title")),
            channel=normalize_text(metadata.get("uploader") or metadata.get("channel")),
            channel_id=normalize_channel_id(metadata.get("channel_id")),
            channel_handle=channel_handle,
            upload_date=normalize_upload_date(metadata.get("upload_date")),
            duration_seconds=normalize_number(metadata.get("duration")),
            view_count=normalize_view_count(metadata.get("view_count")),
            like_count=normalize_view_count(metadata.get("like_count")),
            description=normalize_text(metadata.get("description")),
            thumbnail=normalize_text(metadata.get("thumbnail")),
        )


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """Timed transcript text span used for document rendering and semantic chunking."""

    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptSection:
    """Chapter-oriented transcript section derived from caption chapters or flat transcript text."""

    title: str
    start_seconds: float | None
    end_seconds: float | None
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptDocument:
    """Transcript document boundary returned by caption and STT workflows."""

    metadata: VideoMetadata
    language: str
    source: str
    chunks: list[TranscriptChunk]
    sections: list[TranscriptSection]

    @property
    def text(self) -> str:
        return " ".join(chunk.text for chunk in self.chunks)


@dataclass(frozen=True, slots=True)
class ChannelIdentity:
    """Storage-facing channel identity that may carry a durable ID, a handle, or both."""

    youtube_channel_id: str | None
    channel_handle: str | None

    def has_identity(self) -> bool:
        return self.youtube_channel_id is not None or self.channel_handle is not None


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """Sparse storage update shape extracted from external video metadata."""

    youtube_video_id: str
    title: str | None
    upload_date: str | None
    duration: float | None
    view_count: int | None
    like_count: int | None
    transcript: str | None


class ChannelRemovalResult(TypedDict):
    """Public result shape for channel membership removal commands."""

    channel: str
    removed: bool


class EmbeddingIndexResult(TypedDict):
    """Public per-video result shape for embedding indexing attempts."""

    status: Literal["indexed", "skipped", "failed"]
    video_id: str
    dimensions: NotRequired[int]
    detail: NotRequired[str]
    error: NotRequired[str]


class TranscriptFetchResult(TypedDict):
    """Public per-video result shape for watchlist transcript fetch attempts."""

    status: Literal["stored", "failed"]
    video_id: str
    embedding: NotRequired[EmbeddingIndexResult]
    error: NotRequired[str]


class WatchlistSyncResult(TypedDict):
    """Public result shape for one watchlist sync run."""

    channels: list[str]
    new_videos: list[str]
    transcripts: list[TranscriptFetchResult]
    embeddings: list[EmbeddingIndexResult]


def extract_channel_identity(
    metadata: dict[str, Any],
    *,
    fallback_channel_id: str | None = None,
    fallback_channel_handle: str | None = None,
) -> ChannelIdentity:
    """Extract the best durable channel identity from metadata with explicit fallback ownership."""
    youtube_channel_id = normalize_channel_id(metadata.get("channel_id")) or fallback_channel_id
    handle_candidates = (
        metadata.get("channel_handle"),
        metadata.get("uploader_id"),
        metadata.get("channel_url"),
        metadata.get("uploader_url"),
        metadata.get("webpage_url"),
        fallback_channel_handle,
    )
    channel_handle = next(
        (handle for handle in (normalize_handle(candidate) for candidate in handle_candidates) if handle),
        None,
    )
    return ChannelIdentity(
        youtube_channel_id=youtube_channel_id,
        channel_handle=channel_handle,
    )


def extract_video_record(
    metadata: dict[str, Any],
    *,
    transcript: str | None = None,
) -> VideoRecord | None:
    """Extract a sparse video record suitable for upsert, returning None when no video ID exists."""
    youtube_video_id = normalize_text(metadata.get("id"))
    if youtube_video_id is None:
        return None
    return VideoRecord(
        youtube_video_id=youtube_video_id,
        title=normalize_title(metadata.get("title")),
        upload_date=normalize_upload_date(metadata.get("upload_date")),
        duration=normalize_number(metadata.get("duration")),
        view_count=normalize_view_count(metadata.get("view_count")),
        like_count=normalize_view_count(metadata.get("like_count")),
        transcript=transcript,
    )


def normalize_upload_date(upload_date: Any) -> str | None:
    """Normalize yt-dlp YYYYMMDD dates to ISO dates while preserving already textual dates."""
    if not isinstance(upload_date, str):
        return None
    stripped_upload_date = upload_date.strip()
    if len(stripped_upload_date) != 8 or not stripped_upload_date.isdigit():
        return stripped_upload_date or None
    return f"{stripped_upload_date[:4]}-{stripped_upload_date[4:6]}-{stripped_upload_date[6:8]}"


def normalize_handle(handle: Any) -> str | None:
    """Normalize channel handles from handles, usernames, or channel URLs while rejecting IDs."""
    if not isinstance(handle, str):
        return None
    stripped_handle = handle.strip()
    if not stripped_handle:
        return None
    if stripped_handle.startswith(("http://", "https://")):
        return extract_handle_from_url(stripped_handle)
    if stripped_handle.startswith("@"):
        return stripped_handle
    if stripped_handle.startswith("UC") or "/" in stripped_handle or " " in stripped_handle:
        return None
    return f"@{stripped_handle}"


def normalize_channel_id(channel_id: Any) -> str | None:
    """Store channel IDs as caller-provided text after stripping empty external values."""
    if not isinstance(channel_id, str):
        return None
    stripped_channel_id = channel_id.strip()
    return stripped_channel_id or None


def normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    return stripped_value or None


def normalize_title(value: Any) -> str | None:
    """Normalize title text and convert Chinese script to the toolkit's Hong Kong Traditional policy."""
    normalized_text = normalize_text(value)
    if normalized_text is None:
        return None
    return convert_zh(normalized_text, "s2hk")


def normalize_view_count(view_count: Any) -> int | None:
    """Accept integer counters from external metadata while rejecting booleans."""
    if isinstance(view_count, bool):
        return None
    if isinstance(view_count, int):
        return view_count
    return None


def normalize_number(value: Any) -> float | None:
    """Accept numeric metadata values as floats while rejecting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def extract_handle_from_url(url: str) -> str | None:
    parsed_url = urlparse(url)
    for path_part in parsed_url.path.split("/"):
        if path_part.startswith("@"):
            return path_part
    return None
