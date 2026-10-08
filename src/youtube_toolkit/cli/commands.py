"""CLI command adapters for directory-cached video, search, and channel retrieval."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any, TypedDict, cast
import unicodedata

from ..capabilities import require_capability
from ..config import DEFAULT_CHANNEL_SEARCH_LIMIT, DEFAULT_YOUTUBE_SEARCH_LIMIT
from ..schemas import TranscriptDocument, VideoMetadata
from ..search.request import ChannelTarget, SearchRequest, SearchSource
from ..transcript import fetch_transcript
from ..youtube.channels import ChannelOptions, fetch_channel_videos
from ..youtube.client import YtDlpClient


class TranscriptExportResult(TypedDict):
    """File created by an explicit transcript export request."""

    video_id: str
    path: str


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


class VideoCommands:
    """Handlers for retrieving video metadata, transcripts, and visual context."""

    @staticmethod
    def transcript(
        args: argparse.Namespace,
    ) -> str | list[dict[str, Any]] | TranscriptExportResult:
        """Return transcript data or export the same resolved document to the working directory."""
        if args.force and not args.export:
            raise ValueError("--force requires --export")
        document = fetch_transcript(args.video_ref)
        if args.export:
            return write_transcript_export(
                document,
                timestamps=args.timestamps,
                force=args.force,
            )
        if args.timestamps:
            return [asdict(chunk) for chunk in document.chunks]
        return document.text

    @staticmethod
    def metadata(args: argparse.Namespace) -> dict[str, Any]:
        """Return normalized video facts without exposing the raw yt-dlp payload."""
        return asdict(VideoMetadata.from_yt_dlp(YtDlpClient().metadata(args.video_ref)))

    @staticmethod
    def vision(args: argparse.Namespace) -> dict[str, Any]:
        """Return bounded timestamped visual context and its local collage paths."""
        require_capability("vision", "vision")
        from ..vision import sample_vision

        return sample_vision(args.video_ref, max_frames=args.max_frames)


def write_transcript_export(
    document: TranscriptDocument,
    *,
    timestamps: bool,
    force: bool,
) -> TranscriptExportResult:
    """Write one deterministic UTF-8 transcript file without silently replacing an existing export."""
    path = Path.cwd() / transcript_export_filename(document.metadata)
    try:
        with path.open(
            "w" if force else "x", encoding="utf-8", newline="\n"
        ) as export_file:
            export_file.write(render_transcript_export(document, timestamps=timestamps))
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


def transcript_export_filename(metadata: VideoMetadata) -> str:
    """Build a portable title-and-channel filename without exposing the video ID."""
    title = normalize_filename_component(metadata.title or "") or "transcript"
    channel = normalize_filename_component(metadata.channel or "")
    stem = "_".join(part for part in (title, channel) if part)
    return f"{stem}.txt"


def normalize_filename_component(value: str) -> str:
    """Normalize human-readable metadata for a portable filename component."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = UNSAFE_FILENAME_CHARACTERS.sub("", normalized)
    normalized = FILENAME_WHITESPACE.sub("_", normalized)
    normalized = REPEATED_UNDERSCORES.sub("_", normalized)
    return normalized.strip(" ._")


def render_transcript_export(
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
        lines.append(f"Duration: {format_timestamp(metadata.duration_seconds)}")
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
            f"[{format_timestamp(chunk.start_seconds)}] {chunk.text}"
            for chunk in document.chunks
        )
    return "\n".join([*lines, "", transcript]).rstrip() + "\n"


def format_timestamp(seconds: float) -> str:
    """Format a non-negative second offset as a stable HH:MM:SS timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


class ChannelCommands:
    """Handlers for direct channel listing commands."""

    @staticmethod
    def videos(args: argparse.Namespace) -> dict[str, Any]:
        return fetch_channel_videos(
            args.channel_ref, options=ChannelOptions(last=args.last, days=args.days)
        )


class SearchCommands:
    """Handlers for the public mixed-source search command."""

    @staticmethod
    def search(args: argparse.Namespace) -> list[dict[str, Any]]:
        """Translate parsed CLI arguments into the public search request contract."""
        from ..search.workflow import search

        raw_sources = tuple(dict.fromkeys(args.source or ["youtube"]))
        limit = args.limit
        if limit is None:
            limit = (
                DEFAULT_CHANNEL_SEARCH_LIMIT
                if raw_sources == ("channel",)
                else DEFAULT_YOUTUBE_SEARCH_LIMIT
            )
        sources = cast(tuple[SearchSource, ...], raw_sources)
        return search(
            SearchRequest(
                query=args.query,
                extra_queries=tuple(getattr(args, "extra_queries", [])),
                sources=sources,
                mode=args.mode,
                return_type=args.return_type,
                channels=tuple(
                    ChannelTarget.from_ref(channel) for channel in args.channels
                ),
                limit=limit,
                sort=args.sort,
                min_views=args.min_views,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                uploaded_after=getattr(args, "uploaded_after", None),
                uploaded_within_days=getattr(args, "uploaded_within_days", None),
                include_channel=tuple(args.include_channel),
                exclude_channel=tuple(args.exclude_channel),
                include_title=tuple(args.include_title),
                exclude_title=tuple(args.exclude_title),
            )
        )
