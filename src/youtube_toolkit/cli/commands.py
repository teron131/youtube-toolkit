"""CLI command adapters for video, search, storage, and watchlist workflows."""

from __future__ import annotations

import argparse
import builtins
from dataclasses import asdict
from pathlib import Path
import re
import sys
from typing import Any, TypedDict, cast
import unicodedata

from ..identity import require_video_id
from ..schemas import ChannelRemovalResult, EmbeddingIndexResult, TranscriptDocument, VideoMetadata, WatchlistSyncResult
from ..search import ChannelTarget, SearchRequest, search as run_search_request
from ..search.request import SearchSource
from ..storage import (
    DATABASE_FILENAME,
    get_connection,
    index_video_embedding,
    list_video_ids_for_embedding_index,
    resolve_storage_dir,
    upsert_video,
)
from ..storage.videos import delete_video_record, edit_video_content, get_video_record
from ..storage.viewer import serve_database_viewer
from ..watchlist_schedule import (
    ScheduledSyncResult,
    WatchlistScheduleInstallResult,
    WatchlistScheduleOptions,
    WatchlistScheduleStatusResult,
    WatchlistScheduleUninstallResult,
    install_watchlist_schedule,
    scheduled_sync_watchlist,
    uninstall_watchlist_schedule,
    watchlist_schedule_status,
)
from ..youtube import YtDlpClient
from ..youtube.caption import fetch_transcript
from ..youtube.channels import (
    ChannelOptions,
    add_blacklist_channels,
    add_watchlist_channels,
    fetch_channel_videos,
    list_recent_watchlist_videos,
    list_watchlist_channels,
    remove_blacklist_channels,
    remove_watchlist_channels,
    sync_watchlist,
)


class IndexCommandResult(TypedDict):
    """Embedding index command result split into indexed, skipped, and failed videos."""

    videos: int
    indexed: list[EmbeddingIndexResult]
    skipped: list[EmbeddingIndexResult]
    failed: list[EmbeddingIndexResult]


class TranscriptExportResult(TypedDict):
    """File created by an explicit transcript export request."""

    video_id: str
    path: str


DEFAULT_YOUTUBE_SEARCH_LIMIT = 10
DEFAULT_CHANNEL_SEARCH_LIMIT = 30
DATABASE_REFRESH_METADATA_FIELDS = (
    "channel_id",
    "channel_handle",
    "title",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
)

VIDEO_METADATA_OUTPUT_FIELDS = (
    "video_id",
    "url",
    "title",
    "channel",
    "channel_id",
    "channel_handle",
    "upload_date",
    "duration_seconds",
    "view_count",
    "like_count",
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


class DatabaseCommands:
    """Handlers for inspecting and administering the active database."""

    @staticmethod
    def view(args: argparse.Namespace) -> None:
        """Serve the active database until the user stops the viewer."""
        serve_database_viewer(
            resolve_storage_dir() / DATABASE_FILENAME,
            port=args.port,
            open_browser=not args.no_open,
        )

    @staticmethod
    def get(args: argparse.Namespace) -> dict[str, Any]:
        """Return one complete locally stored video record."""
        video_id = require_video_id(args.video_ref)
        with get_connection() as connection:
            return get_video_record(connection, video_id)

    @staticmethod
    def refresh(args: argparse.Namespace) -> dict[str, Any]:
        """Refresh provider-owned metadata through yt-dlp and report the stored facts."""
        requested_video_id = require_video_id(args.video_ref)
        metadata = YtDlpClient().metadata(args.video_ref)
        fetched_video_id = require_video_id(str(metadata.get("id") or ""))
        if fetched_video_id != requested_video_id:
            raise ValueError(f"yt-dlp returned video {fetched_video_id} for requested video {requested_video_id}")
        with get_connection() as connection:
            stored_video = upsert_video(connection, metadata)
        if stored_video is None:
            raise RuntimeError(f"Could not store refreshed metadata for video {requested_video_id}")
        return {
            "video_id": requested_video_id,
            "metadata": {field: stored_video[field] for field in DATABASE_REFRESH_METADATA_FIELDS},
            "metadata_fetched_at": stored_video["metadata_fetched_at"],
        }

    @staticmethod
    def edit(args: argparse.Namespace) -> dict[str, Any]:
        """Translate explicit agent-owned content fields into one validated edit."""
        video_id = require_video_id(args.video_ref)
        values = {}
        if args.summary is not None:
            values["summary"] = args.summary
        if args.transcript_file is not None:
            values["transcript"] = read_text_input(args.transcript_file)
        clear_fields = [field.replace("-", "_") for field in args.clear]
        with get_connection() as connection:
            return edit_video_content(
                connection,
                video_id,
                values,
                clear_fields=clear_fields,
                dry_run=args.dry_run,
            )

    @staticmethod
    def delete(args: argparse.Namespace) -> dict[str, Any]:
        """Require explicit confirmation before cascading one stored video deletion."""
        video_id = require_video_id(args.video_ref)
        if not args.dry_run and not args.yes:
            raise ValueError("Video deletion requires --yes; use --dry-run to inspect it first")
        with get_connection() as connection:
            return delete_video_record(
                connection,
                video_id,
                dry_run=args.dry_run,
            )


class VideoCommands:
    """Handlers for retrieving video metadata and transcripts."""

    @staticmethod
    def transcript(args: argparse.Namespace) -> str | list[dict[str, Any]] | TranscriptExportResult:
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
        metadata = asdict(VideoMetadata.from_yt_dlp(YtDlpClient().metadata(args.video_ref)))
        return {field: metadata[field] for field in VIDEO_METADATA_OUTPUT_FIELDS}


def write_transcript_export(
    document: TranscriptDocument,
    *,
    timestamps: bool,
    force: bool,
) -> TranscriptExportResult:
    """Write one deterministic UTF-8 transcript file without silently replacing an existing export."""
    path = Path.cwd() / transcript_export_filename(document.metadata)
    try:
        with path.open("w" if force else "x", encoding="utf-8", newline="\n") as export_file:
            export_file.write(render_transcript_export(document, timestamps=timestamps))
    except FileExistsError as error:
        raise ValueError(f"Transcript export already exists: {path}; use --force to replace it") from error
    except OSError as error:
        raise RuntimeError(f"Could not write transcript export {path}: {error}") from error
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
    if language in {"stored", "speech-to-text"}:
        language = None
    if language:
        lines.append(f"Language: {language}")
    source = TRANSCRIPT_SOURCE_LABELS.get(document.source, document.source)
    if source:
        lines.append(f"Transcript source: {source}")

    transcript = document.text
    if timestamps:
        transcript = "\n".join(f"[{format_timestamp(chunk.start_seconds)}] {chunk.text}" for chunk in document.chunks)
    return "\n".join([*lines, "", transcript]).rstrip() + "\n"


def format_timestamp(seconds: float) -> str:
    """Format a non-negative second offset as a stable HH:MM:SS timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def read_text_input(path: str) -> str:
    """Read UTF-8 command text from a file or standard input while preserving the original failure cause."""
    try:
        if path == "-":
            return sys.stdin.read()
        return Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Could not read text input {path}: {error}") from error


class ChannelCommands:
    """Handlers for direct channel listing commands."""

    @staticmethod
    def videos(args: argparse.Namespace) -> dict[str, Any]:
        return fetch_channel_videos(args.channel_ref, options=channel_options(args))


class SearchCommands:
    """Handlers for the public mixed-source search command."""

    @staticmethod
    def search(args: argparse.Namespace) -> list[dict[str, Any]]:
        """Translate parsed CLI arguments into the public search request contract."""
        raw_sources = tuple(dict.fromkeys(args.source or ["youtube"]))
        limit = args.limit
        if limit is None:
            limit = DEFAULT_CHANNEL_SEARCH_LIMIT if raw_sources == ("channel",) else DEFAULT_YOUTUBE_SEARCH_LIMIT
        sources = cast(tuple[SearchSource, ...], raw_sources)
        return run_search_request(
            SearchRequest(
                query=args.query,
                extra_queries=tuple(getattr(args, "extra_queries", [])),
                sources=sources,
                mode=args.mode,
                return_type=args.return_type,
                channels=tuple(ChannelTarget.from_ref(channel) for channel in args.channels),
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


class WatchlistCommands:
    """Handlers for watchlist membership, sync, recency, and scheduler commands."""

    @staticmethod
    def add(args: argparse.Namespace) -> builtins.list[str]:
        return add_watchlist_channels(args.channel_refs)

    @staticmethod
    def list(args: argparse.Namespace) -> builtins.list[str]:
        del args
        return list_watchlist_channels()

    @staticmethod
    def remove(args: argparse.Namespace) -> builtins.list[ChannelRemovalResult]:
        """Remove watchlist membership without deleting catalog data."""
        return remove_watchlist_channels(args.channel_refs)

    @staticmethod
    def sync(args: argparse.Namespace) -> WatchlistSyncResult:
        return sync_watchlist(options=channel_options(args))

    @staticmethod
    def scheduled_sync(args: argparse.Namespace) -> ScheduledSyncResult:
        """The hidden scheduler entrypoint uses scheduled default arguments."""
        return scheduled_sync_watchlist(options=channel_options(args))

    @staticmethod
    def recent(args: argparse.Namespace) -> builtins.list[dict[str, Any]]:
        """Recent watchlist output is storage-only and does not mutate catalog state."""
        return list_recent_watchlist_videos(
            channel_refs=args.channel_refs,
            days=args.days,
        )

    @staticmethod
    def install_schedule(args: argparse.Namespace) -> WatchlistScheduleInstallResult:
        """Install the LaunchAgent that runs hidden watchlist scheduled-sync."""
        return install_watchlist_schedule(
            options=WatchlistScheduleOptions(
                minutes=args.minutes,
                last=args.last,
                days=args.days,
                label=args.label,
                load=not args.no_load,
            )
        )

    @staticmethod
    def schedule_status(args: argparse.Namespace) -> WatchlistScheduleStatusResult:
        """Inspect the LaunchAgent state for the configured watchlist schedule label."""
        return watchlist_schedule_status(label=args.label)

    @staticmethod
    def uninstall_schedule(args: argparse.Namespace) -> WatchlistScheduleUninstallResult:
        """Unload and remove the LaunchAgent for the configured watchlist schedule label."""
        return uninstall_watchlist_schedule(label=args.label)


class BlacklistCommands:
    """Handlers for search blacklist membership commands."""

    @staticmethod
    def add(args: argparse.Namespace) -> list[str]:
        return add_blacklist_channels(args.channel_refs)

    @staticmethod
    def remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
        """Remove blacklist membership without deleting catalog data."""
        return remove_blacklist_channels(args.channel_refs)


class IndexCommands:
    """Handlers for embedding index maintenance commands."""

    @staticmethod
    def videos(args: argparse.Namespace) -> IndexCommandResult:
        """Index explicit videos or all eligible stored videos when no refs are provided."""
        video_ids = [require_video_id(video_ref) for video_ref in args.video_refs]
        indexed: list[EmbeddingIndexResult] = []
        skipped: list[EmbeddingIndexResult] = []
        failed: list[EmbeddingIndexResult] = []

        if not video_ids:
            with get_connection() as connection:
                video_ids = list_video_ids_for_embedding_index(
                    connection,
                    require_transcript=not args.include_without_transcript,
                )

        for video_id in video_ids:
            with get_connection() as connection:
                result = index_one_video(connection, video_id)
            if result["status"] == "indexed":
                indexed.append(result)
            elif result["status"] == "skipped":
                skipped.append(result)
            else:
                failed.append(result)

        return {
            "videos": len(video_ids),
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
        }


def index_one_video(
    connection: Any,
    video_id: str,
) -> EmbeddingIndexResult:
    """Index one stored video and convert storage outcomes into command result rows."""
    try:
        payload = index_video_embedding(connection, video_id)
    except Exception as error:
        return {
            "status": "failed",
            "video_id": video_id,
            "error": str(error),
        }
    if payload is None:
        return {
            "status": "skipped",
            "video_id": video_id,
            "detail": "Video is not stored",
        }
    return {
        "status": "indexed",
        "video_id": payload["video_id"],
        "dimensions": payload["dimensions"],
    }


def channel_options(args: argparse.Namespace) -> ChannelOptions:
    """Translate shared CLI window flags into channel workflow options."""
    return ChannelOptions(last=args.last, days=args.days)
