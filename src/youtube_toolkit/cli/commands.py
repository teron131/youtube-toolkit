"""CLI command handlers: each translates parsed arguments into one retrieval workflow and returns its JSON payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

from ..config import DEFAULT_CHANNEL_SEARCH_LIMIT, DEFAULT_YOUTUBE_SEARCH_LIMIT
from ..schemas import VideoMetadata
from ..search.preprocessing import ChunkResult, VideoResult
from ..search.request import ChannelTarget, SearchRequest
from ..transcript import (
    TranscriptExportResult,
    fetch_transcript,
    write_transcript_export,
)
from ..youtube.channels import ChannelOptions, fetch_channel_videos
from ..youtube.client import YtDlpClient
from .health import require_vision


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


def metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return normalized video facts without exposing the raw yt-dlp payload."""
    return asdict(VideoMetadata.from_yt_dlp(YtDlpClient().metadata(args.video_ref)))


def vision(args: argparse.Namespace) -> dict[str, Any]:
    """Return bounded timestamped visual context and its local collage paths."""
    require_vision()
    from ..vision import sample_vision

    return sample_vision(args.video_ref, max_frames=args.max_frames)


def channel(args: argparse.Namespace) -> dict[str, Any]:
    """Request a channel listing with the parsed count or date window."""
    return fetch_channel_videos(
        args.channel_ref, options=ChannelOptions(last=args.last, days=args.days)
    )


def search(args: argparse.Namespace) -> list[VideoResult] | list[ChunkResult]:
    """Translate parsed CLI arguments into the public search request contract."""
    from ..search import workflow

    limit = args.limit
    if limit is None:
        limit = (
            DEFAULT_CHANNEL_SEARCH_LIMIT
            if args.channels
            else DEFAULT_YOUTUBE_SEARCH_LIMIT
        )
    return workflow.search(
        SearchRequest(
            queries=tuple(query for query in args.queries if query),
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
            uploaded_after=args.uploaded_after,
            uploaded_within_days=args.uploaded_within_days,
            include_channel=tuple(args.include_channel),
            exclude_channel=tuple(args.exclude_channel),
            include_title=tuple(args.include_title),
            exclude_title=tuple(args.exclude_title),
        )
    )
