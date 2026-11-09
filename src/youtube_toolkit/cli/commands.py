"""CLI command handlers: each translates parsed arguments into retrieval workflows and returns their JSON payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import cache, partial
from pathlib import Path
from typing import Any

from ..config import llm_config
from ..rewrite import rewrite_transcript
from ..schemas import TranscriptDocument, VideoMetadata
from ..search import (
    ChannelTarget,
    ChunkResult,
    SearchRequest,
    VideoResult,
    search_videos,
)
from ..stt import (
    check_cloud_stt,
    check_local_stt,
    cloud_stt_status,
    local_stt_installed,
)
from ..transcript import (
    TranscriptExportResult,
    fetch_transcript,
    write_markdown_export,
    write_transcript_export,
)
from ..youtube import (
    DEFAULT_COMMENT_LIMIT,
    ChannelOptions,
    YtDlpClient,
    fetch_channel_videos,
    search_comments,
)
from .health import require_vision

# CLI option names that apply only with their part's flag, and the sample_vision parameter each vision option sets.
VISION_OPTIONS = {
    "max_frames": "max_frames",
    "min_similarity": "min_similarity",
    "min_gap": "min_gap_seconds",
    "local_only": "local_only",
    "auto_clean": "auto_clean",
}
PART_OPTIONS = {
    "transcript": ("timestamps", "stt", "refine", "translate", "export"),
    "vision": tuple(VISION_OPTIONS),
    "comments": ("comment_limit",),
}


def video(args: argparse.Namespace) -> dict[str, Any]:
    """Parts run fastest first and share one metadata fetch; comments alone skip it, since their own request carries the title."""
    for part, options in PART_OPTIONS.items():
        for option in options:
            if getattr(args, option) is not None and getattr(args, part) in (
                None,
                False,
            ):
                raise ValueError(f"--{option.replace('_', '-')} requires --{part}")
    if args.force and not args.export:
        raise ValueError("--force requires --export")
    if args.vision:
        require_vision()
    llm = None
    if args.refine or args.translate is not None:
        llm = llm_config()
        if llm is None:
            raise ValueError(
                "--refine and --translate need LLM_API_KEY; save one with `youtube setup --llm`"
            )
    client = YtDlpClient()
    payload = cache(partial(client.metadata, args.video_ref))
    shown_alone = not (args.transcript or args.comments is not None or args.vision)
    metadata = (
        asdict(VideoMetadata.from_yt_dlp(payload()))
        if args.metadata or shown_alone
        else None
    )
    document = (
        fetch_transcript(
            args.video_ref,
            stt="always" if args.stt else "fallback",
            client=client,
            metadata=payload(),
        )
        if args.transcript
        else None
    )
    comments = (
        search_comments(
            args.video_ref,
            args.comments,
            limit=DEFAULT_COMMENT_LIMIT
            if args.comment_limit is None
            else args.comment_limit,
        )
        if args.comments is not None
        else None
    )
    vision = None
    if args.vision:
        from ..vision import sample_vision

        vision = sample_vision(
            args.video_ref,
            metadata=payload(),
            document=document,
            **{
                parameter: getattr(args, option)
                for option, parameter in VISION_OPTIONS.items()
                if getattr(args, option) is not None
            },
        )
    # Rewriting waits for vision, which matches frames to timed caption lines that an article replaces with one block of paragraphs.
    if document is not None and llm is not None:
        document = rewrite_transcript(
            document,
            llm,
            language=args.translate,
            refine=bool(args.refine),
            article=not args.timestamps,
        )
    parts = {
        "transcript": None if document is None else _transcript(document, args, vision),
        "vision": vision,
        "comments": comments,
        "metadata": metadata,
    }
    return {name: part for name, part in parts.items() if part is not None}


def _transcript(
    document: TranscriptDocument,
    args: argparse.Namespace,
    vision: dict[str, Any] | None,
) -> str | list[dict[str, Any]] | TranscriptExportResult:
    timestamps = bool(args.timestamps)
    if args.export and vision is not None:
        return write_markdown_export(
            document, Path(vision["directory"]), timestamps=timestamps
        )
    if args.export:
        return write_transcript_export(
            document, timestamps=timestamps, force=args.force
        )
    if timestamps:
        return [asdict(chunk) for chunk in document.chunks]
    return document.text


def channel(args: argparse.Namespace) -> dict[str, Any]:
    return fetch_channel_videos(
        args.channel_ref, options=ChannelOptions(last=args.last, days=args.days)
    )


def stt_setup(args: argparse.Namespace) -> dict[str, Any]:
    cloud = check_cloud_stt() if args.cloud_test else cloud_stt_status()
    local = check_local_stt() if local_stt_installed() else {"installed": False}
    return {"cloud": cloud, "local": local}


def search(args: argparse.Namespace) -> list[VideoResult] | list[ChunkResult]:
    return search_videos(
        SearchRequest(
            queries=tuple(query for query in args.queries if query),
            mode=args.mode,
            return_type=args.return_type,
            channels=tuple(
                ChannelTarget.from_ref(channel) for channel in args.channels
            ),
            limit=args.limit,
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
