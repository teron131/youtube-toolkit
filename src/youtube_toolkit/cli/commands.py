"""CLI command handlers: each translates parsed arguments into retrieval workflows and returns their JSON payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import cache, partial
from pathlib import Path
from typing import Any

from ..config import VISION_SOURCE_HEIGHT, llm_config
from ..download import (
    DEFAULT_DOWNLOAD_QUALITY,
    DOWNLOAD_SIZES,
    DownloadResult,
    download_frames,
    download_frames_at,
    download_media,
)
from ..export import (
    TranscriptExportResult,
    write_markdown_export,
    write_text_export,
)
from ..rewrite import rewrite_transcript
from ..schemas import TranscriptDocument, VideoMetadata
from ..search import (
    ChannelTarget,
    ChunkResult,
    SearchRequest,
    VideoResult,
    parse_duration,
    search_videos,
)
from ..stt import (
    check_cloud_stt,
    check_local_stt,
    cloud_stt_status,
    local_stt_installed,
)
from ..transcript import fetch_transcript
from ..youtube import (
    DEFAULT_COMMENT_LIMIT,
    ChannelOptions,
    YtDlpClient,
    fetch_channel_videos,
    search_comments,
)
from .health import require_vision

# CLI option names that apply only with one of their parts' flags, and the sample_vision parameter each vision option sets.
VISION_OPTIONS = {
    "max_frames": "max_frames",
    "rows": "rows",
    "columns": "columns",
    "resolution": "resolution",
    "timestamp_annotation": "timestamp_annotation",
    "min_similarity": "min_similarity",
    "min_gap": "min_gap_seconds",
    "local_only": "local_only",
    "auto_clean": "auto_clean",
}
PART_OPTIONS = {
    ("transcript",): ("timestamps", "refine", "export"),
    ("transcript", "summary"): ("stt", "translate"),
    ("vision",): tuple(VISION_OPTIONS),
    ("download",): ("quality", "at"),
    ("download", "vision"): ("start", "end"),
    ("comments",): ("comment_limit",),
}


def video(args: argparse.Namespace) -> dict[str, Any]:
    """Parts share one metadata fetch, which comments alone skip since their own request carries the title."""
    for parts, options in PART_OPTIONS.items():
        if any(getattr(args, part) not in (None, False) for part in parts):
            continue
        for option in options:
            if getattr(args, option) is not None:
                flags = " or ".join(f"--{part}" for part in parts)
                raise ValueError(f"--{option.replace('_', '-')} requires {flags}")
    if args.fresh and not (args.export or args.vision or args.download):
        raise ValueError("--fresh requires --export, --vision, or --download")
    if args.download == "audio" and args.quality is not None:
        raise ValueError("--quality applies only to video and frame downloads")
    if args.at is not None and args.download != "frames":
        raise ValueError("--at applies only to --download frames")
    if args.at is not None and (args.start is not None or args.end is not None):
        raise ValueError("--at gives its own times, so it cannot take --start or --end")
    start = parse_duration(args.start)
    end = parse_duration(args.end)
    quality = args.quality or DEFAULT_DOWNLOAD_QUALITY
    if args.vision or args.download in ("frames", "keyframes"):
        require_vision()
    llm = None
    if args.refine or args.translate is not None or args.summary is not None:
        llm = llm_config()
        if llm is None:
            raise ValueError(
                "--refine, --translate, and --summary need LLM_API_KEY; save one with `youtube setup --llm`"
            )
    client = YtDlpClient()
    payload = cache(partial(client.metadata, args.video_ref))
    shown_alone = not (
        args.transcript
        or args.summary is not None
        or args.comments is not None
        or args.vision
        or args.download
    )
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
        if args.transcript or args.summary is not None
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
    summary = None
    if args.summary is not None and document is not None and llm is not None:
        from ..summary import summarize_transcript

        # The summary reads the transcript as fetched rather than a translation of it, and writes in the --translate language itself.
        summary = summarize_transcript(
            document, llm, language=args.summary or args.translate
        )
    download: DownloadResult | list[DownloadResult] | None = None
    downloaded_video = None
    if args.download in ("video", "audio"):
        media = download_media(
            payload(),
            args.download,
            quality=quality,
            start=start,
            end=end,
            fresh=args.fresh,
            client=client,
        )
        download = media
        # Vision downloads its source at VISION_SOURCE_HEIGHT, so a video download at least that tall, over the same range, serves it instead.
        if (
            args.download == "video"
            and DOWNLOAD_SIZES[quality][1] >= VISION_SOURCE_HEIGHT
        ):
            downloaded_video = Path(media["path"])
    elif args.at is not None:
        download = download_frames_at(
            payload(),
            [parse_duration(time) for time in args.at],
            quality=quality,
            fresh=args.fresh,
            client=client,
        )
    elif args.download is not None:
        download = download_frames(
            payload(),
            args.download,
            quality=quality,
            start=start,
            end=end,
            fresh=args.fresh,
            client=client,
        )
    vision = None
    if args.vision:
        from ..vision import sample_vision

        vision = sample_vision(
            args.video_ref,
            metadata=payload(),
            document=document,
            start=start,
            end=end,
            fresh=args.fresh,
            video_path=downloaded_video,
            **{
                parameter: getattr(args, option)
                for option, parameter in VISION_OPTIONS.items()
                if getattr(args, option) is not None
            },
        )
    # Rewriting waits for vision, which matches frames to timed caption lines that an article replaces with one block of paragraphs.
    # A summary alone prints no transcript, so its --translate language must not buy a translation nobody reads.
    if args.transcript and document is not None and llm is not None:
        document = rewrite_transcript(
            document,
            llm,
            language=args.translate,
            refine=bool(args.refine),
            article=not args.timestamps,
        )
    parts = {
        "transcript": None
        if document is None or not args.transcript
        else _transcript(document, args, vision),
        "summary": summary,
        "vision": vision,
        "download": download,
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
        return write_text_export(document, timestamps=timestamps, fresh=args.fresh)
    if timestamps:
        return [asdict(segment) for segment in document.segments]
    return document.text


def channel(args: argparse.Namespace) -> dict[str, Any]:
    return fetch_channel_videos(
        args.channel_ref, options=ChannelOptions(last=args.last, days=args.days)
    )


def stt_check(args: argparse.Namespace) -> dict[str, Any]:
    cloud = check_cloud_stt() if args.cloud_test else cloud_stt_status()
    local = check_local_stt() if local_stt_installed() else {"installed": False}
    return {"cloud": cloud, "local": local}


def search(args: argparse.Namespace) -> list[VideoResult] | list[ChunkResult]:
    return search_videos(
        SearchRequest(
            queries=tuple(query for query in args.queries if query),
            mode="native" if args.semantic is None else "semantic",
            return_type=args.semantic,
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
