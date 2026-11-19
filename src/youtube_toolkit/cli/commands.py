"""CLI command handlers: each translates parsed arguments into retrieval workflows and returns their JSON payload."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import cache, partial
from pathlib import Path
from typing import Any

from ..artifacts import Cached, save_json, save_text, video_directory
from ..config import VISION_SOURCE_HEIGHT, llm_config
from ..download import (
    DEFAULT_DOWNLOAD_QUALITY,
    DOWNLOAD_SIZES,
    FrameDownload,
    FrameFolderDownload,
    MediaDownload,
    download_frames,
    download_frames_at,
    download_media,
)
from ..identity import VideoRef, require_video_id
from ..markdown import TranscriptLayout, summary_markdown, transcript_markdown
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
    ("transcript",): ("timestamps", "refine"),
    ("transcript", "summary"): ("stt", "translate"),
    ("vision",): tuple(VISION_OPTIONS),
    ("download",): ("quality", "at"),
    ("download", "vision"): ("start", "end"),
    ("comments",): ("comment_limit",),
}


def video(args: argparse.Namespace) -> dict[str, Any]:
    """Each part is saved in the video's folder, and the output gives its path with a few facts, so its size never depends on the content's.

    Parts share one metadata fetch, which comments alone skip since their own request carries the title.
    """
    for parts, options in PART_OPTIONS.items():
        if any(getattr(args, part) not in (None, False) for part in parts):
            continue
        for option in options:
            if getattr(args, option) is not None:
                flags = " or ".join(f"--{part}" for part in parts)
                raise ValueError(f"--{option.replace('_', '-')} requires {flags}")
    if args.fresh and not (
        args.transcript or args.summary is not None or args.vision or args.download
    ):
        raise ValueError(
            "--fresh requires --transcript, --summary, --vision, or --download"
        )
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
    metadata = _metadata(payload()) if args.metadata or shown_alone else None
    transcript = (
        fetch_transcript(
            args.video_ref,
            stt="always" if args.stt else "fallback",
            client=client,
            metadata=payload(),
            fresh=args.fresh,
        )
        if args.transcript or args.summary is not None
        else None
    )
    comments = (
        _comments(
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
    if args.summary is not None and transcript is not None and llm is not None:
        from ..summary import summarize_transcript

        # The summary reads the transcript as fetched rather than a translation of it, and writes in the --translate language itself.
        language = args.summary or args.translate
        summary = _summary(
            summarize_transcript(transcript, llm, language=language, fresh=args.fresh),
            transcript.value,
            language,
            model=llm.model,
        )
    download: MediaDownload | FrameFolderDownload | list[FrameDownload] | None = None
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
            document=None if transcript is None else transcript.value,
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
    # A summary alone shows no transcript, so its --translate language must not buy a translation nobody reads.
    if args.transcript and transcript is not None and llm is not None:
        transcript = rewrite_transcript(
            transcript,
            llm,
            language=args.translate,
            refine=bool(args.refine),
            article=not args.timestamps,
            fresh=args.fresh,
        )
    parts = {
        "transcript": None
        if transcript is None or not args.transcript
        else _transcript(transcript, timestamps=bool(args.timestamps)),
        "summary": summary,
        "vision": vision,
        "download": download,
        "comments": comments,
        "metadata": metadata,
    }
    return {name: part for name, part in parts.items() if part is not None}


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = VideoMetadata.from_yt_dlp(payload)
    path = video_directory(metadata.video_id, metadata.title) / "metadata.json"
    save_json(path, asdict(metadata))
    return {
        "path": str(path),
        "title": metadata.title,
        "channel": metadata.channel,
        "duration_seconds": metadata.duration_seconds,
    }


def _transcript(
    transcript: Cached[TranscriptDocument], *, timestamps: bool
) -> dict[str, Any]:
    """The timed reading copy ends in `.timed.md`, so a transcript read both ways keeps both copies."""
    document = transcript.value
    layout: TranscriptLayout = (
        "timed"
        if timestamps and document.timed
        # Without --timestamps, a rewrite takes the article route.
        else "article"
        if document.rewrite is not None and not timestamps
        else "sentences"
    )
    suffix = ".timed.md" if layout == "timed" else ".md"
    return {
        **_reading_copy(
            transcript.path.with_name(transcript.path.stem + suffix),
            transcript_markdown(document, layout),
        ),
        "language": document.language,
        "source": document.source,
        "reused": transcript.reused,
    }


def _summary(
    summary: Cached[str],
    document: TranscriptDocument,
    language: str | None,
    *,
    model: str,
) -> dict[str, Any]:
    return {
        **_reading_copy(
            summary.path.with_suffix(".md"),
            summary_markdown(document.metadata, summary.value, model=model),
        ),
        "language": language or document.language,
        "reused": summary.reused,
    }


def _comments(video_ref: VideoRef, query: str, *, limit: int) -> dict[str, Any]:
    result = search_comments(video_ref, query, limit=limit)
    path = (
        video_directory(require_video_id(video_ref), result["title"]) / "comments.json"
    )
    save_json(path, result)
    return {
        "path": str(path),
        "matches": len(result["comments"]),
        "sampled": result["sampled"],
    }


def _reading_copy(path: Path, text: str) -> dict[str, Any]:
    save_text(path, text)
    return {"path": str(path), "characters": len(text)}


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
