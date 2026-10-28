"""The youtube command's arguments; each subcommand stores its handler under `handler`, which the entrypoint calls with the parsed namespace."""

from __future__ import annotations

import argparse

from ..config import (
    DEFAULT_LAST_VIDEOS,
    DEFAULT_LOCAL_STT_MODEL,
    DEFAULT_SEARCH_MAX_DURATION,
    DEFAULT_SEARCH_MIN_DURATION,
    DEFAULT_SEARCH_MIN_VIEWS,
    DEFAULT_VISION_MAX_FRAMES,
    DEFAULT_VISION_MIN_GAP_SECONDS,
    DEFAULT_VISION_MIN_SIMILARITY,
    MAX_VISION_FRAMES,
)
from ..youtube import DEFAULT_COMMENT_LIMIT, MAX_COMMENT_LIMIT
from . import commands
from .health import run_doctor
from .setup import run_setup


class YoutubeArgumentParser(argparse.ArgumentParser):
    """Untyped string arguments are stripped and lose one layer of matching quotes, which shell-passed values can retain."""

    def add_argument(self, *args, **kwargs):
        action = kwargs.get("action", "store")
        if "type" not in kwargs and action in {"store", "append"}:
            kwargs["type"] = normalize_arg_string
        return super().add_argument(*args, **kwargs)


def normalize_arg_string(value: str) -> str:
    candidate = str(value).strip()
    if (
        len(candidate) >= 2
        and candidate[0] == candidate[-1]
        and candidate[0] in {"'", '"'}
    ):
        return candidate[1:-1].strip()
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = YoutubeArgumentParser(
        prog="youtube", description="Transcript-first YouTube toolkit."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=YoutubeArgumentParser
    )
    _add_video_commands(subparsers)
    _add_search_command(subparsers)
    _add_channel_command(subparsers)
    _add_setup_command(subparsers)
    _add_stt_setup_command(subparsers)
    _add_doctor_command(subparsers)
    return parser


def _add_video_commands(subparsers: argparse._SubParsersAction) -> None:
    metadata_parser = subparsers.add_parser(
        "metadata", help="Fetch video metadata JSON"
    )
    metadata_parser.add_argument(
        "video_ref", metavar="url", help="YouTube video URL or ID"
    )
    metadata_parser.set_defaults(handler=commands.metadata)

    comments_parser = subparsers.add_parser(
        "comments",
        help="Search text within a video's Top comments",
        description="Search a bounded sample of top-level comments by a case-insensitive literal keyword or phrase. Results alternate between YouTube's Top order and highest likes, skipping duplicates; replies are excluded.",
    )
    comments_parser.add_argument(
        "video_ref", metavar="url", help="YouTube video URL or ID"
    )
    comments_parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="Text to find in comments; omit or pass an empty string to return the full fetched sample",
    )
    comments_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_COMMENT_LIMIT,
        help=f"Maximum top-level comments to fetch before filtering (1-{MAX_COMMENT_LIMIT}; default: %(default)s)",
    )
    comments_parser.set_defaults(handler=commands.comments)

    transcript_parser = subparsers.add_parser(
        "transcript", help="Fetch a normalized transcript"
    )
    transcript_parser.add_argument(
        "video_ref", metavar="url", help="YouTube video URL or ID"
    )
    transcript_parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Return timed transcript chunks instead of one text string",
    )
    transcript_parser.add_argument(
        "--export",
        action="store_true",
        help="Write metadata and transcript to a title-and-channel .txt file in the working directory",
    )
    transcript_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing transcript export; requires --export",
    )
    transcript_parser.set_defaults(handler=commands.transcript)

    vision_parser = subparsers.add_parser(
        "vision", help="Generate timestamped visual-context collages"
    )
    vision_parser.add_argument(
        "video_ref", metavar="url", help="YouTube video URL or ID"
    )
    vision_parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_VISION_MAX_FRAMES,
        help=f"Maximum output screenshots (1-{MAX_VISION_FRAMES}; default: %(default)s); candidate coverage is independent",
    )
    vision_parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_VISION_MIN_SIMILARITY,
        help="Minimum raw cosine for semantic matches (default: %(default)s)",
    )
    vision_parser.add_argument(
        "--min-gap",
        type=float,
        default=DEFAULT_VISION_MIN_GAP_SECONDS,
        help="Minimum seconds between retained frames (default: %(default)s)",
    )
    vision_parser.add_argument(
        "--local-only",
        action="store_true",
        help="Pick evenly spaced frames locally without captions or Gemini embedding requests",
    )
    vision_parser.add_argument(
        "--auto-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the downloaded video and candidate frames after processing (default: on)",
    )
    vision_parser.set_defaults(handler=commands.vision)


def _add_search_command(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser(
        "search",
        help="Search videos or transcript chunks across YouTube or within channels",
    )
    search_parser.add_argument(
        "queries",
        nargs="*",
        metavar="query",
        help="Search query; give several to interleave their results",
    )
    search_parser.add_argument(
        "--mode", choices=("native", "semantic"), default="native"
    )
    search_parser.add_argument(
        "--return", dest="return_type", choices=("video", "chunk"), default=None
    )
    search_parser.add_argument(
        "--channels",
        action="append",
        default=[],
        metavar="CHANNEL",
        help="Search only this channel's uploads; URL, handle, or ID",
    )
    search_parser.add_argument("--limit", type=int, default=None)
    search_parser.add_argument(
        "--sort", choices=("relevance", "recency", "popularity"), default=None
    )
    search_parser.add_argument(
        "--min-views", type=int, default=DEFAULT_SEARCH_MIN_VIEWS
    )
    search_parser.add_argument("--min-duration", default=DEFAULT_SEARCH_MIN_DURATION)
    search_parser.add_argument("--max-duration", default=DEFAULT_SEARCH_MAX_DURATION)
    search_parser.add_argument(
        "--uploaded-after",
        default=None,
        help="Only include videos uploaded on or after YYYY-MM-DD",
    )
    search_parser.add_argument(
        "--uploaded-within-days",
        type=int,
        default=None,
        help="Only include videos uploaded within the last N days",
    )
    search_parser.add_argument(
        "--include-channel",
        "--include-channels",
        dest="include_channel",
        action="append",
        default=[],
    )
    search_parser.add_argument(
        "--exclude-channel",
        "--exclude-channels",
        dest="exclude_channel",
        action="append",
        default=[],
    )
    search_parser.add_argument(
        "--include-title",
        "--include-titles",
        dest="include_title",
        action="append",
        default=[],
    )
    search_parser.add_argument(
        "--exclude-title",
        "--exclude-titles",
        dest="exclude_title",
        action="append",
        default=[],
    )
    search_parser.set_defaults(handler=commands.search)


def _add_channel_command(subparsers: argparse._SubParsersAction) -> None:
    channel_parser = subparsers.add_parser(
        "channel", help="List recent videos from a YouTube channel"
    )
    channel_parser.add_argument(
        "channel_ref",
        metavar="channel",
        help="YouTube channel URL, handle, or channel ID",
    )
    channel_parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST_VIDEOS,
        help="Return the latest N videos",
    )
    channel_parser.add_argument(
        "--days", type=int, help="Return videos uploaded in the last N days"
    )
    channel_parser.set_defaults(handler=commands.channel)


def _add_setup_command(subparsers: argparse._SubParsersAction) -> None:
    setup_parser = subparsers.add_parser(
        "setup", help="Save API keys and settings for YouTube Toolkit only"
    )
    setup_parser.add_argument(
        "--embeddings", action="store_true", help="Configure embedding settings only"
    )
    setup_parser.add_argument(
        "--stt", action="store_true", help="Configure speech-to-text settings only"
    )
    setup_parser.set_defaults(handler=run_setup)


def _add_stt_setup_command(subparsers: argparse._SubParsersAction) -> None:
    setup_parser = subparsers.add_parser(
        "stt-setup",
        help=f"Report the speech-to-text setup; download the Hugging Face ASR model named by STT_LOCAL_MODEL (default: {DEFAULT_LOCAL_STT_MODEL}) and check that it runs",
    )
    setup_parser.add_argument(
        "--cloud-test",
        action="store_true",
        help="Also send a two-second test recording to the cloud provider (a real request, billed by the provider)",
    )
    setup_parser.set_defaults(handler=commands.stt_setup)


def _add_doctor_command(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check runtime, artifact path, and optional capabilities"
    )
    doctor_parser.set_defaults(handler=run_doctor)
