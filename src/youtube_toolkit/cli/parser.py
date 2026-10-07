"""Argument parser construction for the youtube command."""

from __future__ import annotations

import argparse

from ..config import (
    DEFAULT_SEARCH_MAX_DURATION,
    DEFAULT_SEARCH_MIN_DURATION,
    DEFAULT_SEARCH_MIN_VIEWS,
)
from ..youtube.channels import DEFAULT_LAST_VIDEOS
from .commands import ChannelCommands, SearchCommands, VideoCommands
from .health import run_doctor


class YoutubeArgumentParser(argparse.ArgumentParser):
    """CLI parser normalizes shell-passed string argument values before handlers run."""

    def add_argument(self, *args, **kwargs):
        action = kwargs.get("action", "store")
        if "type" not in kwargs and action in {"store", "append"}:
            kwargs["type"] = normalize_arg_string
        return super().add_argument(*args, **kwargs)


def normalize_arg_string(value: str) -> str:
    """Normalize one shell-passed argument string before argparse stores or validates it."""
    candidate = str(value).strip()
    if (
        len(candidate) >= 2
        and candidate[0] == candidate[-1]
        and candidate[0] in {"'", '"'}
    ):
        return candidate[1:-1].strip()
    return candidate


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and attach command handlers as parser defaults."""
    parser = YoutubeArgumentParser(
        prog="youtube", description="Transcript-first YouTube toolkit."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=YoutubeArgumentParser
    )
    VideoCommandParser.add_to(subparsers)
    SearchCommandParser.add_to(subparsers)
    ChannelCommandParser.add_to(subparsers)
    add_doctor_parser(subparsers)
    return parser


class VideoCommandParser:
    """Register top-level metadata, transcript, and vision commands with their existing retrieval handlers."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        metadata_parser = subparsers.add_parser(
            "metadata", help="Fetch video metadata JSON"
        )
        metadata_parser.add_argument(
            "video_ref", metavar="url", help="YouTube video URL or ID"
        )
        metadata_parser.set_defaults(handler=VideoCommands.metadata)

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
        transcript_parser.set_defaults(handler=VideoCommands.transcript)

        vision_parser = subparsers.add_parser(
            "vision", help="Generate timestamped visual-context collages"
        )
        vision_parser.add_argument(
            "video_ref", metavar="url", help="YouTube video URL or ID"
        )
        vision_parser.set_defaults(handler=VideoCommands.vision)


class SearchCommandParser:
    """Parser group for YouTube and channel search."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        search_parser = subparsers.add_parser(
            "search", help="Search videos or transcript chunks from selected sources"
        )
        search_parser.add_argument("query", nargs="?", default="", help="Search query")
        search_parser.add_argument(
            "--extra-query",
            dest="extra_queries",
            action="append",
            default=[],
            help="Additional YouTube relevance query",
        )
        search_parser.add_argument(
            "--source", action="append", choices=("youtube", "channel"), default=None
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
            help="Channel URL, handle, or ID for channel source",
        )
        search_parser.add_argument("--limit", type=int, default=None)
        search_parser.add_argument(
            "--sort", choices=("relevance", "recency", "popularity"), default=None
        )
        search_parser.add_argument(
            "--min-views", type=int, default=DEFAULT_SEARCH_MIN_VIEWS
        )
        search_parser.add_argument(
            "--min-duration", default=DEFAULT_SEARCH_MIN_DURATION
        )
        search_parser.add_argument(
            "--max-duration", default=DEFAULT_SEARCH_MAX_DURATION
        )
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
        search_parser.set_defaults(handler=SearchCommands.search)


class ChannelCommandParser:
    """Parser group for direct channel listing commands."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
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
        channel_parser.set_defaults(handler=ChannelCommands.videos)


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check runtime, cache path, and optional capabilities"
    )
    doctor_parser.set_defaults(handler=run_doctor)
