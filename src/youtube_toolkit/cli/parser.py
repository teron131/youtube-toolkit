"""Argument parser construction for the youtube command."""

from __future__ import annotations

import argparse

from .commands import ChannelCommands, SearchCommands, VideoCommands
from .health import run_doctor

DEFAULT_SEARCH_MIN_VIEWS = 10_000
DEFAULT_SEARCH_MIN_DURATION = "3:00"
DEFAULT_SEARCH_MAX_DURATION = "8:00:00"


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


def add_command_subparsers(
    parser: argparse.ArgumentParser,
    **kwargs,
) -> argparse._SubParsersAction:
    """Create command subparsers that inherit CLI string normalization."""
    return parser.add_subparsers(**kwargs, parser_class=YoutubeArgumentParser)


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and attach command handlers as parser defaults."""
    parser = YoutubeArgumentParser(
        prog="youtube", description="Transcript-first YouTube toolkit."
    )
    subparsers = add_command_subparsers(parser, dest="command", required=True)
    VideoCommandParser.add_to(subparsers)
    SearchCommandParser.add_to(subparsers)
    ChannelCommandParser.add_to(subparsers)
    add_doctor_parser(subparsers)
    return parser


class VideoCommandParser:
    """Parser group for retrieving video metadata, transcripts, and visual context."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        video_parser = subparsers.add_parser(
            "video", help="Fetch video metadata, transcripts, or visual context"
        )
        video_subparsers = add_command_subparsers(
            video_parser,
            dest="video_command",
            required=True,
            metavar="{metadata,transcript,vision}",
        )

        metadata_parser = video_subparsers.add_parser(
            "metadata", help="Fetch video metadata JSON"
        )
        metadata_parser.add_argument(
            "video_ref", metavar="url", help="YouTube video URL or ID"
        )
        metadata_parser.set_defaults(handler=VideoCommands.metadata)

        transcript_parser = video_subparsers.add_parser(
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

        vision_parser = video_subparsers.add_parser(
            "vision", help="Generate timestamped visual-context collages"
        )
        vision_parser.add_argument(
            "video_ref", metavar="url", help="YouTube video URL or ID"
        )
        vision_parser.set_defaults(
            handler=VideoCommands.vision,
            required_capability="vision",
            capability_feature="video vision",
        )


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
            "channel", help="Work with YouTube channels"
        )
        channel_subparsers = add_command_subparsers(
            channel_parser, dest="channel_command", required=True, metavar="{videos}"
        )

        videos_parser = channel_subparsers.add_parser(
            "videos", help="List recent videos from a YouTube channel"
        )
        videos_parser.add_argument(
            "channel_ref", help="YouTube channel URL, handle, or channel ID"
        )
        add_channel_window_args(videos_parser)
        videos_parser.set_defaults(handler=ChannelCommands.videos)


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check runtime, cache path, and optional capabilities"
    )
    doctor_parser.set_defaults(handler=run_doctor)


def add_channel_window_args(
    parser: argparse.ArgumentParser,
    *,
    default_last: int | None = 10,
    default_days: int | None = None,
) -> None:
    """Add the shared count and recency window controls for channel listings."""
    parser.add_argument(
        "--last", type=int, default=default_last, help="Return the latest N videos"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=default_days,
        help="Return videos uploaded in the last N days",
    )
