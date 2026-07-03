"""Argument parser construction for the youtube command."""

from __future__ import annotations

import argparse

from ..watchlist_schedule import (
    DEFAULT_WATCHLIST_SCHEDULE_DAYS,
    DEFAULT_WATCHLIST_SCHEDULE_LABEL,
    DEFAULT_WATCHLIST_SCHEDULE_LAST,
    DEFAULT_WATCHLIST_SCHEDULE_MINUTES,
)
from .commands import (
    BlacklistCommands,
    ChannelCommands,
    IndexCommands,
    SetupCommands,
    VideoCommands,
    WatchlistCommands,
    run_doctor,
    run_search,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="youtube", description="Transcript-first YouTube toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    VideoCommandParser.add_to(subparsers)
    add_search_parser(subparsers)
    ChannelCommandParser.add_to(subparsers)
    WatchlistCommandParser.add_to(subparsers)
    BlacklistCommandParser.add_to(subparsers)
    IndexCommandParser.add_to(subparsers)
    SetupCommandParser.add_to(subparsers)
    add_doctor_parser(subparsers)
    return parser


class VideoCommandParser:
    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        video_parser = subparsers.add_parser("video", help="Fetch video metadata or transcripts")
        video_subparsers = video_parser.add_subparsers(
            dest="video_command",
            required=True,
            metavar="{metadata,transcript}",
        )

        metadata_parser = video_subparsers.add_parser("metadata", help="Fetch video metadata JSON")
        metadata_parser.add_argument("video_ref", help="YouTube video URL or ID")
        metadata_parser.set_defaults(handler=VideoCommands.metadata)

        transcript_parser = video_subparsers.add_parser("transcript", help="Fetch a normalized transcript")
        transcript_parser.add_argument("video_ref", help="YouTube video URL or ID")
        transcript_parser.set_defaults(handler=VideoCommands.transcript)


def add_search_parser(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser("search", help="Query YouTube through yt-dlp with local filters")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--sort", choices=("relevance", "views"), default="relevance")
    search_parser.add_argument("--min-views", type=int, default=10_000)
    search_parser.add_argument("--min-duration", default="3:00")
    search_parser.add_argument("--max-duration", default="8:00:00")
    search_parser.add_argument("--include-channel", action="append", default=[])
    search_parser.add_argument("--exclude-channel", action="append", default=[])
    search_parser.add_argument("--include-title", action="append", default=[])
    search_parser.add_argument("--exclude-title", action="append", default=[])
    search_parser.set_defaults(handler=run_search)


class ChannelCommandParser:
    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        channel_parser = subparsers.add_parser("channel", help="Work with YouTube channels")
        channel_subparsers = channel_parser.add_subparsers(
            dest="channel_command",
            required=True,
            metavar="{videos}",
        )

        videos_parser = channel_subparsers.add_parser("videos", help="List recent videos from a YouTube channel")
        videos_parser.add_argument("channel_ref", help="YouTube channel URL, handle, or channel ID")
        add_channel_window_args(videos_parser)
        videos_parser.set_defaults(handler=ChannelCommands.videos)


class WatchlistCommandParser:
    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        watchlist_parser = subparsers.add_parser("watchlist", help="Manage or sync watchlist channels")
        watchlist_subparsers = watchlist_parser.add_subparsers(
            dest="watchlist_command",
            required=True,
            metavar="{add,list,remove,sync,recent,schedule}",
        )
        watchlist_add_parser = watchlist_subparsers.add_parser("add", help="Add one or more channels to the watchlist")
        add_channel_refs_arg(watchlist_add_parser)
        watchlist_add_parser.set_defaults(handler=WatchlistCommands.add)
        watchlist_list_parser = watchlist_subparsers.add_parser("list", help="List watched channels")
        watchlist_list_parser.set_defaults(handler=WatchlistCommands.list)
        watchlist_remove_parser = watchlist_subparsers.add_parser("remove", help="Remove one or more channels from the watchlist")
        add_channel_refs_arg(watchlist_remove_parser)
        watchlist_remove_parser.set_defaults(handler=WatchlistCommands.remove)
        watchlist_sync_parser = watchlist_subparsers.add_parser("sync", help="Sync watchlist videos and transcripts")
        add_channel_window_args(watchlist_sync_parser)
        watchlist_sync_parser.set_defaults(handler=WatchlistCommands.sync)
        watchlist_scheduled_parser = watchlist_subparsers.add_parser("scheduled-sync", help=argparse.SUPPRESS)
        add_channel_window_args(
            watchlist_scheduled_parser,
            default_last=DEFAULT_WATCHLIST_SCHEDULE_LAST,
            default_days=DEFAULT_WATCHLIST_SCHEDULE_DAYS,
        )
        watchlist_scheduled_parser.set_defaults(handler=WatchlistCommands.scheduled_sync)
        # argparse has no public API for hidden-but-parseable subcommands.
        watchlist_subparsers._choices_actions = [action for action in watchlist_subparsers._choices_actions if action.dest != "scheduled-sync"]
        watchlist_recent_parser = watchlist_subparsers.add_parser("recent", help="Show recently uploaded videos from watched channels")
        watchlist_recent_parser.add_argument("channel_refs", nargs="*", metavar="channel", help="Limit recent videos to watched channels")
        watchlist_recent_parser.add_argument(
            "--days",
            type=int,
            default=DEFAULT_WATCHLIST_SCHEDULE_DAYS,
            metavar="N",
            help="Return videos uploaded in the last N days",
        )
        watchlist_recent_parser.set_defaults(handler=WatchlistCommands.recent)
        WatchlistCommandParser.add_schedule(watchlist_subparsers)

    @staticmethod
    def add_schedule(subparsers: argparse._SubParsersAction) -> None:
        schedule_parser = subparsers.add_parser("schedule", help="Manage the watchlist scheduled sync LaunchAgent")
        schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command", required=True)
        install_parser = schedule_subparsers.add_parser("install", help="Install and load the watchlist scheduled sync job")
        install_parser.add_argument("--minutes", type=int, default=DEFAULT_WATCHLIST_SCHEDULE_MINUTES, help="Run interval in minutes")
        install_parser.add_argument("--last", type=int, default=DEFAULT_WATCHLIST_SCHEDULE_LAST, help="Fetch the latest N videos per watched channel; omit for no limit")
        install_parser.add_argument("--days", type=int, default=DEFAULT_WATCHLIST_SCHEDULE_DAYS, help="Look back this many days")
        install_parser.add_argument("--label", default=DEFAULT_WATCHLIST_SCHEDULE_LABEL, help="LaunchAgent label")
        install_parser.add_argument("--no-load", action="store_true", help="Write the plist without loading it")
        install_parser.set_defaults(handler=WatchlistCommands.install_schedule)
        status_parser = schedule_subparsers.add_parser("status", help="Show watchlist scheduled sync status")
        status_parser.add_argument("--label", default=DEFAULT_WATCHLIST_SCHEDULE_LABEL, help="LaunchAgent label")
        status_parser.set_defaults(handler=WatchlistCommands.schedule_status)
        uninstall_parser = schedule_subparsers.add_parser("uninstall", help="Unload and remove the watchlist scheduled sync job")
        uninstall_parser.add_argument("--label", default=DEFAULT_WATCHLIST_SCHEDULE_LABEL, help="LaunchAgent label")
        uninstall_parser.set_defaults(handler=WatchlistCommands.uninstall_schedule)


class BlacklistCommandParser:
    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        blacklist_parser = subparsers.add_parser("blacklist", help="Manage search blacklist channels")
        blacklist_subparsers = blacklist_parser.add_subparsers(dest="blacklist_command", required=True)
        blacklist_add_parser = blacklist_subparsers.add_parser("add", help="Add one or more channels to the search blacklist")
        add_channel_refs_arg(blacklist_add_parser)
        blacklist_add_parser.set_defaults(handler=BlacklistCommands.add)
        blacklist_remove_parser = blacklist_subparsers.add_parser("remove", help="Remove one or more channels from the search blacklist")
        add_channel_refs_arg(blacklist_remove_parser)
        blacklist_remove_parser.set_defaults(handler=BlacklistCommands.remove)


class IndexCommandParser:
    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        index_parser = subparsers.add_parser("index", help="Index stored video embeddings")
        index_parser.add_argument("video_refs", nargs="*", metavar="video", help="Stored YouTube video URL or ID")
        index_parser.add_argument(
            "--include-without-transcript",
            action="store_true",
            help="Include stored videos without transcripts when indexing all videos",
        )
        index_parser.set_defaults(handler=IndexCommands.videos)


class SetupCommandParser:
    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        setup_parser = subparsers.add_parser("setup", help="Configure model API keys, base URLs, and model IDs")
        setup_parser.add_argument("--embedding-api-key")
        setup_parser.add_argument("--embedding-base-url")
        setup_parser.add_argument("--embedding-model")
        setup_parser.add_argument("--embedding-dimensions", type=int)
        setup_parser.add_argument("--llm-api-key")
        setup_parser.add_argument("--llm-base-url")
        setup_parser.add_argument("--llm-model")
        setup_parser.add_argument(
            "--auto-embeddings",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Index transcript embeddings automatically after transcript storage",
        )
        setup_parser.add_argument(
            "--verify",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Send 'Hello' to the configured model APIs after saving",
        )
        setup_parser.set_defaults(handler=SetupCommands.models)


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser("doctor", help="Check runtime and database path")
    doctor_parser.set_defaults(handler=run_doctor)


def add_channel_window_args(
    parser: argparse.ArgumentParser,
    *,
    default_last: int | None = 10,
    default_days: int | None = None,
) -> None:
    parser.add_argument("--last", type=int, default=default_last, help="Return the latest N videos")
    parser.add_argument("--days", type=int, default=default_days, help="Return videos uploaded in the last N days")


def add_channel_refs_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "channel_refs",
        nargs="+",
        metavar="channel",
        help="YouTube channel URL, handle, or channel ID; pass multiple as space-separated arguments",
    )
