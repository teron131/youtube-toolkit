"""Argument parser construction for the youtube command."""

from __future__ import annotations

import argparse

from ..storage.videos import VIDEO_AGENT_EDITABLE_FIELDS
from ..watchlist_schedule import (
    DEFAULT_WATCHLIST_SCHEDULE_DAYS,
    DEFAULT_WATCHLIST_SCHEDULE_LABEL,
    DEFAULT_WATCHLIST_SCHEDULE_LAST,
    DEFAULT_WATCHLIST_SCHEDULE_MINUTES,
)
from .commands import (
    BlacklistCommands,
    ChannelCommands,
    DatabaseCommands,
    IndexCommands,
    SearchCommands,
    SetupCommands,
    VideoCommands,
    WatchlistCommands,
    run_doctor,
)

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
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {"'", '"'}:
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
    parser = YoutubeArgumentParser(prog="youtube", description="Transcript-first YouTube toolkit.")
    subparsers = add_command_subparsers(parser, dest="command", required=True)
    VideoCommandParser.add_to(subparsers)
    SearchCommandParser.add_to(subparsers)
    ChannelCommandParser.add_to(subparsers)
    WatchlistCommandParser.add_to(subparsers)
    BlacklistCommandParser.add_to(subparsers)
    DatabaseCommandParser.add_to(subparsers)
    IndexCommandParser.add_to(subparsers)
    SetupCommandParser.add_to(subparsers)
    add_doctor_parser(subparsers)
    return parser


class VideoCommandParser:
    """Parser group for retrieving video metadata and transcripts."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        video_parser = subparsers.add_parser("video", help="Fetch video metadata or transcripts")
        video_subparsers = add_command_subparsers(
            video_parser,
            dest="video_command",
            required=True,
            metavar="{metadata,transcript}",
        )

        metadata_parser = video_subparsers.add_parser("metadata", help="Fetch video metadata JSON")
        metadata_parser.add_argument("video_ref", metavar="url", help="YouTube video URL or ID")
        metadata_parser.set_defaults(handler=VideoCommands.metadata)

        transcript_parser = video_subparsers.add_parser("transcript", help="Fetch a normalized transcript")
        transcript_parser.add_argument("video_ref", metavar="url", help="YouTube video URL or ID")
        transcript_parser.set_defaults(handler=VideoCommands.transcript)


class SearchCommandParser:
    """Parser group for mixed YouTube, channel, and stored-video search."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        search_parser = subparsers.add_parser("search", help="Search videos or transcript chunks from selected sources")
        search_parser.add_argument("query", nargs="?", default="", help="Search query")
        search_parser.add_argument("--extra-query", dest="extra_queries", action="append", default=[], help="Additional YouTube relevance query")
        search_parser.add_argument("--source", action="append", choices=("youtube", "channel", "db"), default=None)
        search_parser.add_argument("--mode", choices=("native", "semantic"), default="native")
        search_parser.add_argument("--return", dest="return_type", choices=("video", "chunk"), default=None)
        search_parser.add_argument("--channels", action="append", default=[], metavar="CHANNEL", help="Channel URL, handle, or ID for channel source")
        search_parser.add_argument("--limit", type=int, default=None)
        search_parser.add_argument("--sort", choices=("relevance", "recency", "popularity"), default=None)
        search_parser.add_argument("--min-views", type=int, default=DEFAULT_SEARCH_MIN_VIEWS)
        search_parser.add_argument("--min-duration", default=DEFAULT_SEARCH_MIN_DURATION)
        search_parser.add_argument("--max-duration", default=DEFAULT_SEARCH_MAX_DURATION)
        search_parser.add_argument("--uploaded-after", default=None, help="Only include videos uploaded on or after YYYY-MM-DD")
        search_parser.add_argument("--uploaded-within-days", type=int, default=None, help="Only include videos uploaded within the last N days")
        search_parser.add_argument("--include-channel", "--include-channels", dest="include_channel", action="append", default=[])
        search_parser.add_argument("--exclude-channel", "--exclude-channels", dest="exclude_channel", action="append", default=[])
        search_parser.add_argument("--include-title", "--include-titles", dest="include_title", action="append", default=[])
        search_parser.add_argument("--exclude-title", "--exclude-titles", dest="exclude_title", action="append", default=[])
        search_parser.set_defaults(handler=SearchCommands.search)


class ChannelCommandParser:
    """Parser group for direct channel listing commands."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        channel_parser = subparsers.add_parser("channel", help="Work with YouTube channels")
        channel_subparsers = add_command_subparsers(channel_parser, dest="channel_command", required=True, metavar="{videos}")

        videos_parser = channel_subparsers.add_parser("videos", help="List recent videos from a YouTube channel")
        videos_parser.add_argument("channel_ref", help="YouTube channel URL, handle, or channel ID")
        add_channel_window_args(videos_parser)
        videos_parser.set_defaults(handler=ChannelCommands.videos)


class WatchlistCommandParser:
    """Parser group for watchlist membership, sync, recency, and schedule commands."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        watchlist_parser = subparsers.add_parser("watchlist", help="Manage or sync watchlist channels")
        watchlist_subparsers = add_command_subparsers(watchlist_parser, dest="watchlist_command", required=True, metavar="{add,list,remove,sync,recent,schedule}")
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
        schedule_subparsers = add_command_subparsers(schedule_parser, dest="schedule_command", required=True)
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
    """Parser group for search blacklist membership commands."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        blacklist_parser = subparsers.add_parser("blacklist", help="Manage search blacklist channels")
        blacklist_subparsers = add_command_subparsers(blacklist_parser, dest="blacklist_command", required=True)
        blacklist_add_parser = blacklist_subparsers.add_parser("add", help="Add one or more channels to the search blacklist")
        add_channel_refs_arg(blacklist_add_parser)
        blacklist_add_parser.set_defaults(handler=BlacklistCommands.add)
        blacklist_remove_parser = blacklist_subparsers.add_parser("remove", help="Remove one or more channels from the search blacklist")
        add_channel_refs_arg(blacklist_remove_parser)
        blacklist_remove_parser.set_defaults(handler=BlacklistCommands.remove)


class IndexCommandParser:
    """Parser group for embedding index maintenance commands."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        index_parser = subparsers.add_parser("index", help="Index stored video embeddings")
        index_parser.add_argument("video_refs", nargs="*", metavar="url", help="Stored YouTube video URL or ID")
        index_parser.add_argument(
            "--include-without-transcript",
            action="store_true",
            help="Include stored videos without transcripts when indexing all videos",
        )
        index_parser.set_defaults(handler=IndexCommands.videos)


class DatabaseCommandParser:
    """Parser group for inspecting and administering stored video records."""

    @staticmethod
    def add_to(subparsers: argparse._SubParsersAction) -> None:
        database_parser = subparsers.add_parser("db", help="Inspect or edit the active LanceDB store")
        database_subparsers = add_command_subparsers(
            database_parser,
            dest="database_command",
            required=True,
            metavar="{view,get,refresh,edit,delete}",
        )
        view_parser = database_subparsers.add_parser("view", help="Open the live read-only LanceDB viewer")
        view_parser.add_argument("--port", type=int, default=0, help="Loopback port; defaults to an available port")
        view_parser.add_argument("--no-open", action="store_true", help="Start the viewer without opening a browser")
        view_parser.set_defaults(handler=DatabaseCommands.view)

        get_parser = database_subparsers.add_parser("get", help="Return one locally stored video record")
        get_parser.add_argument("video_ref", metavar="url", help="Stored YouTube video URL or ID")
        get_parser.set_defaults(handler=DatabaseCommands.get)

        refresh_parser = database_subparsers.add_parser("refresh", help="Refresh YouTube-owned metadata in storage")
        refresh_parser.add_argument("video_ref", metavar="url", help="YouTube video URL or ID")
        refresh_parser.set_defaults(handler=DatabaseCommands.refresh)

        edit_parser = database_subparsers.add_parser("edit", help="Edit agent-owned content on one stored video")
        edit_parser.add_argument("video_ref", metavar="url", help="Stored YouTube video URL or ID")
        edit_parser.add_argument("--summary", help="Replace the stored summary")
        edit_parser.add_argument("--transcript-file", metavar="PATH", help="Replace the transcript from a UTF-8 file; use - for stdin")
        edit_parser.add_argument(
            "--clear",
            action="append",
            choices=tuple(field.replace("_", "-") for field in VIDEO_AGENT_EDITABLE_FIELDS),
            default=[],
            help="Clear one stored field; repeat for multiple fields",
        )
        edit_parser.add_argument("--dry-run", action="store_true", help="Validate and report changes without writing")
        edit_parser.set_defaults(handler=DatabaseCommands.edit)

        delete_parser = database_subparsers.add_parser("delete", help="Delete one stored video and its derived rows")
        delete_parser.add_argument("video_ref", metavar="url", help="Stored YouTube video URL or ID")
        delete_parser.add_argument("--dry-run", action="store_true", help="Report the rows that would be deleted")
        delete_parser.add_argument("--yes", action="store_true", help="Confirm the deletion")
        delete_parser.set_defaults(handler=DatabaseCommands.delete)


class SetupCommandParser:
    """Parser group for persistent model API configuration."""

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
    """Add the shared count and recency window controls for channel listings."""
    parser.add_argument("--last", type=int, default=default_last, help="Return the latest N videos")
    parser.add_argument("--days", type=int, default=default_days, help="Return videos uploaded in the last N days")


def add_channel_refs_arg(parser: argparse.ArgumentParser) -> None:
    """Add the shared one-or-more channel reference positional argument."""
    parser.add_argument(
        "channel_refs",
        nargs="+",
        metavar="channel",
        help="YouTube channel URL, handle, or channel ID; pass multiple as space-separated arguments",
    )
