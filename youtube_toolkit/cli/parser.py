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
    run_blacklist_add,
    run_blacklist_remove,
    run_channel_videos,
    run_doctor,
    run_inspect,
    run_search,
    run_transcript,
    run_watchlist_add,
    run_watchlist_new,
    run_watchlist_remove,
    run_watchlist_schedule_install,
    run_watchlist_schedule_status,
    run_watchlist_schedule_uninstall,
    run_watchlist_scheduled_sync,
    run_watchlist_sync,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="youtube", description="Transcript-first YouTube toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_transcript_parser(subparsers)
    add_inspect_parser(subparsers)
    add_search_parser(subparsers)
    add_channel_videos_parser(subparsers)
    add_watchlist_parser(subparsers)
    add_blacklist_parser(subparsers)
    add_doctor_parser(subparsers)
    return parser


def add_transcript_parser(subparsers: argparse._SubParsersAction) -> None:
    transcript_parser = subparsers.add_parser("transcript", help="Fetch a normalized transcript")
    transcript_parser.add_argument("video_ref", help="YouTube video URL or ID")
    transcript_parser.set_defaults(handler=run_transcript)


def add_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    inspect_parser = subparsers.add_parser("inspect", help="Fetch video metadata JSON")
    inspect_parser.add_argument("video_ref", help="YouTube video URL or ID")
    inspect_parser.set_defaults(handler=run_inspect)


def add_search_parser(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser("search", help="Search YouTube videos")
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


def add_channel_videos_parser(subparsers: argparse._SubParsersAction) -> None:
    channel_parser = subparsers.add_parser("channel-videos", help="List recent videos from a YouTube channel")
    channel_parser.add_argument("channel_ref", help="YouTube channel URL, handle, or channel ID")
    add_channel_window_args(channel_parser)
    channel_parser.set_defaults(handler=run_channel_videos)


def add_watchlist_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add watchlist management subcommands to the root parser."""
    watchlist_parser = subparsers.add_parser("watchlist", help="Manage or sync watchlist channels")
    watchlist_subparsers = watchlist_parser.add_subparsers(
        dest="watchlist_command",
        required=True,
        metavar="{add,remove,sync,new,schedule}",
    )
    watchlist_add_parser = watchlist_subparsers.add_parser("add", help="Add one or more channels to the watchlist")
    add_channel_refs_arg(watchlist_add_parser)
    watchlist_add_parser.set_defaults(handler=run_watchlist_add)
    watchlist_remove_parser = watchlist_subparsers.add_parser("remove", help="Remove one or more channels from the watchlist")
    add_channel_refs_arg(watchlist_remove_parser)
    watchlist_remove_parser.set_defaults(handler=run_watchlist_remove)
    watchlist_sync_parser = watchlist_subparsers.add_parser("sync", help="Sync watchlist videos and transcripts")
    add_channel_window_args(watchlist_sync_parser)
    watchlist_sync_parser.set_defaults(handler=run_watchlist_sync)
    watchlist_scheduled_parser = watchlist_subparsers.add_parser("scheduled-sync", help=argparse.SUPPRESS)
    add_channel_window_args(watchlist_scheduled_parser)
    watchlist_scheduled_parser.set_defaults(handler=run_watchlist_scheduled_sync)
    # argparse has no public API for hidden-but-parseable subcommands.
    watchlist_subparsers._choices_actions = [action for action in watchlist_subparsers._choices_actions if action.dest != "scheduled-sync"]
    watchlist_new_parser = watchlist_subparsers.add_parser("new", help="Return and mark new watched videos")
    watchlist_new_parser.add_argument("--channel", action="append", default=[], dest="channel_refs", help="Consume new videos from a watched channel; repeat for more channels")
    watchlist_new_parser.add_argument("--all", action="store_true", help="Consume new videos from all watched channels")
    watchlist_new_parser.set_defaults(handler=run_watchlist_new)
    add_watchlist_schedule_parser(watchlist_subparsers)


def add_watchlist_schedule_parser(subparsers: argparse._SubParsersAction) -> None:
    schedule_parser = subparsers.add_parser("schedule", help="Manage the watchlist scheduled sync LaunchAgent")
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command", required=True)
    install_parser = schedule_subparsers.add_parser("install", help="Install and load the watchlist scheduled sync job")
    install_parser.add_argument("--minutes", type=int, default=DEFAULT_WATCHLIST_SCHEDULE_MINUTES, help="Run interval in minutes")
    install_parser.add_argument("--last", type=int, default=DEFAULT_WATCHLIST_SCHEDULE_LAST, help="Fetch the latest N videos per watched channel")
    install_parser.add_argument("--days", type=int, default=DEFAULT_WATCHLIST_SCHEDULE_DAYS, help="Look back this many days")
    install_parser.add_argument("--label", default=DEFAULT_WATCHLIST_SCHEDULE_LABEL, help="LaunchAgent label")
    install_parser.add_argument("--no-load", action="store_true", help="Write the plist without loading it")
    install_parser.set_defaults(handler=run_watchlist_schedule_install)
    status_parser = schedule_subparsers.add_parser("status", help="Show watchlist scheduled sync status")
    status_parser.add_argument("--label", default=DEFAULT_WATCHLIST_SCHEDULE_LABEL, help="LaunchAgent label")
    status_parser.set_defaults(handler=run_watchlist_schedule_status)
    uninstall_parser = schedule_subparsers.add_parser("uninstall", help="Unload and remove the watchlist scheduled sync job")
    uninstall_parser.add_argument("--label", default=DEFAULT_WATCHLIST_SCHEDULE_LABEL, help="LaunchAgent label")
    uninstall_parser.set_defaults(handler=run_watchlist_schedule_uninstall)


def add_blacklist_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add blacklist management subcommands to the root parser."""
    blacklist_parser = subparsers.add_parser("blacklist", help="Manage search blacklist channels")
    blacklist_subparsers = blacklist_parser.add_subparsers(dest="blacklist_command", required=True)
    blacklist_add_parser = blacklist_subparsers.add_parser("add", help="Add one or more channels to the search blacklist")
    add_channel_refs_arg(blacklist_add_parser)
    blacklist_add_parser.set_defaults(handler=run_blacklist_add)
    blacklist_remove_parser = blacklist_subparsers.add_parser("remove", help="Remove one or more channels from the search blacklist")
    add_channel_refs_arg(blacklist_remove_parser)
    blacklist_remove_parser.set_defaults(handler=run_blacklist_remove)


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser("doctor", help="Check runtime and database path")
    doctor_parser.set_defaults(handler=run_doctor)


def add_channel_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--last", type=int, default=10, help="Return the latest N videos")
    parser.add_argument("--days", type=int, help="Return videos uploaded in the last N days")


def add_channel_refs_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "channel_refs",
        nargs="+",
        metavar="channel",
        help="YouTube channel URL, handle, or channel ID; pass multiple as space-separated arguments",
    )
