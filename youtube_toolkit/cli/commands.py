"""CLI command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, TypedDict

from ..schemas import ChannelRemovalResult, WatchlistSyncResult
from ..storage import DATABASE_FILENAME, resolve_storage_dir
from ..watchlist_schedule import (
    ScheduledSyncResult,
    WatchlistScheduleInstallResult,
    WatchlistScheduleOptions,
    WatchlistScheduleStatusResult,
    WatchlistScheduleUninstallResult,
    install_watchlist_schedule,
    scheduled_sync_watchlist,
    uninstall_watchlist_schedule,
    watchlist_schedule_status,
)
from ..youtube import YtDlpClient
from ..youtube.caption import fetch_transcript
from ..youtube.channels import (
    ChannelOptions,
    add_blacklist_channels,
    add_watchlist_channels,
    fetch_channel_videos,
    remove_blacklist_channels,
    remove_watchlist_channels,
    sync_watchlist,
    watch_new_videos,
)
from ..youtube.search import SearchOptions, search_videos


class DoctorResult(TypedDict):
    yt_dlp_version: str
    database: str


def run_transcript(args: argparse.Namespace) -> dict[str, Any]:
    return asdict(fetch_transcript(args.video_ref))


def run_inspect(args: argparse.Namespace) -> dict[str, Any]:
    return YtDlpClient().metadata(args.video_ref)


def run_search(args: argparse.Namespace) -> list[dict[str, Any]]:
    return search_videos(
        args.query,
        options=SearchOptions(
            limit=args.limit,
            sort=args.sort,
            min_views=args.min_views,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            include_channel=tuple(args.include_channel),
            exclude_channel=tuple(args.exclude_channel),
            include_title=tuple(args.include_title),
            exclude_title=tuple(args.exclude_title),
        ),
    )


def run_channel_videos(args: argparse.Namespace) -> dict[str, Any]:
    return fetch_channel_videos(args.channel_ref, options=channel_options(args))


def run_watchlist_add(args: argparse.Namespace) -> list[str]:
    return add_watchlist_channels(args.channel_refs)


def run_watchlist_remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
    return remove_watchlist_channels(args.channel_refs)


def run_watchlist_sync(args: argparse.Namespace) -> WatchlistSyncResult:
    return sync_watchlist(options=channel_options(args))


def run_watchlist_scheduled_sync(args: argparse.Namespace) -> ScheduledSyncResult:
    return scheduled_sync_watchlist(options=channel_options(args))


def run_watchlist_new(args: argparse.Namespace) -> list[dict[str, Any]]:
    return watch_new_videos(
        channel_refs=args.channel_refs,
        include_all=args.all,
    )


def run_watchlist_schedule_install(args: argparse.Namespace) -> WatchlistScheduleInstallResult:
    return install_watchlist_schedule(
        options=WatchlistScheduleOptions(
            minutes=args.minutes,
            last=args.last,
            days=args.days,
            label=args.label,
            load=not args.no_load,
        )
    )


def run_watchlist_schedule_status(args: argparse.Namespace) -> WatchlistScheduleStatusResult:
    return watchlist_schedule_status(label=args.label)


def run_watchlist_schedule_uninstall(args: argparse.Namespace) -> WatchlistScheduleUninstallResult:
    return uninstall_watchlist_schedule(label=args.label)


def run_blacklist_add(args: argparse.Namespace) -> list[str]:
    return add_blacklist_channels(args.channel_refs)


def run_blacklist_remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
    return remove_blacklist_channels(args.channel_refs)


def run_doctor(_args: argparse.Namespace) -> DoctorResult:
    import yt_dlp

    return {
        "yt_dlp_version": yt_dlp.version.__version__,
        "database": str(resolve_storage_dir() / DATABASE_FILENAME),
    }


def channel_options(args: argparse.Namespace) -> ChannelOptions:
    return ChannelOptions(last=args.last, days=args.days)
