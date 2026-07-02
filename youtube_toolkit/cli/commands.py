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
    list_recent_watchlist_videos,
    list_watchlist_channels,
    remove_blacklist_channels,
    remove_watchlist_channels,
    sync_watchlist,
)
from ..youtube.search import SearchOptions, search_videos


class DoctorResult(TypedDict):
    yt_dlp_version: str
    database: str


class VideoCommands:
    @staticmethod
    def transcript(args: argparse.Namespace) -> dict[str, Any]:
        return asdict(fetch_transcript(args.video_ref))

    @staticmethod
    def metadata(args: argparse.Namespace) -> dict[str, Any]:
        return YtDlpClient().metadata(args.video_ref)


class ChannelCommands:
    @staticmethod
    def videos(args: argparse.Namespace) -> dict[str, Any]:
        return fetch_channel_videos(args.channel_ref, options=channel_options(args))


class WatchlistCommands:
    @staticmethod
    def add(args: argparse.Namespace) -> list[str]:
        return add_watchlist_channels(args.channel_refs)

    @staticmethod
    def list(args: argparse.Namespace) -> list[str]:
        del args
        return list_watchlist_channels()

    @staticmethod
    def remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
        return remove_watchlist_channels(args.channel_refs)

    @staticmethod
    def sync(args: argparse.Namespace) -> WatchlistSyncResult:
        return sync_watchlist(options=channel_options(args))

    @staticmethod
    def scheduled_sync(args: argparse.Namespace) -> ScheduledSyncResult:
        return scheduled_sync_watchlist(options=channel_options(args))

    @staticmethod
    def recent(args: argparse.Namespace) -> list[dict[str, Any]]:
        return list_recent_watchlist_videos(
            channel_refs=args.channel_refs,
            days=args.days,
        )

    @staticmethod
    def install_schedule(args: argparse.Namespace) -> WatchlistScheduleInstallResult:
        return install_watchlist_schedule(
            options=WatchlistScheduleOptions(
                minutes=args.minutes,
                last=args.last,
                days=args.days,
                label=args.label,
                load=not args.no_load,
            )
        )

    @staticmethod
    def schedule_status(args: argparse.Namespace) -> WatchlistScheduleStatusResult:
        return watchlist_schedule_status(label=args.label)

    @staticmethod
    def uninstall_schedule(args: argparse.Namespace) -> WatchlistScheduleUninstallResult:
        return uninstall_watchlist_schedule(label=args.label)


class BlacklistCommands:
    @staticmethod
    def add(args: argparse.Namespace) -> list[str]:
        return add_blacklist_channels(args.channel_refs)

    @staticmethod
    def remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
        return remove_blacklist_channels(args.channel_refs)


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


def run_doctor(args: argparse.Namespace) -> DoctorResult:
    del args
    import yt_dlp

    return {
        "yt_dlp_version": yt_dlp.version.__version__,
        "database": str(resolve_storage_dir() / DATABASE_FILENAME),
    }


def channel_options(args: argparse.Namespace) -> ChannelOptions:
    return ChannelOptions(last=args.last, days=args.days)
