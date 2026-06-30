"""CLI command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from ..audio_transcription import transcribe_audio_to_storage
from ..watchlist_schedule import (
    WatchlistScheduleOptions,
    install_watchlist_schedule,
    scheduled_sync_watchlist,
    uninstall_watchlist_schedule,
    watchlist_schedule_status,
)
from ..youtube import YtDlpClient
from ..youtube.caption import fetch_transcript
from ..youtube.channels import (
    ChannelOptions,
    add_blacklist_channel,
    add_watchlist_channel,
    fetch_channel_videos,
    remove_blacklist_channel,
    remove_watchlist_channel,
    sync_watchlist,
    watch_new_videos,
)
from ..youtube.search import SearchOptions, search_videos


def run_transcript(args: argparse.Namespace) -> int:
    document = fetch_transcript(args.video)
    print_json(asdict(document))
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    print_json(YtDlpClient().metadata(args.video))
    return 0


def run_audio(args: argparse.Namespace) -> int:
    print(YtDlpClient().download_transcription_audio(args.video))
    return 0


def run_transcribe_audio(args: argparse.Namespace) -> int:
    payload = transcribe_audio_to_storage(args.audio)
    print_json(payload)
    return 0


def run_search(args: argparse.Namespace) -> int:
    entries = search_videos(
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
    print_json(entries)
    return 0


def run_channel_videos(args: argparse.Namespace) -> int:
    metadata = fetch_channel_videos(
        args.channel,
        options=ChannelOptions(
            last=args.last,
            days=args.days,
        ),
    )
    print_json(metadata)
    return 0


def run_watchlist_add(args: argparse.Namespace) -> int:
    print(add_watchlist_channel(args.channel))
    return 0


def run_watchlist_remove(args: argparse.Namespace) -> int:
    print(remove_watchlist_channel(args.channel))
    return 0


def run_watchlist_sync(args: argparse.Namespace) -> int:
    print_json(sync_watchlist(options=channel_options_from_args(args)))
    return 0


def run_watchlist_scheduled_sync(args: argparse.Namespace) -> int:
    print_json(scheduled_sync_watchlist(options=channel_options_from_args(args)))
    return 0


def run_watchlist_new(args: argparse.Namespace) -> int:
    print_json(
        watch_new_videos(
            channels=args.channel,
            include_all=args.all,
        )
    )
    return 0


def run_watchlist_schedule_install(args: argparse.Namespace) -> int:
    print_json(
        install_watchlist_schedule(
            options=WatchlistScheduleOptions(
                minutes=args.minutes,
                last=args.last,
                days=args.days,
                label=args.label,
                load=not args.no_load,
            )
        )
    )
    return 0


def run_watchlist_schedule_status(args: argparse.Namespace) -> int:
    print_json(watchlist_schedule_status(label=args.label))
    return 0


def run_watchlist_schedule_uninstall(args: argparse.Namespace) -> int:
    print_json(uninstall_watchlist_schedule(label=args.label))
    return 0


def run_blacklist_add(args: argparse.Namespace) -> int:
    print(add_blacklist_channel(args.channel))
    return 0


def run_blacklist_remove(args: argparse.Namespace) -> int:
    print(remove_blacklist_channel(args.channel))
    return 0


def run_doctor(_args: argparse.Namespace) -> int:
    import yt_dlp

    print(f"yt-dlp import ok: {yt_dlp.version.__version__}")
    return 0


def channel_options_from_args(args: argparse.Namespace) -> ChannelOptions:
    return ChannelOptions(
        last=args.last,
        days=args.days,
    )


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
