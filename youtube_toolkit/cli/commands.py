"""CLI command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from ..stt import transcribe_audio_to_db
from ..ytdlp import YtDlpClient
from ..ytdlp.caption import fetch_transcript
from ..ytdlp.channels import (
    ChannelOptions,
    add_blacklist_channel,
    add_watchlist_channel,
    fetch_channel_videos,
    remove_blacklist_channel,
    remove_watchlist_channel,
    sync_watchlist,
)
from ..ytdlp.search import SearchOptions, search_videos


def run_transcript(args: argparse.Namespace) -> int:
    document = fetch_transcript(args.video)
    print(json.dumps(asdict(document), indent=2, ensure_ascii=False))
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    metadata = YtDlpClient().metadata(args.video)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def run_audio(args: argparse.Namespace) -> int:
    output_path = YtDlpClient().download_transcription_audio(args.video)
    print(output_path)
    return 0


def run_transcribe_audio(args: argparse.Namespace) -> int:
    payload = transcribe_audio_to_db(args.audio)
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output)
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
    print(json.dumps(entries, indent=2, ensure_ascii=False))
    return 0


def run_channel_videos(args: argparse.Namespace) -> int:
    metadata = fetch_channel_videos(
        args.channel,
        options=ChannelOptions(
            last=args.last,
            days=args.days,
        ),
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


def run_watchlist(args: argparse.Namespace) -> int:
    """Dispatch watchlist add, remove, and sync subcommands."""
    if args.watchlist_command == "add":
        print(add_watchlist_channel(args.channel))
        return 0
    if args.watchlist_command == "remove":
        print(remove_watchlist_channel(args.channel))
        return 0

    fetched_video_ids = sync_watchlist(
        options=ChannelOptions(
            last=args.last,
            days=args.days,
        )
    )
    print(
        "\n".join(fetched_video_ids) if fetched_video_ids else "No missing watched-channel transcripts found.",
    )
    return 0


def run_blacklist(args: argparse.Namespace) -> int:
    """Dispatch blacklist add and remove subcommands."""
    if args.blacklist_command == "add":
        print(add_blacklist_channel(args.channel))
        return 0

    print(remove_blacklist_channel(args.channel))
    return 0


def run_doctor(_args: argparse.Namespace) -> int:
    import yt_dlp

    print(f"yt-dlp import ok: {yt_dlp.version.__version__}")
    return 0
