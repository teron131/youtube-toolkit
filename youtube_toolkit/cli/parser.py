"""Argument parser construction for the youtube command."""

from __future__ import annotations

import argparse
from pathlib import Path

from .commands import (
    run_audio,
    run_blacklist,
    run_channel_videos,
    run_doctor,
    run_inspect,
    run_search,
    run_transcribe_audio,
    run_transcript,
    run_watchlist,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="youtube", description="Transcript-first YouTube toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_transcript_parser(subparsers)
    add_inspect_parser(subparsers)
    add_audio_parser(subparsers)
    add_transcribe_audio_parser(subparsers)
    add_search_parser(subparsers)
    add_channel_videos_parser(subparsers)
    add_watchlist_parser(subparsers)
    add_blacklist_parser(subparsers)
    add_doctor_parser(subparsers)
    return parser


def add_transcript_parser(subparsers: argparse._SubParsersAction) -> None:
    transcript_parser = subparsers.add_parser("transcript", help="Fetch a normalized transcript")
    transcript_parser.add_argument("video", help="YouTube video URL or ID")
    transcript_parser.set_defaults(handler=run_transcript)


def add_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    inspect_parser = subparsers.add_parser("inspect", help="Fetch video metadata JSON")
    inspect_parser.add_argument("video", help="YouTube video URL or ID")
    inspect_parser.set_defaults(handler=run_inspect)


def add_audio_parser(subparsers: argparse._SubParsersAction) -> None:
    audio_parser = subparsers.add_parser("audio", help="Download provider-compatible audio for STT")
    audio_parser.add_argument("video", help="YouTube video URL or ID")
    audio_parser.set_defaults(handler=run_audio)


def add_transcribe_audio_parser(subparsers: argparse._SubParsersAction) -> None:
    transcribe_parser = subparsers.add_parser("transcribe-audio", help="Store a local audio transcript and return JSON")
    transcribe_parser.add_argument("audio", type=Path, help="Local audio file named with a YouTube video ID")
    transcribe_parser.set_defaults(handler=run_transcribe_audio)


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
    channel_parser.add_argument("channel", help="YouTube channel URL, handle, or channel ID")
    add_channel_window_args(channel_parser)
    channel_parser.set_defaults(handler=run_channel_videos)


def add_watchlist_parser(subparsers: argparse._SubParsersAction) -> None:
    watchlist_parser = subparsers.add_parser("watchlist", help="Manage or sync watchlist channels")
    watchlist_subparsers = watchlist_parser.add_subparsers(dest="watchlist_command", required=True)
    watchlist_add_parser = watchlist_subparsers.add_parser("add", help="Add a channel to the watchlist")
    watchlist_add_parser.add_argument("channel", help="YouTube channel URL, handle, or channel ID")
    watchlist_sync_parser = watchlist_subparsers.add_parser("sync", help="Sync watchlist videos and transcripts")
    add_channel_window_args(watchlist_sync_parser)
    watchlist_parser.set_defaults(handler=run_watchlist)


def add_blacklist_parser(subparsers: argparse._SubParsersAction) -> None:
    blacklist_parser = subparsers.add_parser("blacklist", help="Manage search blacklist channels")
    blacklist_subparsers = blacklist_parser.add_subparsers(dest="blacklist_command", required=True)
    blacklist_add_parser = blacklist_subparsers.add_parser("add", help="Add a channel to the search blacklist")
    blacklist_add_parser.add_argument("channel", help="YouTube channel URL, handle, or channel ID")
    blacklist_parser.set_defaults(handler=run_blacklist)


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser("doctor", help="Check local runtime")
    doctor_parser.set_defaults(handler=run_doctor)


def add_channel_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--last", type=int, default=10, help="Return the latest N videos")
    parser.add_argument("--days", type=int, help="Return videos uploaded in the last N days")
