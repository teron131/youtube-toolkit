"""CLI command handlers."""

from __future__ import annotations

import argparse
import json

from ..audio import AudioOptions, download_transcription_audio, remove_cached_audio_file
from ..caption import TranscriptOptions, fetch_transcript
from ..channels import (
    ChannelOptions,
    add_blacklist_channel,
    add_watchlist_channel,
    fetch_channel_videos,
    format_channel_results,
    sync_watchlist,
)
from ..output import (
    normalize_text_output_path,
    render_plain_text,
    render_transcript_json,
    render_transcript_markdown,
    write_or_print,
)
from ..search import SearchOptions, format_search_results, search_videos
from ..stt import SpeechToTextOptions, transcribe_audio_file
from ..ytdlp_client import YtDlpClient


def run_transcript(args: argparse.Namespace) -> int:
    languages = tuple(args.languages or ["en-orig", "en"])
    document = fetch_transcript(
        args.video,
        options=TranscriptOptions(
            languages=languages,
            source=args.caption_source,
        ),
    )
    output = render_transcript_json(document) if args.format == "json" else render_transcript_markdown(document)
    write_or_print(output, args.output)
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    metadata = YtDlpClient().metadata(args.video)
    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def run_audio(args: argparse.Namespace) -> int:
    output_path = download_transcription_audio(
        args.video,
        options=AudioOptions(output=args.output),
    )
    print(output_path)
    return 0


def run_transcribe_audio(args: argparse.Namespace) -> int:
    output_path = normalize_text_output_path(args.output) if args.output else None
    transcription = transcribe_audio_file(args.audio, options=SpeechToTextOptions())
    output = render_plain_text(transcription.text)
    write_or_print(output, output_path)
    remove_cached_audio_file(args.audio)
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
    if args.json:
        print(
            json.dumps(
                entries,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(format_search_results(entries))
    return 0


def run_channel_videos(args: argparse.Namespace) -> int:
    metadata = fetch_channel_videos(
        args.channel,
        options=ChannelOptions(
            last=args.last,
            days=args.days,
        ),
    )
    if args.json:
        print(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(format_channel_results(metadata))
    return 0


def run_watchlist(args: argparse.Namespace) -> int:
    if args.watchlist_command == "add":
        print(add_watchlist_channel(args.channel))
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
    print(add_blacklist_channel(args.channel))
    return 0


def run_doctor(_args: argparse.Namespace) -> int:
    import yt_dlp

    print(f"yt-dlp import ok: {yt_dlp.version.__version__}")
    return 0
