"""The youtube command's arguments; each subcommand stores its handler under `handler`, which the entrypoint calls with the parsed namespace."""

from __future__ import annotations

import argparse

from ..config import (
    DEFAULT_LOCAL_STT_MODEL,
    DEFAULT_VISION_COLUMNS,
    DEFAULT_VISION_MAX_FRAMES,
    DEFAULT_VISION_MIN_GAP_SECONDS,
    DEFAULT_VISION_RESOLUTION,
    DEFAULT_VISION_ROWS,
    MAX_VISION_COLUMNS,
    MAX_VISION_FRAMES,
    MAX_VISION_ROWS,
    VISION_RESOLUTIONS,
)
from ..export import DEFAULT_DOWNLOAD_QUALITY, DOWNLOAD_HEIGHTS
from ..search import DEFAULT_CHANNEL_SEARCH_LIMIT, DEFAULT_YOUTUBE_SEARCH_LIMIT
from ..youtube import DEFAULT_COMMENT_LIMIT, DEFAULT_LAST_VIDEOS, MAX_COMMENT_LIMIT
from . import commands
from .health import run_doctor
from .setup import run_setup


class YoutubeArgumentParser(argparse.ArgumentParser):
    """Untyped string arguments are stripped and lose one layer of matching quotes, which shell-passed values can retain."""

    def add_argument(self, *args, **kwargs):
        action = kwargs.get("action", "store")
        if "type" not in kwargs and action in {"store", "append", "extend"}:
            kwargs["type"] = normalize_arg_string
        return super().add_argument(*args, **kwargs)


def normalize_arg_string(value: str) -> str:
    candidate = str(value).strip()
    if (
        len(candidate) >= 2
        and candidate[0] == candidate[-1]
        and candidate[0] in {"'", '"'}
    ):
        return candidate[1:-1].strip()
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = YoutubeArgumentParser(
        prog="youtube", description="Transcript-first YouTube toolkit."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=YoutubeArgumentParser
    )
    _add_video_command(subparsers)
    _add_search_command(subparsers)
    _add_channel_command(subparsers)
    _add_setup_command(subparsers)
    _add_stt_check_command(subparsers)
    _add_doctor_command(subparsers)
    return parser


def _add_video_command(subparsers: argparse._SubParsersAction) -> None:
    video_parser = subparsers.add_parser(
        "video",
        help="Fetch a video's metadata, transcript, comments, or screenshots, or download it",
        description="Each part flag adds that part to the output, keyed by part; with none, the output holds the metadata alone. Options for a part require its flag; --stt and --translate also work with --summary.",
    )
    video_parser.add_argument(
        "video_ref", metavar="url", help="YouTube video URL or ID"
    )
    transcript = video_parser.add_argument_group("transcript")
    transcript.add_argument(
        "--transcript",
        action="store_true",
        help="Captions, or speech-to-text when configured and there are none",
    )
    transcript.add_argument(
        "--timestamps",
        action="store_true",
        default=None,
        help="Timed transcript segments instead of one text string",
    )
    transcript.add_argument(
        "--stt",
        action="store_true",
        default=None,
        help="Transcribe the audio with speech-to-text even when captions exist",
    )
    transcript.add_argument(
        "--refine",
        action="store_true",
        default=None,
        help="Clean up the transcript with the model saved by `youtube setup --llm`: each caption line corrected on its own timestamps with --timestamps, otherwise paragraphs of complete sentences under the video's chapter titles",
    )
    transcript.add_argument(
        "--translate",
        nargs="?",
        const="en",
        metavar="LANGUAGE",
        help="Translate the transcript into English, or into LANGUAGE in any wording the model understands, such as zh-TW or Japanese, with the model saved by `youtube setup --llm`: each caption line translated on its own timestamps with --timestamps, otherwise paragraphs of complete sentences; a transcript whose language code matches comes back as it is unless --refine is given",
    )
    transcript.add_argument(
        "--export",
        action="store_true",
        default=None,
        help="Write metadata and transcript to a title-and-channel .txt file in the working directory; with --vision, a .md file beside the collages instead",
    )
    summary = video_parser.add_argument_group("summary")
    summary.add_argument(
        "--summary",
        nargs="?",
        const="",
        metavar="LANGUAGE",
        help="A Markdown summary of the transcript by the model saved by `youtube setup --llm`, written in LANGUAGE, else the --translate language, else the transcript's own; the transcript itself is printed only with --transcript",
    )
    vision = video_parser.add_argument_group("vision")
    vision.add_argument(
        "--vision",
        action="store_true",
        help="Timestamped screenshot collages, ranked against the transcript when a model is set up",
    )
    vision.add_argument(
        "--max-frames",
        type=int,
        help=f"Maximum output screenshots (1-{MAX_VISION_FRAMES}; default: {DEFAULT_VISION_MAX_FRAMES}); candidate coverage is independent",
    )
    vision.add_argument(
        "--rows",
        type=int,
        metavar="N",
        help=f"Rows of screenshots on each collage page (1-{MAX_VISION_ROWS}; default: {DEFAULT_VISION_ROWS})",
    )
    vision.add_argument(
        "--columns",
        type=int,
        metavar="N",
        help=f"Screenshots in each collage row (1-{MAX_VISION_COLUMNS}; default: {DEFAULT_VISION_COLUMNS})",
    )
    vision.add_argument(
        "--resolution",
        choices=list(VISION_RESOLUTIONS),
        help=f"Screenshot resolution (default: {DEFAULT_VISION_RESOLUTION}); 360p saves image tokens but loses small text and digits, and 720p decodes the picked frames again from the 720p download",
    )
    vision.add_argument(
        "--min-similarity",
        type=float,
        help="Minimum raw cosine for semantic matches (default: the cutoff measured for the ranking embedding model)",
    )
    vision.add_argument(
        "--min-gap",
        type=float,
        help=f"Minimum seconds between retained frames (default: {DEFAULT_VISION_MIN_GAP_SECONDS})",
    )
    vision.add_argument(
        "--local-only",
        action="store_true",
        default=None,
        help="Skip the decision model, Gemini, and speech-to-text: rank against captions with the local embedding model when installed, otherwise pick evenly spaced frames",
    )
    vision.add_argument(
        "--auto-clean",
        action=argparse.BooleanOptionalAction,
        help="Delete the downloaded video and candidate frames after processing (default: on)",
    )
    download = video_parser.add_argument_group("download")
    download.add_argument(
        "--download",
        nargs="?",
        const="video",
        choices=("video", "audio"),
        help="Save the video with its sound as an mp4, or with audio the sound alone as an m4a, to the working directory under the title, channel, quality, and range; a file already there for the same request comes back as it is",
    )
    download.add_argument(
        "--quality",
        choices=list(DOWNLOAD_HEIGHTS),
        help=f"Largest video resolution (default: {DEFAULT_DOWNLOAD_QUALITY}); the sound is the same at every level",
    )
    download.add_argument(
        "--start",
        metavar="TIME",
        help="Download from this time, as seconds, MM:SS, or HH:MM:SS; a video opens at the keyframe at or before it, up to a few seconds early",
    )
    download.add_argument(
        "--end",
        metavar="TIME",
        help="Download up to this time (default: the end of the video)",
    )
    comments = video_parser.add_argument_group("comments")
    comments.add_argument(
        "--comments",
        nargs="?",
        const="",
        metavar="QUERY",
        type=normalize_arg_string,
        help="Top-level comments containing QUERY as a case-insensitive literal; without QUERY, the whole sample. Results alternate between YouTube's Top order and highest likes; replies are excluded.",
    )
    comments.add_argument(
        "--comment-limit",
        type=int,
        help=f"Top-level comments to fetch before filtering (1-{MAX_COMMENT_LIMIT}; default: {DEFAULT_COMMENT_LIMIT})",
    )
    metadata = video_parser.add_argument_group("metadata")
    metadata.add_argument(
        "--metadata",
        action="store_true",
        help="Video facts, including the description and chapters",
    )
    video_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Redo what an earlier run left: replace an existing transcript export, pick screenshots afresh instead of reusing earlier picks, or download again instead of returning the file already there; requires --export, --vision, or --download",
    )
    video_parser.set_defaults(handler=commands.video)


def _add_search_command(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser(
        "search",
        help="Search videos or transcript chunks across YouTube or within channels",
    )
    search_parser.add_argument(
        "queries",
        nargs="*",
        metavar="query",
        help="Search query; give several to interleave their results",
    )
    search_parser.add_argument(
        "--semantic",
        choices=("chunk", "video"),
        help="Rank caption chunks by meaning with embeddings, and return the chunks or their videos",
    )
    search_parser.add_argument(
        "--in-channels",
        dest="channels",
        nargs="+",
        action="extend",
        default=[],
        metavar="CHANNEL",
        help="Search only these channels' uploads; URLs, handles, or IDs",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=f"Maximum results (default: {DEFAULT_YOUTUBE_SEARCH_LIMIT}, or {DEFAULT_CHANNEL_SEARCH_LIMIT} inside channels)",
    )
    search_parser.add_argument(
        "--sort",
        choices=("relevance", "recency", "popularity"),
        default=None,
        help="Result order (default: relevance, or recency inside channels without --semantic); a search across YouTube without --semantic cannot sort by recency",
    )
    search_parser.add_argument(
        "--min-views",
        type=int,
        default=10_000,
        metavar="N",
        help="Only include videos with at least N views (default: %(default)s)",
    )
    search_parser.add_argument(
        "--min-duration",
        default="3:00",
        metavar="TIME",
        help="Only include videos at least this long, as seconds, MM:SS, or HH:MM:SS (default: %(default)s)",
    )
    search_parser.add_argument(
        "--max-duration",
        default="8:00:00",
        metavar="TIME",
        help="Only include videos at most this long (default: %(default)s)",
    )
    search_parser.add_argument(
        "--uploaded-after",
        default=None,
        help="Only include videos uploaded on or after YYYY-MM-DD",
    )
    search_parser.add_argument(
        "--days",
        dest="uploaded_within_days",
        type=int,
        metavar="N",
        default=None,
        help="Only include videos uploaded within the last N days",
    )
    search_parser.add_argument(
        "--include-channel-names",
        dest="include_channel",
        nargs="+",
        action="extend",
        default=[],
        metavar="TEXT",
        help="Keep videos whose channel name or handle contains every TEXT",
    )
    search_parser.add_argument(
        "--exclude-channel-names",
        dest="exclude_channel",
        nargs="+",
        action="extend",
        default=[],
        metavar="TEXT",
        help="Drop videos whose channel name or handle contains any TEXT",
    )
    search_parser.add_argument(
        "--include-titles",
        dest="include_title",
        nargs="+",
        action="extend",
        default=[],
        metavar="TEXT",
        help="Keep videos whose title contains every TEXT",
    )
    search_parser.add_argument(
        "--exclude-titles",
        dest="exclude_title",
        nargs="+",
        action="extend",
        default=[],
        metavar="TEXT",
        help="Drop videos whose title contains any TEXT",
    )
    search_parser.set_defaults(handler=commands.search)


def _add_channel_command(subparsers: argparse._SubParsersAction) -> None:
    channel_parser = subparsers.add_parser(
        "channel", help="List recent videos from a YouTube channel"
    )
    channel_parser.add_argument(
        "channel_ref",
        metavar="channel",
        help="YouTube channel URL, handle, or channel ID",
    )
    channel_parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST_VIDEOS,
        help="Return the latest N videos",
    )
    channel_parser.add_argument(
        "--days", type=int, help="Return videos uploaded in the last N days"
    )
    channel_parser.set_defaults(handler=commands.channel)


def _add_setup_command(subparsers: argparse._SubParsersAction) -> None:
    setup_parser = subparsers.add_parser(
        "setup", help="Save API keys and settings for YouTube Toolkit only"
    )
    setup_parser.add_argument(
        "--embeddings", action="store_true", help="Configure embedding settings only"
    )
    setup_parser.add_argument(
        "--decisions",
        action="store_true",
        help="Configure decision model settings only",
    )
    setup_parser.add_argument(
        "--stt", action="store_true", help="Configure speech-to-text settings only"
    )
    setup_parser.add_argument(
        "--llm",
        action="store_true",
        help="Configure LLM settings only",
    )
    setup_parser.set_defaults(handler=run_setup)


def _add_stt_check_command(subparsers: argparse._SubParsersAction) -> None:
    check_parser = subparsers.add_parser(
        "stt-check",
        help=f"Report the speech-to-text setup; download the Hugging Face ASR model named by STT_LOCAL_MODEL (default: {DEFAULT_LOCAL_STT_MODEL}) and check that it runs",
    )
    check_parser.add_argument(
        "--cloud-test",
        action="store_true",
        help="Also send a two-second test recording to the cloud provider (a real request, billed by the provider)",
    )
    check_parser.set_defaults(handler=commands.stt_check)


def _add_doctor_command(subparsers: argparse._SubParsersAction) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check runtime, artifact path, and optional capabilities"
    )
    doctor_parser.set_defaults(handler=run_doctor)
