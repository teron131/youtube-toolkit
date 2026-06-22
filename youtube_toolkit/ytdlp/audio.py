"""Audio extraction workflow for STT model input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..url import extract_video_id
from .client import YtDlpClient

DEFAULT_AUDIO_DIR = Path(".cache/audio")
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_BITRATE = "32k"


@dataclass(frozen=True, slots=True)
class AudioOptions:
    output: Path | None = None


def download_transcription_audio(
    video: str,
    *,
    options: AudioOptions | None = None,
    client: YtDlpClient | None = None,
) -> Path:
    options = options or AudioOptions()
    client = client or YtDlpClient()
    output_path = resolve_audio_output_path(
        video,
        options.output,
    )
    return client.download_transcription_audio(
        video,
        output_path=output_path,
        audio_format=DEFAULT_AUDIO_FORMAT,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        bitrate=DEFAULT_BITRATE,
    )


def resolve_audio_output_path(
    video: str,
    output_path: Path | None,
) -> Path:
    if output_path is not None:
        return normalize_audio_output_path(output_path)

    video_id = extract_video_id(video)
    if video_id is None:
        raise ValueError("Could not find a YouTube video ID")
    return DEFAULT_AUDIO_DIR / f"{video_id}.{DEFAULT_AUDIO_FORMAT}"


def normalize_audio_output_path(output_path: Path) -> Path:
    suffix = f".{DEFAULT_AUDIO_FORMAT}"
    if output_path.suffix == "":
        return output_path.with_suffix(suffix)
    if output_path.suffix.casefold() != suffix:
        raise ValueError(f"audio output must use {suffix}")
    return output_path


def remove_cached_audio_file(audio_path: Path) -> bool:
    if not is_default_audio_cache_path(audio_path):
        return False
    audio_path.unlink(missing_ok=True)
    return True


def is_default_audio_cache_path(audio_path: Path) -> bool:
    try:
        audio_path.resolve().relative_to(DEFAULT_AUDIO_DIR.resolve())
    except ValueError:
        return False
    return True
