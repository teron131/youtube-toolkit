"""Small subprocess wrapper around the installed yt-dlp package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .url import ALLOWED_VIDEO_HOSTS, normalize_youtube_url, validate_url

SEARCH_SORT_PARAMS = {
    "relevance": "EgIQAQ==",
    "views": "CAMSAhAB",
}


class YtDlpClient:
    def command_prefix(self) -> list[str]:
        return [sys.executable, "-m", "yt_dlp"]

    def metadata(self, video: str) -> dict[str, Any]:
        url = normalize_youtube_url(video)
        validate_url(url, allowed_hosts=ALLOWED_VIDEO_HOSTS)
        return self._json(
            [
                *self.command_prefix(),
                "-J",
                "--skip-download",
                "--sponsorblock-mark",
                "default",
                url,
            ],
            error_message="yt-dlp metadata request failed",
        )

    def search(
        self,
        query: str,
        *,
        fetch: int,
        sort: str,
    ) -> dict[str, Any]:
        search_url = f"https://www.youtube.com/results?{urlencode({'search_query': query, 'sp': SEARCH_SORT_PARAMS[sort]})}"
        return self._json(
            [
                *self.command_prefix(),
                "--quiet",
                "--no-warnings",
                "--dump-single-json",
                "--flat-playlist",
                "--playlist-end",
                str(fetch),
                search_url,
            ],
            error_message="yt-dlp search query failed",
        )

    def channel_videos(
        self,
        channel_url: str,
        *,
        last: int,
    ) -> dict[str, Any]:
        return self._json(
            [
                *self.command_prefix(),
                "--quiet",
                "--no-warnings",
                "--dump-single-json",
                "--playlist-end",
                str(last),
                channel_url,
            ],
            error_message="yt-dlp channel query failed",
        )

    def download_transcription_audio(
        self,
        video: str,
        *,
        output_path: Path,
        audio_format: str,
        sample_rate: int,
        channels: int,
        bitrate: str,
    ) -> Path:
        url = normalize_youtube_url(video)
        validate_url(url, allowed_hosts=ALLOWED_VIDEO_HOSTS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *self.command_prefix(),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--extract-audio",
            "--audio-format",
            audio_format,
            "--sponsorblock-remove",
            "default",
            "--postprocessor-args",
            build_ffmpeg_audio_args(
                sample_rate=sample_rate,
                channels=channels,
                bitrate=bitrate,
            ),
            "--output",
            str(output_path.with_suffix(".%(ext)s")),
            url,
        ]
        result = subprocess.run(  # noqa: S603 - command is assembled from a fixed yt-dlp executable prefix and argument list.
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "yt-dlp audio download failed"
            raise RuntimeError(detail)
        if not output_path.exists():
            raise RuntimeError(f"yt-dlp completed but did not create {output_path}")
        return output_path

    def _json(
        self,
        command: list[str],
        *,
        error_message: str,
    ) -> dict[str, Any]:
        result = subprocess.run(  # noqa: S603 - command is assembled from a fixed yt-dlp executable prefix and argument list.
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or error_message
            raise RuntimeError(detail)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("yt-dlp did not return valid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("yt-dlp returned a non-object JSON payload")
        return data


def build_ffmpeg_audio_args(
    *,
    sample_rate: int,
    channels: int,
    bitrate: str,
) -> str:
    return f"ffmpeg:-ac {channels} -ar {sample_rate} -b:a {bitrate}"
