"""Small subprocess wrapper around the installed yt-dlp package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from ..cache import REQUEST_MAX_AGE_SECONDS, cache_root, read_cache, write_cache
from ..identity import (
    ALLOWED_VIDEO_HOSTS,
    VideoRef,
    normalize_youtube_url,
    require_video_id,
    validate_url,
)

SEARCH_SORT_PARAMS = {
    "relevance": "EgIQAQ==",  # boolean "true"
    "popularity": "CAMSAhAB",  # sort search results by view count in descending order
}
STT_AUDIO_FORMAT_SELECTOR = "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"
# Budget one quarter of the host for each job while capping low-resolution codec parallelism.
VISION_FFMPEG_THREADS = max(1, min(4, (os.process_cpu_count() or 1) // 4))
VISION_VIDEO_FORMAT_SELECTOR = "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"
VISION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS = 10 * 60
YTDLP_REQUEST_TIMEOUT_SECONDS = 60


class YtDlpClient:
    """Subprocess adapter that owns all direct yt-dlp command construction."""

    def command_prefix(self) -> list[str]:
        """Return the Python-module command prefix used to invoke yt-dlp from this environment."""
        return [sys.executable, "-m", "yt_dlp", "--no-cache-dir"]

    def metadata(self, video_ref: VideoRef) -> dict[str, Any]:
        """Fetch full yt-dlp metadata for one validated YouTube video reference."""
        url = normalize_youtube_url(video_ref)
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
        """Fetch a flat YouTube search result page through yt-dlp."""
        search_params = {
            "search_query": query,
            "sp": SEARCH_SORT_PARAMS[sort],
        }
        search_url = f"https://www.youtube.com/results?{urlencode(search_params)}"
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
        last: int | None,
        flat: bool = True,
    ) -> dict[str, Any]:
        """Fetch a channel videos listing through yt-dlp with optional playlist truncation."""
        command = [
            *self.command_prefix(),
            "--quiet",
            "--no-warnings",
            "--dump-single-json",
        ]
        if flat:
            command.append("--flat-playlist")
        if last is not None:
            command.extend(
                [
                    "--playlist-end",
                    str(last),
                ]
            )
        command.append(channel_url)
        return self._json(
            command,
            error_message="yt-dlp channel query failed",
        )

    def download_transcription_audio(self, video_ref: VideoRef) -> Path:
        """Download one temporary audio file for STT and return its created path."""
        video_id = require_video_id(video_ref)
        url = normalize_youtube_url(video_ref)
        validate_url(url, allowed_hosts=ALLOWED_VIDEO_HOSTS)
        output_stem = cache_root() / "audio" / f"{video_id}-{uuid4().hex}"
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *self.command_prefix(),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--format",
            STT_AUDIO_FORMAT_SELECTOR,
            "--output",
            str(output_stem.with_suffix(".%(ext)s")),
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
        candidates = [
            path
            for path in output_stem.parent.glob(f"{output_stem.name}.*")
            if path.is_file()
        ]
        if candidates:
            return max(
                candidates,
                key=lambda path: path.stat().st_mtime,
            )
        raise RuntimeError(f"yt-dlp completed but did not create {output_stem}.*")

    def download_video_for_vision(
        self,
        video_ref: VideoRef,
        output_path: Path,
        *,
        max_bytes: int,
    ) -> Path:
        """Download one bounded temporary video with the vision format preset."""
        url = normalize_youtube_url(video_ref)
        validate_url(url, allowed_hosts=ALLOWED_VIDEO_HOSTS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(  # noqa: S603 - command is assembled from a fixed yt-dlp executable prefix and argument list.
                [
                    *self.command_prefix(),
                    "--no-playlist",
                    "--quiet",
                    "--no-warnings",
                    "--concurrent-fragments",
                    "1",
                    "--max-filesize",
                    str(max_bytes),
                    "--format",
                    VISION_VIDEO_FORMAT_SELECTOR,
                    "--output",
                    str(output_path),
                    url,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=VISION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"yt-dlp video download for vision timed out after {VISION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS} seconds"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Could not start yt-dlp video download for vision: {error}"
            ) from error
        if result.returncode != 0:
            detail = result.stderr.strip() or "yt-dlp video download for vision failed"
            raise RuntimeError(detail)
        if not output_path.exists():
            raise RuntimeError(f"yt-dlp completed but did not create {output_path}")
        if output_path.stat().st_size > max_bytes:
            output_path.unlink()
            raise RuntimeError(
                f"yt-dlp video download for vision exceeded {max_bytes} bytes"
            )
        return output_path

    def _json(
        self,
        command: list[str],
        *,
        error_message: str,
    ) -> dict[str, Any]:
        """Reuse recent metadata/listings locally; expired signed caption URLs are fetched again after 15 minutes."""
        key = {"arguments": command[len(self.command_prefix()) :]}
        cached = read_cache("requests", key, max_age=REQUEST_MAX_AGE_SECONDS)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(  # noqa: S603 - command is assembled from a fixed yt-dlp executable prefix and argument list.
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=YTDLP_REQUEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"yt-dlp request timed out after {YTDLP_REQUEST_TIMEOUT_SECONDS} seconds"
            ) from error
        except OSError as error:
            raise RuntimeError(f"Could not start yt-dlp request: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or error_message
            raise RuntimeError(detail)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("yt-dlp did not return valid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("yt-dlp returned a non-object JSON payload")
        write_cache("requests", key, data)
        return data
