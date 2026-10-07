"""Small subprocess wrapper around the installed yt-dlp package."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from ..cache import cache_root, read_cache, write_cache
from ..identity import VideoRef, normalize_youtube_url, require_video_id

SEARCH_SORT_PARAMS = {
    "relevance": "EgIQAQ==",  # boolean "true"
    "popularity": "CAMSAhAB",  # sort search results by view count in descending order
}
STT_AUDIO_FORMAT_SELECTOR = "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"
VISION_VIDEO_FORMAT_SELECTOR = "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"
VISION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS = 10 * 60
YTDLP_REQUEST_TIMEOUT_SECONDS = 60
REQUEST_MAX_AGE_SECONDS = 15 * 60
AUDIO_DOWNLOAD_TIMEOUT_SECONDS = 30


class YtDlpClient:
    """Subprocess adapter that owns all direct yt-dlp command construction."""

    def command_prefix(self) -> list[str]:
        """Return the Python-module command prefix used to invoke yt-dlp from this environment."""
        return [sys.executable, "-m", "yt_dlp", "--no-cache-dir"]

    def metadata(self, video_ref: VideoRef) -> dict[str, Any]:
        """Fetch full yt-dlp metadata for one validated YouTube video reference."""
        return self._json(
            [
                *self.command_prefix(),
                "-J",
                "--skip-download",
                "--sponsorblock-mark",
                "default",
                normalize_youtube_url(video_ref),
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
        """Bound STT audio acquisition and clean partial files; the caller owns the returned complete file."""
        video_id = require_video_id(video_ref)
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
            normalize_youtube_url(video_id),
        ]
        downloaded_path: Path | None = None
        try:
            _run(
                command,
                action="audio download",
                timeout_seconds=AUDIO_DOWNLOAD_TIMEOUT_SECONDS,
            )
            candidates = [
                path
                for path in output_stem.parent.glob(f"{output_stem.name}.*")
                if path.is_file()
            ]
            if not candidates:
                raise RuntimeError(
                    f"yt-dlp completed but did not create {output_stem}.*"
                )
            downloaded_path = max(candidates, key=lambda path: path.stat().st_mtime)
            return downloaded_path
        finally:
            # The caller owns the returned audio; this invocation owns all partial files.
            for path in output_stem.parent.glob(f"{output_stem.name}.*"):
                if path.is_file() and path != downloaded_path:
                    path.unlink(missing_ok=True)

    def download_video_for_vision(
        self,
        video_ref: VideoRef,
        output_path: Path,
        *,
        max_bytes: int,
    ) -> Path:
        """Download one bounded temporary video with the vision format preset."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
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
                normalize_youtube_url(video_ref),
            ],
            action="video download for vision",
            timeout_seconds=VISION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS,
        )
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
        result = _run(
            command,
            action="request",
            timeout_seconds=YTDLP_REQUEST_TIMEOUT_SECONDS,
            failure_message=error_message,
        )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("yt-dlp did not return valid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("yt-dlp returned a non-object JSON payload")
        write_cache("requests", key, data)
        return data


def _run(
    command: list[str],
    *,
    action: str,
    timeout_seconds: int,
    failure_message: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded yt-dlp process, translating timeouts, launch errors, and exit codes into RuntimeError."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed yt-dlp executable prefix and argument list.
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"yt-dlp {action} timed out after {timeout_seconds} seconds"
        ) from error
    except OSError as error:
        raise RuntimeError(f"Could not start yt-dlp {action}: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or failure_message or f"yt-dlp {action} failed"
        )
    return result
