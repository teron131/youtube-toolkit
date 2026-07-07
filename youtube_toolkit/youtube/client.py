"""Small subprocess wrapper around the installed yt-dlp package."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from ..identity import ALLOWED_VIDEO_HOSTS, VideoRef, normalize_youtube_url, require_video_id, validate_url

SEARCH_SORT_PARAMS = {
    "relevance": "EgIQAQ==",  # boolean "true"
    "popularity": "CAMSAhAB",  # sort search results by view count in descending order
}
DEFAULT_AUDIO_DIR = Path(".cache/audio")
STT_AUDIO_FORMAT_SELECTOR = "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"


class YtDlpClient:
    def command_prefix(self) -> list[str]:
        return [sys.executable, "-m", "yt_dlp"]

    def metadata(self, video_ref: VideoRef) -> dict[str, Any]:
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
        video_id = require_video_id(video_ref)
        url = normalize_youtube_url(video_ref)
        validate_url(url, allowed_hosts=ALLOWED_VIDEO_HOSTS)
        output_stem = DEFAULT_AUDIO_DIR / f"{video_id}-{uuid4().hex}"
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
        candidates = [path for path in output_stem.parent.glob(f"{output_stem.name}.*") if path.is_file()]
        if candidates:
            return max(
                candidates,
                key=lambda path: path.stat().st_mtime,
            )
        raise RuntimeError(f"yt-dlp completed but did not create {output_stem}.*")

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
