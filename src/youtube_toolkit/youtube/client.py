"""Small subprocess wrapper around the installed yt-dlp package."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlencode, urlparse
from uuid import uuid4

from ..artifacts import artifact_root, write_artifact
from ..execution import retry_transient
from ..identity import VideoRef, normalize_youtube_url, require_video_id

SEARCH_SORT_PARAMS = {
    "relevance": "EgIQAQ==",  # boolean "true"
    "popularity": "CAMSAhAB",  # sort search results by view count in descending order
}
STT_AUDIO_FORMAT_SELECTOR = "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"
VISION_VIDEO_FORMAT_SELECTOR = "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"
# A metadata, search, or listing request that takes over a minute is stuck.
YTDLP_REQUEST_TIMEOUT_SECONDS = 60
# Keep comment pagination bounded even when a caller overrides the normal sample size.
MAX_COMMENT_LIMIT = 1000
# A download can simply be slow: YouTube was measured as low as 0.5 MB/s, and a two-hour 720p video is about 300 MB.
YTDLP_DOWNLOAD_TIMEOUT_SECONDS = 10 * 60
# YouTube refuses requests in short bursts, so a refused process is retried three times, after 5, 10, and 15 seconds.
FORBIDDEN_RETRIES = 3
FORBIDDEN_BACKOFF_SECONDS = 5.0
# YouTube marks machine-translated caption tracks with a tlang query parameter.
TRANSLATED_TRACK_URL = re.compile(r"[?&]tlang=")
# yt-dlp runs as a module of this interpreter, so the installed package version is the one used.
YTDLP_COMMAND = (sys.executable, "-m", "yt_dlp", "--no-cache-dir")


class YtDlpForbidden(RuntimeError):
    """Raised when YouTube refuses a yt-dlp process with HTTP 403, which usually clears on a later attempt."""


class YtDlpClient:
    """Subprocess adapter that owns all direct yt-dlp command construction."""

    def metadata(self, video_ref: VideoRef) -> dict[str, Any]:
        """Fetch full yt-dlp metadata for one validated YouTube video reference.

        Machine-translated caption tracks are dropped before the response is returned or archived.
        """
        return self._json(
            [
                *YTDLP_COMMAND,
                "-J",
                "--skip-download",
                "--sponsorblock-mark",
                "default",
                normalize_youtube_url(video_ref),
            ],
            namespace="metadata",
            identifier=require_video_id(video_ref),
            error_message="yt-dlp metadata request failed",
            prepare=_drop_translated_captions,
        )

    def comments(self, video_ref: VideoRef, *, limit: int) -> dict[str, Any]:
        """Fetch up to limit Top comments without replies or media; the comment workflow owns the lean archive."""
        if not 1 <= limit <= MAX_COMMENT_LIMIT:
            raise ValueError(f"--limit must be between 1 and {MAX_COMMENT_LIMIT}")
        return _request_json(
            [
                *YTDLP_COMMAND,
                "--ignore-config",
                "-J",
                "--skip-download",
                "--no-playlist",
                "--write-comments",
                "--extractor-args",
                f"youtube:comment_sort=top;max_comments={limit},all,0,0,1",
                normalize_youtube_url(video_ref),
            ],
            error_message="yt-dlp comments request failed",
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
                *YTDLP_COMMAND,
                "--quiet",
                "--no-warnings",
                "--dump-single-json",
                "--flat-playlist",
                "--playlist-end",
                str(fetch),
                search_url,
            ],
            namespace="searches",
            identifier=query,
            error_message="yt-dlp search query failed",
        )

    def channel_videos(
        self,
        channel_url: str,
        *,
        last: int | None,
    ) -> dict[str, Any]:
        """Fetch a flat channel videos listing through yt-dlp with optional playlist truncation."""
        command = [
            *YTDLP_COMMAND,
            "--quiet",
            "--no-warnings",
            "--dump-single-json",
            "--flat-playlist",
        ]
        if last is not None:
            command.extend(
                [
                    "--playlist-end",
                    str(last),
                ]
            )
        command.append(channel_url)
        channel_path = urlparse(channel_url).path.rstrip("/").removesuffix("/videos")
        return self._json(
            command,
            namespace="channels",
            identifier=unquote(channel_path.rsplit("/", 1)[-1]),
            error_message="yt-dlp channel query failed",
        )

    def download_transcription_audio(self, video_ref: VideoRef) -> Path:
        """Bound STT audio acquisition and clean partial files; the caller owns the returned complete file."""
        video_id = require_video_id(video_ref)
        output_stem = artifact_root() / "audio" / f"{video_id}-{uuid4().hex}"
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *YTDLP_COMMAND,
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
                timeout_seconds=YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
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
                *YTDLP_COMMAND,
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
            timeout_seconds=YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
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
        namespace: str,
        identifier: str,
        error_message: str,
        prepare: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Fetch a fresh yt-dlp response, apply any response policy, and archive the result without consulting previous requests."""
        request = {"arguments": command[len(YTDLP_COMMAND) :]}
        data = _request_json(command, error_message=error_message)
        if prepare is not None:
            data = prepare(data)
        write_artifact(namespace, identifier, request, data)
        return data


def _request_json(command: list[str], *, error_message: str) -> dict[str, Any]:
    """Run a bounded metadata request and reject malformed JSON before a workflow can archive it."""
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
    return data


def _drop_translated_captions(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only original automatic-caption tracks.

    YouTube offers each recognized track machine-translated into every supported language, which can make up over 90% of a metadata payload.
    """
    captions = metadata.get("automatic_captions")
    if not isinstance(captions, dict):
        return metadata
    originals = {
        language: kept
        for language, tracks in captions.items()
        if (
            kept := [
                track
                for track in tracks
                if not TRANSLATED_TRACK_URL.search(track.get("url") or "")
            ]
        )
    }
    return {**metadata, "automatic_captions": originals}


def _run(
    command: list[str],
    *,
    action: str,
    timeout_seconds: int,
    failure_message: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded yt-dlp process, translating timeouts, launch errors, and exit codes into RuntimeError.

    A process YouTube refuses with HTTP 403 is rerun after a pause, because a fresh run requests new media URLs.
    """

    def attempt() -> subprocess.CompletedProcess[str]:
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
            message = (
                result.stderr.strip() or failure_message or f"yt-dlp {action} failed"
            )
            if "HTTP Error 403" in result.stderr:
                raise YtDlpForbidden(message)
            raise RuntimeError(message)
        return result

    return retry_transient(
        attempt,
        attempts=FORBIDDEN_RETRIES + 1,
        backoff_seconds=FORBIDDEN_BACKOFF_SECONDS,
        retryable_errors=(YtDlpForbidden,),
    )
