"""YouTube URL normalization and validation."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

ALLOWED_VIDEO_HOSTS = {
    "youtu.be",
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
ALLOWED_CAPTION_HOST_SUFFIXES = ("youtube.com",)
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
VIDEO_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")
VIDEO_ID_URL_PATTERNS = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})"),
)


def validate_url(
    url: str,
    *,
    allowed_hosts: set[str] | tuple[str, ...],
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or 'missing'}")

    hostname = parsed.hostname or ""
    if hostname in allowed_hosts:
        return
    if any(hostname.endswith(f".{suffix}") for suffix in allowed_hosts):
        return
    raise ValueError(f"Unsupported URL host: {hostname or 'missing'}")


def extract_video_id(value: str) -> str | None:
    candidate = value.strip()
    if VIDEO_ID_PATTERN.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)
    path_parts = [part for part in parsed.path.split("/") if part]

    if hostname == "youtu.be" and path_parts and VIDEO_ID_PATTERN.fullmatch(path_parts[0]):
        return path_parts[0]

    if hostname in ALLOWED_VIDEO_HOSTS:
        query_video_id = query.get("v", [""])[0]
        if VIDEO_ID_PATTERN.fullmatch(query_video_id):
            return query_video_id

        normalized_path = f"/{'/'.join(path_parts)}" if path_parts else ""
        for prefix in VIDEO_PATH_PREFIXES:
            if normalized_path.startswith(prefix):
                path_video_id = normalized_path[len(prefix) :].split("/", 1)[0]
                if VIDEO_ID_PATTERN.fullmatch(path_video_id):
                    return path_video_id

    for pattern in VIDEO_ID_URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    return None


def normalize_youtube_url(value: str) -> str:
    video_id = extract_video_id(value)
    if not video_id:
        raise ValueError("Could not find a YouTube video ID")
    return f"https://www.youtube.com/watch?v={video_id}"
