"""User references and yt-dlp payloads become validated video IDs, channel IDs, handles, or canonical URLs here."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

type YouTubeRef = str
type ChannelRef = YouTubeRef
type VideoRef = YouTubeRef

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
ALLOWED_CHANNEL_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
}
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
CHANNEL_ID_PATTERN = re.compile(r"^UC[A-Za-z0-9_-]+$")
VIDEO_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")
CHANNEL_PREFIXES = ("/@", "/channel/", "/user/", "/c/")
VIDEO_ID_URL_PATTERNS = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})"),
)


@dataclass(frozen=True, slots=True)
class ChannelIdentity:
    youtube_channel_id: str | None
    channel_handle: str | None

    def has_identity(self) -> bool:
        return self.youtube_channel_id is not None or self.channel_handle is not None


def validate_url(
    url: str,
    *,
    allowed_hosts: set[str] | tuple[str, ...],
) -> None:
    """Each allowed host also admits its subdomains."""
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or 'missing'}")

    hostname = parsed.hostname or ""
    if hostname in allowed_hosts:
        return
    if any(hostname.endswith(f".{suffix}") for suffix in allowed_hosts):
        return
    raise ValueError(f"Unsupported URL host: {hostname or 'missing'}")


def is_video_id(value: str) -> bool:
    return bool(VIDEO_ID_PATTERN.fullmatch(value.strip()))


def is_channel_id(value: str) -> bool:
    return bool(CHANNEL_ID_PATTERN.fullmatch(value.strip()))


def extract_video_id(video_ref: VideoRef) -> str | None:
    candidate = video_ref.strip()
    if is_video_id(candidate):
        return candidate

    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)
    path_parts = [part for part in parsed.path.split("/") if part]

    if hostname == "youtu.be" and path_parts and is_video_id(path_parts[0]):
        return path_parts[0]

    if hostname in ALLOWED_VIDEO_HOSTS:
        query_video_id = query.get("v", [""])[0]
        if is_video_id(query_video_id):
            return query_video_id

        normalized_path = f"/{'/'.join(path_parts)}" if path_parts else ""
        for prefix in VIDEO_PATH_PREFIXES:
            if normalized_path.startswith(prefix):
                path_video_id = normalized_path[len(prefix) :].split("/", 1)[0]
                if is_video_id(path_video_id):
                    return path_video_id

    for pattern in VIDEO_ID_URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    return None


def normalize_youtube_url(video_ref: VideoRef) -> str:
    video_id = require_video_id(video_ref)
    return f"https://www.youtube.com/watch?v={video_id}"


def require_video_id(video_ref: VideoRef) -> str:
    video_id = extract_video_id(video_ref)
    if video_id is None:
        raise ValueError("Could not find a YouTube video ID")
    return video_id


def parse_channel_ref(channel_ref: ChannelRef) -> ChannelIdentity:
    """Resolve only durable channel IDs and handles; names that require lookup stay unresolved."""
    channel_handle = normalize_handle(channel_ref)
    if channel_handle is not None:
        return ChannelIdentity(
            youtube_channel_id=None,
            channel_handle=channel_handle,
        )

    stripped_channel_ref = channel_ref.strip()
    if stripped_channel_ref.startswith(("http://", "https://")):
        parsed_url = urlparse(stripped_channel_ref)
        path_parts = [
            path_part for path_part in parsed_url.path.split("/") if path_part
        ]
        if len(path_parts) >= 2 and path_parts[0] == "channel":
            return ChannelIdentity(
                youtube_channel_id=extract_channel_id(path_parts[1]),
                channel_handle=None,
            )
        return ChannelIdentity(
            youtube_channel_id=None,
            channel_handle=None,
        )

    return ChannelIdentity(
        youtube_channel_id=extract_channel_id(stripped_channel_ref),
        channel_handle=None,
    )


def extract_channel_id(channel_ref: str) -> str | None:
    channel_id = normalize_channel_id(channel_ref)
    if channel_id is None or not is_channel_id(channel_id):
        return None
    return channel_id


def normalize_channel_url(channel_ref: ChannelRef) -> str:
    """Canonicalize to the videos tab URL that yt-dlp lists, treating a bare name as a handle: `OpenAI` and `https://youtube.com/@OpenAI` both become `https://www.youtube.com/@OpenAI/videos`."""
    candidate = channel_ref.strip()
    if not candidate:
        raise ValueError("Expected a non-empty YouTube channel reference")

    parsed = urlparse(candidate)
    if parsed.scheme in {"https", "http"}:
        hostname = (parsed.hostname or "").lower()
        if hostname not in ALLOWED_CHANNEL_HOSTS:
            raise ValueError(f"Unsupported YouTube host: {hostname or 'missing'}")
        path = parsed.path.rstrip("/")
    else:
        stripped_candidate = candidate.lstrip("/")
        if candidate.startswith("@"):
            path = f"/{candidate}"
        elif stripped_candidate.startswith(("channel/", "user/", "c/")):
            path = f"/{stripped_candidate}"
        elif is_channel_id(stripped_candidate):
            path = f"/channel/{stripped_candidate}"
        else:
            path = f"/@{stripped_candidate.lstrip('@')}"

    base_path = path[: -len("/videos")] if path.endswith("/videos") else path
    if not any(base_path.startswith(prefix) for prefix in CHANNEL_PREFIXES):
        raise ValueError("Expected a YouTube channel URL, handle, or channel ID")
    return f"https://www.youtube.com{base_path}/videos"


def extract_channel_identity(
    metadata: dict[str, Any],
    *,
    fallback_channel_id: str | None = None,
    fallback_channel_handle: str | None = None,
) -> ChannelIdentity:
    youtube_channel_id = (
        normalize_channel_id(metadata.get("channel_id")) or fallback_channel_id
    )
    handle_candidates = (
        metadata.get("channel_handle"),
        metadata.get("uploader_id"),
        metadata.get("channel_url"),
        metadata.get("uploader_url"),
        metadata.get("webpage_url"),
        fallback_channel_handle,
    )
    channel_handle = next(
        (
            handle
            for handle in (
                normalize_handle(candidate) for candidate in handle_candidates
            )
            if handle
        ),
        None,
    )
    return ChannelIdentity(
        youtube_channel_id=youtube_channel_id,
        channel_handle=channel_handle,
    )


def normalize_handle(handle: Any) -> str | None:
    if not isinstance(handle, str):
        return None
    stripped_handle = handle.strip()
    if not stripped_handle:
        return None
    if stripped_handle.startswith(("http://", "https://")):
        return extract_handle_from_url(stripped_handle)
    if stripped_handle.startswith("@"):
        return stripped_handle
    if (
        stripped_handle.startswith("UC")
        or "/" in stripped_handle
        or " " in stripped_handle
    ):
        return None
    return f"@{stripped_handle}"


def normalize_channel_id(channel_id: Any) -> str | None:
    if not isinstance(channel_id, str):
        return None
    stripped_channel_id = channel_id.strip()
    return stripped_channel_id or None


def extract_handle_from_url(url: str) -> str | None:
    parsed_url = urlparse(url)
    for path_part in parsed_url.path.split("/"):
        if path_part.startswith("@"):
            return path_part
    return None
