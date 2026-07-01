"""YouTube reference parsing for video and channel identities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import re
from urllib.parse import parse_qs, urlparse

from .schemas import ChannelIdentity, normalize_channel_id, normalize_handle

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
class YouTubeRefs:
    values: tuple[YouTubeRef, ...]

    @classmethod
    def parse(cls, refs: YouTubeRefInput | None) -> YouTubeRefs:
        """Normalize an optional single ref or list of refs into one iterable identity set."""
        if refs is None:
            return cls(())
        if isinstance(refs, YouTubeRefs):
            return refs
        if isinstance(refs, str):
            return cls((refs,))
        return cls(tuple(refs))

    def __iter__(self) -> Iterator[YouTubeRef]:
        return iter(self.values)

    def video_ids(self) -> list[str]:
        return [require_video_id(video_ref) for video_ref in self.values]

    def video_urls(self) -> list[str]:
        return [normalize_youtube_url(video_ref) for video_ref in self.values]

    def channel_identities(self) -> list[ChannelIdentity]:
        return [parse_channel_ref(channel_ref) for channel_ref in self.values]


type YouTubeRefInput = YouTubeRef | list[YouTubeRef] | YouTubeRefs
type ChannelRefInput = YouTubeRefInput
type VideoRefInput = YouTubeRefInput


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
    """Parse a durable channel identity from a user-facing channel reference."""
    channel_handle = normalize_handle(channel_ref)
    if channel_handle is not None:
        return ChannelIdentity(
            youtube_channel_id=None,
            channel_handle=channel_handle,
        )

    stripped_channel_ref = channel_ref.strip()
    if stripped_channel_ref.startswith(("http://", "https://")):
        parsed_url = urlparse(stripped_channel_ref)
        path_parts = [path_part for path_part in parsed_url.path.split("/") if path_part]
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
