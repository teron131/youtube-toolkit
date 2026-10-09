"""Transcript retrieval owns caption selection, STT fallback, and cleanup policy."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Any, Protocol
import urllib.error
import urllib.request

from ..convert_zh import convert_zh
from ..execution import retry_transient
from ..identity import (
    ALLOWED_CAPTION_HOST_SUFFIXES,
    VideoRef,
    require_video_id,
    validate_url,
)
from ..schemas import TranscriptChunk, TranscriptDocument, VideoMetadata
from ..stt import transcribe_audio_details
from .client import YtDlpClient

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
DEFAULT_TRANSCRIPT_LANGUAGES = (
    "en-orig",
    "en",
    "zh-Hant",
    "zh-HK",
    "yue-HK",
    "yue",
    "zh-TW",
    "zh-Hans",
    "zh-CN",
    "zh",
)
CAPTION_REQUEST_ATTEMPTS = 3
CAPTION_RETRY_BACKOFF_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class TranscriptOptions:
    """Caller policy for caption language, source, and permission to fall back to untimed STT."""

    languages: tuple[str, ...] = DEFAULT_TRANSCRIPT_LANGUAGES
    source: str = "any"
    allow_stt: bool = True


class CaptionTranscriptUnavailable(RuntimeError):
    """Raised when metadata has no usable caption transcript for the requested policy."""


class TranscriptClient(Protocol):
    """yt-dlp surface for caption metadata and the speech-to-text audio fallback."""

    def metadata(self, video_ref: VideoRef, /) -> dict[str, Any]:
        """Fetch raw video metadata containing caption URLs and fallback STT identity."""
        ...

    def download_transcription_audio(self, video_ref: VideoRef, /) -> Path:
        """Download one temporary audio file for the transcript STT fallback."""
        ...


def fetch_live_transcript(
    video_ref: VideoRef,
    *,
    options: TranscriptOptions | None = None,
    client: TranscriptClient | None = None,
    metadata: dict[str, Any] | None = None,
) -> TranscriptDocument:
    """Prefer captions and fall back to STT, using a matching metadata response already fetched by the caller when available."""
    options = options or TranscriptOptions()
    client = client or YtDlpClient()
    if metadata is not None and metadata.get("id") != require_video_id(video_ref):
        raise ValueError("Supplied metadata does not match requested video")
    metadata_payload = client.metadata(video_ref) if metadata is None else metadata
    try:
        return fetch_caption_transcript(metadata_payload, options=options)
    except CaptionTranscriptUnavailable:
        if not options.allow_stt:
            raise
        return fetch_stt_transcript(metadata_payload, client=client)


def fetch_caption_transcript(
    metadata_payload: dict,
    *,
    options: TranscriptOptions,
) -> TranscriptDocument:
    """Build a transcript document from yt-dlp caption metadata under the requested caption policy."""
    metadata = VideoMetadata.from_yt_dlp(metadata_payload)
    caption_url, language, caption_source = select_caption_track(
        metadata_payload,
        list(options.languages),
        options.source,
    )
    caption_payload = fetch_caption_payload(caption_url)
    chunks = extract_timed_chunks(caption_payload)
    chunks = remove_sponsorblock_chunks(
        chunks,
        extract_sponsorblock_intervals(metadata_payload),
    )
    if not chunks:
        raise CaptionTranscriptUnavailable(
            f"Caption track resolved from {caption_source}:{language}, but no transcript text was found"
        )

    return TranscriptDocument(
        metadata=metadata,
        language=language,
        source=caption_source,
        chunks=chunks,
    )


def fetch_stt_transcript(
    metadata_payload: dict,
    *,
    client: TranscriptClient,
) -> TranscriptDocument:
    """Download temporary audio, run STT, and return its untimed text as one transcript chunk."""
    metadata = VideoMetadata.from_yt_dlp(metadata_payload)
    audio_path = client.download_transcription_audio(metadata.video_id)
    try:
        transcription = transcribe_audio_details(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
    return TranscriptDocument(
        metadata=metadata,
        language="speech-to-text",
        source="speech-to-text",
        chunks=[
            TranscriptChunk(
                start_seconds=0,
                end_seconds=0,
                text=convert_zh(transcription.text),
            )
        ],
    )


def select_caption_track(
    metadata: dict,
    languages: list[str],
    source: str,
) -> tuple[str, str, str]:
    """Select the first usable caption URL matching requested language and source precedence."""
    automatic = metadata.get("automatic_captions") or {}
    manual = metadata.get("subtitles") or {}
    if source == "auto":
        caption_sources = [("automatic_captions", automatic)]
    elif source == "manual":
        caption_sources = [("subtitles", manual)]
    elif source == "any":
        caption_sources = [("automatic_captions", automatic), ("subtitles", manual)]
    else:
        raise ValueError("caption source must be auto, manual, or any")

    for language in languages:
        for source_name, caption_pool in caption_sources:
            tracks = caption_pool.get(language) or []
            for track in tracks:
                if track.get("ext") == "json3" and track.get("url"):
                    return track["url"], language, source_name
            for track in tracks:
                if track.get("url"):
                    return track["url"], language, source_name

    available_languages = sorted(
        {language for _, caption_pool in caption_sources for language in caption_pool}
    )
    available_summary = ", ".join(available_languages) or "none"
    requested = ", ".join(languages)
    raise CaptionTranscriptUnavailable(
        f"No caption track found for requested languages [{requested}]. Available languages: {available_summary}"
    )


def fetch_caption_payload(url: str) -> dict:
    """Caption URLs are host-validated before the remote JSON payload is fetched."""
    validate_url(url, allowed_hosts=ALLOWED_CAPTION_HOST_SUFFIXES)
    request = urllib.request.Request(  # noqa: S310 - caption URLs are scheme/host validated above.
        url,
        headers={"User-Agent": USER_AGENT},
    )

    def request_payload() -> dict:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            data = response.read().decode("utf-8")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise RuntimeError("Caption track returned a non-object JSON payload")
        return payload

    try:
        return retry_transient(
            request_payload,
            attempts=CAPTION_REQUEST_ATTEMPTS,
            backoff_seconds=CAPTION_RETRY_BACKOFF_SECONDS,
            retryable_errors=(urllib.error.URLError, json.JSONDecodeError),
        )
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Failed to download caption track ({error.code}): {error.reason}"
        ) from error
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        detail = (
            "Caption track was not valid JSON"
            if isinstance(error, json.JSONDecodeError)
            else f"Failed to download caption track: {error}"
        )
        raise RuntimeError(detail) from error


def extract_timed_chunks(caption_payload: dict) -> list[TranscriptChunk]:
    """Convert YouTube JSON3 caption events into deduped timed transcript chunks."""
    last_chunk = ""
    chunks: list[TranscriptChunk] = []
    for event in caption_payload.get("events", []):
        segments = event.get("segs") or []
        if not segments:
            continue
        text = normalize_caption_chunk(
            "".join(segment.get("utf8", "") for segment in segments)
        )
        if not text or text == last_chunk:
            continue
        start_seconds = float(event.get("tStartMs", 0)) / 1000
        duration_seconds = float(event.get("dDurationMs") or 0) / 1000
        chunks.append(
            TranscriptChunk(
                start_seconds=start_seconds,
                end_seconds=start_seconds + duration_seconds,
                text=text,
            )
        )
        last_chunk = text
    return chunks


def normalize_caption_chunk(raw_text: str) -> str:
    """Normalize caption event text by unescaping HTML and collapsing whitespace."""
    text = html.unescape(raw_text.replace("\n", " "))
    return " ".join(text.split())


def extract_sponsorblock_intervals(metadata: dict) -> list[tuple[float, float]]:
    """Read SponsorBlock chapter intervals from yt-dlp metadata for transcript filtering."""
    intervals: list[tuple[float, float]] = []
    for chapter in metadata.get("sponsorblock_chapters") or []:
        try:
            start_time = float(chapter["start_time"])
            end_time = float(chapter["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_time > start_time:
            intervals.append((start_time, end_time))
    return intervals


def remove_sponsorblock_chunks(
    chunks: list[TranscriptChunk],
    intervals: list[tuple[float, float]],
) -> list[TranscriptChunk]:
    """Drop caption chunks whose time span overlaps SponsorBlock intervals."""
    if not intervals:
        return chunks

    filtered: list[TranscriptChunk] = []
    for chunk in chunks:
        effective_end = (
            chunk.end_seconds
            if chunk.end_seconds > chunk.start_seconds
            else chunk.start_seconds + 0.001
        )
        overlaps_skip = any(
            chunk.start_seconds < interval_end and effective_end > interval_start
            for interval_start, interval_end in intervals
        )
        if not overlaps_skip:
            filtered.append(chunk)
    return filtered
