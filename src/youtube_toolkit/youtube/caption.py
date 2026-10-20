"""YouTube caption tracks become timed transcript documents here: track choice, JSON3 download, parsing, and SponsorBlock removal.

YouTube's machine translations are never used: the client drops them from metadata, and automatic captions are chosen only in the spoken language of the video.
Language preference and the speech-to-text fallback belong to the transcript workflow that calls this adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
import html
import json
import urllib.error
import urllib.request

from ..execution import retry_transient
from ..identity import ALLOWED_CAPTION_HOST_SUFFIXES, validate_url
from ..schemas import TranscriptChunk, TranscriptDocument, VideoMetadata

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
CAPTION_REQUEST_ATTEMPTS = 3
CAPTION_RETRY_BACKOFF_SECONDS = 1.0
# YouTube auto-captions English almost always and Chinese rarely, and yt-dlp only reports a spoken language from auto captions, so a video with none is taken to be Chinese.
UNCAPTIONED_LANGUAGE = "zh"
CHINESE_LANGUAGES = frozenset({"zh", "yue"})


class CaptionTranscriptUnavailable(RuntimeError):
    """Only this failure triggers the transcript workflow's speech-to-text fallback; a failed caption download raises a plain RuntimeError instead."""


def fetch_caption_transcript(
    metadata_payload: dict,
    *,
    languages: Sequence[str],
    source: str,
) -> TranscriptDocument:
    metadata = VideoMetadata.from_yt_dlp(metadata_payload)
    caption_url, language, caption_source = select_caption_track(
        metadata_payload, languages, source
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


def select_caption_track(
    metadata: dict,
    languages: Sequence[str],
    source: str,
) -> tuple[str, str, str]:
    """Manual subtitles in any preferred language outrank automatic captions, which are taken only in the spoken language of the video.

    Preferred languages in the spoken language's family are tried first, so a Chinese video with English and Chinese subtitles yields Chinese.
    """
    if source not in {"auto", "manual", "any"}:
        raise ValueError("caption source must be auto, manual, or any")
    manual = {} if source == "auto" else metadata.get("subtitles") or {}
    automatic = {} if source == "manual" else metadata.get("automatic_captions") or {}
    spoken = metadata.get("language")
    if not isinstance(spoken, str) or not spoken:
        spoken = None
    candidates = [
        ("subtitles", key, manual[key])
        for language in _spoken_first(languages, spoken)
        for key in _language_keys(language, manual)
    ] + [
        ("automatic_captions", key, automatic[key])
        for key in _original_language_keys(spoken, automatic)
    ]
    for source_name, key, tracks in candidates:
        url = _track_url(tracks)
        if url is not None:
            return url, key, source_name

    available_summary = ", ".join(sorted({*manual, *automatic})) or "none"
    requested = ", ".join(languages)
    raise CaptionTranscriptUnavailable(
        f"No caption track found for requested languages [{requested}]. Available languages: {available_summary}"
    )


def fetch_caption_payload(url: str) -> dict:
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
    text = html.unescape(raw_text.replace("\n", " "))
    return " ".join(text.split())


def extract_sponsorblock_intervals(metadata: dict) -> list[tuple[float, float]]:
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


def _spoken_first(languages: Sequence[str], spoken: str | None) -> list[str]:
    family = _family(spoken)
    return sorted(languages, key=lambda language: _family(language) != family)


def _family(language: str | None) -> str:
    base = (language or UNCAPTIONED_LANGUAGE).partition("-")[0]
    return "zh" if base in CHINESE_LANGUAGES else base


def _language_keys(language: str, captions: dict) -> list[str]:
    variants = [key for key in captions if key.startswith(f"{language}-")]
    return [*([language] if language in captions else []), *variants]


def _original_language_keys(language: str | None, automatic: dict) -> list[str]:
    """Dubbed videos mark one original track per audio language, so the video's language decides; without it, only a single original mark or a single remaining language is trusted."""
    if language is None:
        marked = [key for key in automatic if key.endswith("-orig")]
        unambiguous = marked or list(automatic)
        return unambiguous if len(unambiguous) == 1 else []
    base = language.partition("-")[0]
    candidates = (f"{language}-orig", language, f"{base}-orig", base)
    return [key for key in dict.fromkeys(candidates) if key in automatic]


def _track_url(tracks: list[dict]) -> str | None:
    usable = [track for track in tracks if track.get("url")]
    preferred = next((track for track in usable if track.get("ext") == "json3"), None)
    if preferred is not None:
        return preferred["url"]
    return usable[0]["url"] if usable else None
