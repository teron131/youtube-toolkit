"""YouTube caption extraction and formatting-ready transcript construction."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import urllib.error
import urllib.request

from ..schemas import TranscriptChunk, TranscriptDocument, TranscriptSection, VideoMetadata
from ..storage import get_connection, upsert_video
from ..url import ALLOWED_CAPTION_HOST_SUFFIXES, validate_url
from .client import YtDlpClient

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


@dataclass(frozen=True, slots=True)
class TranscriptOptions:
    languages: tuple[str, ...] = ("en-orig", "en")
    source: str = "auto"


def fetch_transcript(
    video: str,
    *,
    options: TranscriptOptions | None = None,
    client: YtDlpClient | None = None,
) -> TranscriptDocument:
    options = options or TranscriptOptions()
    client = client or YtDlpClient()
    metadata_payload = client.metadata(video)
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
        raise RuntimeError(f"Caption track resolved from {caption_source}:{language}, but no transcript text was found")

    document = TranscriptDocument(
        metadata=metadata,
        language=language,
        source=caption_source,
        chunks=chunks,
        sections=build_sections(metadata_payload, chunks),
    )
    with get_connection() as connection:
        upsert_video(
            connection,
            metadata_payload,
            transcript=document.text,
        )
    return document


def select_caption_track(
    metadata: dict,
    languages: list[str],
    source: str,
) -> tuple[str, str, str]:
    caption_sources = iter_caption_sources(metadata, source)
    for source_name, caption_pool in caption_sources:
        for language in languages:
            tracks = caption_pool.get(language) or []
            for track in tracks:
                if track.get("ext") == "json3" and track.get("url"):
                    return track["url"], language, source_name
            for track in tracks:
                if track.get("url"):
                    return track["url"], language, source_name

    available_languages = sorted({language for _, caption_pool in caption_sources for language in caption_pool})
    available_summary = ", ".join(available_languages) or "none"
    requested = ", ".join(languages)
    raise RuntimeError(f"No caption track found for requested languages [{requested}]. Available languages: {available_summary}")


def iter_caption_sources(metadata: dict, source: str) -> list[tuple[str, dict]]:
    automatic = metadata.get("automatic_captions") or {}
    manual = metadata.get("subtitles") or {}
    if source == "auto":
        return [("automatic_captions", automatic)]
    if source == "manual":
        return [("subtitles", manual)]
    if source == "any":
        return [("automatic_captions", automatic), ("subtitles", manual)]
    raise ValueError("--caption-source must be auto, manual, or any")


def fetch_caption_payload(url: str) -> dict:
    validate_url(url, allowed_hosts=ALLOWED_CAPTION_HOST_SUFFIXES)
    request = urllib.request.Request(  # noqa: S310 - caption URLs are scheme/host validated above.
        url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            data = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download caption track: {exc}") from exc

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Caption track was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Caption track returned a non-object JSON payload")
    return payload


def extract_timed_chunks(caption_payload: dict) -> list[TranscriptChunk]:
    last_chunk = ""
    chunks: list[TranscriptChunk] = []
    for event in caption_payload.get("events", []):
        segments = event.get("segs") or []
        if not segments:
            continue
        text = normalize_caption_chunk("".join(segment.get("utf8", "") for segment in segments))
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
        effective_end = chunk.end_seconds if chunk.end_seconds > chunk.start_seconds else chunk.start_seconds + 0.001
        overlaps_skip = any(chunk.start_seconds < interval_end and effective_end > interval_start for interval_start, interval_end in intervals)
        if not overlaps_skip:
            filtered.append(chunk)
    return filtered


def build_sections(
    metadata: dict,
    chunks: list[TranscriptChunk],
) -> list[TranscriptSection]:
    chapters = metadata.get("chapters") or []
    if not chapters:
        return [
            TranscriptSection(
                title="Transcript",
                start_seconds=None,
                end_seconds=None,
                text=" ".join(chunk.text for chunk in chunks),
            )
        ]

    sections: list[TranscriptSection] = []
    for chapter in chapters:
        start_time = float(chapter.get("start_time") or 0)
        end_time = float(chapter["end_time"]) if chapter.get("end_time") is not None else None
        section_chunks = [chunk.text for chunk in chunks if chunk.start_seconds >= start_time and (end_time is None or chunk.start_seconds < end_time)]
        sections.append(
            TranscriptSection(
                title=str(chapter.get("title") or "Untitled chapter"),
                start_seconds=start_time,
                end_seconds=end_time,
                text=" ".join(section_chunks),
            )
        )
    return sections
