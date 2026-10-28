"""Own video preflight, the choice between Gemini ranking and local time slots, temporary media lifecycle, and the vision result."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from ..artifacts import create_artifact_directory
from ..config import (
    DEFAULT_VISION_MAX_FRAMES,
    DEFAULT_VISION_MIN_GAP_SECONDS,
    DEFAULT_VISION_MIN_SIMILARITY,
    MAX_VISION_FRAMES,
    vision_embedding_key,
)
from ..embeddings.gemini import MODEL, EmbeddingUsage, GeminiEmbeddingClient
from ..identity import VideoRef, require_video_id
from ..schemas import VideoMetadata
from ..transcript import fetch_transcript
from ..youtube import YtDlpClient
from ..youtube.caption import extract_sponsorblock_intervals
from .candidates import collect_candidates, select_slot_frames
from .collages import render_timestamped_collages
from .reranking import FrameRanking, rerank_frames
from .text import prepare_targets

MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
SPONSOR_TITLE = re.compile(r"\b(?:sponsor(?:ed|ship)?|advertisement|ad break)\b", re.I)


def sample_vision(
    video_ref: VideoRef,
    *,
    max_frames: int = DEFAULT_VISION_MAX_FRAMES,
    min_similarity: float = DEFAULT_VISION_MIN_SIMILARITY,
    min_gap_seconds: float = DEFAULT_VISION_MIN_GAP_SECONDS,
    local_only: bool = False,
    auto_clean: bool = True,
) -> dict[str, Any]:
    """Gemini ranks the video's keyframes when credentials and a timed transcript are available; otherwise time slots spread a local selection across the same keyframes.

    The transcript is fetched only when Gemini credentials exist, since local selection never reads it; without captions it comes from configured speech-to-text, which ranks frames only when the route reports timing.
    max_frames limits the output, never the candidate search; embedding API errors propagate.
    Collages go to a new directory in the working directory's archive.
    """
    if not 1 <= max_frames <= MAX_VISION_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_VISION_FRAMES}")
    if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be finite and between -1 and 1")
    if not math.isfinite(min_gap_seconds) or min_gap_seconds < 0:
        raise ValueError("min_gap_seconds must be finite and non-negative")
    video_id = require_video_id(video_ref)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required for video vision extraction")
    client = YtDlpClient()
    metadata = client.metadata(video_ref)
    fetched_video_id = require_video_id(str(metadata.get("id") or ""))
    if fetched_video_id != video_id:
        raise ValueError(
            f"yt-dlp returned video {fetched_video_id} for requested video {video_id}"
        )
    duration = metadata.get("duration")
    if (
        not isinstance(duration, int | float)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"Missing valid duration for {video_id}")
    duration = float(duration)
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError(
            f"Video is longer than {MAX_VIDEO_DURATION_SECONDS // 3600} hours"
        )
    excluded_intervals = _excluded_intervals(metadata, duration)
    configured_key = None if local_only else vision_embedding_key()
    transcript_warning = None
    embedding_warning = None
    targets = []
    if configured_key:
        try:
            document = fetch_transcript(video_ref, client=client, metadata=metadata)
        # Speech-to-text reports a recording without speech as ValueError, which leaves nothing to rank against, as missing captions do.
        except (RuntimeError, ValueError) as error:
            transcript_warning = f"No timed transcript; using visual sampling: {error}"
        else:
            targets = prepare_targets(document, excluded_intervals)
            if not targets:
                untimed = all(
                    chunk.end_seconds <= chunk.start_seconds
                    for chunk in document.chunks
                )
                transcript_warning = (
                    "Speech-to-text gave no timestamps; using visual sampling."
                    if untimed
                    else "No usable timed transcript passages; using visual sampling."
                )
    elif not local_only:
        embedding_warning = "Set GEMINI_API_KEY for relative-relevance ranking; using local frame selection."
    api_key = configured_key if targets else None
    selection_mode = "semantic" if api_key else "visual"
    usage: EmbeddingUsage | None = None
    rankings: dict[float, tuple[int, FrameRanking]] = {}
    if api_key:
        # Texts need no video, so a bad key or quota fails before the download and decoding.
        embedder = GeminiEmbeddingClient(api_key)
        embedded_texts = embedder.embed([], [target.text for target in targets])
        usage = embedded_texts.usage
    destination = None
    try:
        destination = create_artifact_directory("vision", video_id)
        with TemporaryDirectory(
            prefix="work-", dir=destination, delete=auto_clean
        ) as work_dir_name:
            work_dir = Path(work_dir_name)
            video_path = work_dir / f"{video_id}.mp4"
            client.download_video_for_vision(
                video_ref, video_path, max_bytes=MAX_VIDEO_BYTES
            )
            candidates = collect_candidates(
                video_path,
                ffmpeg_path,
                duration=duration,
                directory=work_dir / "candidates",
                excluded_intervals=excluded_intervals,
            )
            if api_key:
                selected = []
                if candidates:
                    embedded_images = embedder.embed(
                        [frame.image_path for frame in candidates], []
                    )
                    usage = embedded_texts.usage + embedded_images.usage
                    ranked = rerank_frames(
                        embedded_images.images,
                        embedded_texts.texts,
                        targets,
                        [frame.timestamp_seconds for frame in candidates],
                        max_frames=max_frames,
                        min_similarity=min_similarity,
                        min_gap_seconds=min_gap_seconds,
                    )
                    for rank, ranking in enumerate(ranked, start=1):
                        frame = candidates[ranking.index].load()
                        selected.append(frame)
                        rankings[frame.timestamp_seconds] = (rank, ranking)
            else:
                selected = select_slot_frames(
                    candidates,
                    duration=duration,
                    max_frames=max_frames,
                    min_gap_seconds=min_gap_seconds,
                )
            selected.sort(key=lambda frame: frame.timestamp_seconds)
            collage_paths = render_timestamped_collages(
                selected, destination / f"{video_id}_vision"
            )
    except OSError as error:
        raise RuntimeError(
            f"Vision artifact filesystem error under {destination}: {error}"
        ) from error
    frame_results = []
    for frame in selected:
        result = frame.to_dict()
        result["relevance"] = None
        if frame.timestamp_seconds in rankings:
            rank, ranking = rankings[frame.timestamp_seconds]
            result["relevance"] = {
                "selection_rank": rank,
                "rrf_score": ranking.score,
                "matches": [asdict(match) for match in ranking.matches],
            }
        frame_results.append(result)
    return {
        "video_id": video_id,
        "duration_seconds": duration,
        "heatmap_available": bool(metadata.get("heatmap")),
        "selection_mode": selection_mode,
        "transcript_warning": transcript_warning,
        "embedding_warning": embedding_warning,
        "embedding_model": MODEL if api_key else None,
        "embedding_usage": asdict(usage) if usage is not None else None,
        "candidate_count": len(candidates),
        "min_similarity": min_similarity,
        "min_gap_seconds": min_gap_seconds,
        "frame_count": len(selected),
        "timestamps_seconds": [frame.timestamp_seconds for frame in selected],
        "collage_paths": [str(path) for path in collage_paths],
        "media_dir": None if auto_clean else str(work_dir),
        "frames": frame_results,
    }


def _excluded_intervals(
    metadata: dict[str, Any], duration: float
) -> list[tuple[float, float]]:
    intervals = extract_sponsorblock_intervals(metadata)
    chapters = sorted(
        (
            chapter
            for chapter in VideoMetadata.from_yt_dlp(metadata).chapters
            if math.isfinite(chapter.start_seconds)
            and 0 <= chapter.start_seconds < duration
        ),
        key=lambda chapter: chapter.start_seconds,
    )
    for index, chapter in enumerate(chapters):
        if not SPONSOR_TITLE.search(chapter.title):
            continue
        stop = (
            chapters[index + 1].start_seconds if index + 1 < len(chapters) else duration
        )
        if chapter.end_seconds is not None and math.isfinite(chapter.end_seconds):
            stop = min(stop, chapter.end_seconds)
        if stop > chapter.start_seconds:
            intervals.append((chapter.start_seconds, stop))
    return intervals
