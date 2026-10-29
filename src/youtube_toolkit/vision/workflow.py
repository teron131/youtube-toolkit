"""Own video preflight, the choice of ranking route (decision model, Gemini embeddings, local embeddings, or time slots), temporary media lifecycle, and the vision result."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Literal

from ..artifacts import create_artifact_directory
from ..config import (
    API_EMBEDDING_CUTOFFS,
    DEFAULT_VISION_MAX_FRAMES,
    DEFAULT_VISION_MIN_GAP_SECONDS,
    MAX_VISION_FRAMES,
    decision_config,
    local_embedding_cutoffs,
    vision_embedding_key,
)
from ..embeddings import (
    EmbeddingUsage,
    GeminiEmbeddingClient,
    LocalImageEmbeddingClient,
    local_embedding_installed,
)
from ..identity import VideoRef, require_video_id
from ..schemas import TranscriptDocument, VideoMetadata
from ..transcript import fetch_transcript
from ..youtube import YtDlpClient
from ..youtube.caption import extract_sponsorblock_intervals
from .candidates import collect_candidates, select_ranked_frames, select_slot_frames
from .collages import render_timestamped_collages
from .decisions import MIN_SCORE, DecisionUsage, score_frames
from .reranking import rerank_frames
from .text import prepare_targets

MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
SPONSOR_TITLE = re.compile(r"\b(?:sponsor(?:ed|ship)?|advertisement|ad break)\b", re.I)


def sample_vision(
    video_ref: VideoRef,
    *,
    max_frames: int = DEFAULT_VISION_MAX_FRAMES,
    min_similarity: float | None = None,
    min_gap_seconds: float = DEFAULT_VISION_MIN_GAP_SECONDS,
    local_only: bool = False,
    auto_clean: bool = True,
) -> dict[str, Any]:
    """With a timed transcript, the first available route ranks the video's keyframes: the decision model with DECISION_API_KEY, Gemini embeddings with a Gemini key, then the local embedding model when its libraries are installed; without any, time slots spread a local selection across the same keyframes.

    local_only skips both APIs and speech-to-text, so the local model ranks against captions only.
    The transcript is fetched only when a model will rank, since time slots never read it; without captions it comes from configured speech-to-text, which ranks frames only when the route reports timing.
    min_similarity applies to embedding routes, and None means the cutoff measured for the ranking model.
    max_frames limits the output, never the candidate search; ranking errors propagate.
    Collages go to a new directory in the working directory's archive.
    """
    if not 1 <= max_frames <= MAX_VISION_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_VISION_FRAMES}")
    if min_similarity is not None and (
        not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1
    ):
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
    decision = None if local_only else decision_config()
    gemini_key = None if local_only else vision_embedding_key()
    route: Literal["decision", "gemini", "local"] | None
    if decision:
        route = "decision"
    elif gemini_key:
        route = "gemini"
    elif local_embedding_installed():
        route = "local"
    else:
        route = None
    transcript_warning = None
    embedding_warning = None
    decision_warning = None
    document: TranscriptDocument | None = None
    targets = []
    if route:
        try:
            document = fetch_transcript(
                video_ref, allow_stt=not local_only, client=client, metadata=metadata
            )
        # Speech-to-text reports a recording without speech as ValueError, which leaves nothing to rank against, as missing captions do.
        except (RuntimeError, ValueError) as error:
            transcript_warning = f"No timed transcript; using visual sampling: {error}"
        else:
            # Every route needs timed text to place frames in; the decision route reads the lines themselves.
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
        embedding_warning = "Set DECISION_API_KEY or GEMINI_API_KEY, or install youtube-toolkit[embeddings], for relevance ranking; using local frame selection."
    if not targets:
        route = None
    selection_mode = {
        "decision": "decision",
        "gemini": "semantic",
        "local": "semantic",
        None: "visual",
    }[route]
    embedding_model = None
    usage: EmbeddingUsage | None = None
    decision_usage: DecisionUsage | None = None
    cutoffs = API_EMBEDDING_CUTOFFS
    relevance: dict[float, dict[str, Any]] = {}
    if route in ("gemini", "local"):
        # Texts need no video, so a bad key, quota, or model fails before the download and decoding.
        if gemini_key:
            embedder = GeminiEmbeddingClient(gemini_key)
        else:
            embedder = LocalImageEmbeddingClient()
            cutoffs, embedding_warning = local_embedding_cutoffs(embedder.model)
        embedding_model = embedder.model
        embedded_texts = embedder.embed([], [target.text for target in targets])
        usage = embedded_texts.usage
    match_floor = cutoffs.vision_match if min_similarity is None else min_similarity
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
            selected = []
            if route == "decision" and document is not None and decision:
                decisions, decision_usage = score_frames(candidates, document, decision)
                if unscored := decisions.count(None):
                    decision_warning = f"{unscored} of {len(candidates)} frames went unscored after repeated decision host errors and were left out of the ranking."
                passing = {
                    index: decision
                    for index, decision in enumerate(decisions)
                    if decision is not None and decision.score >= MIN_SCORE
                }
                order = sorted(
                    passing,
                    key=lambda index: (
                        -passing[index].score,
                        candidates[index].timestamp_seconds,
                    ),
                )
                picks = select_ranked_frames(
                    candidates,
                    order,
                    max_frames=max_frames,
                    min_gap_seconds=min_gap_seconds,
                )
                for rank, (index, frame) in enumerate(picks, start=1):
                    selected.append(frame)
                    relevance[frame.timestamp_seconds] = {
                        "selection_rank": rank,
                        **asdict(passing[index]),
                    }
            elif route in ("gemini", "local"):
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
                        min_similarity=match_floor,
                        min_gap_seconds=min_gap_seconds,
                        duplicate_cosine=cutoffs.vision_duplicate,
                    )
                    for rank, ranking in enumerate(ranked, start=1):
                        frame = candidates[ranking.index].load()
                        selected.append(frame)
                        relevance[frame.timestamp_seconds] = {
                            "selection_rank": rank,
                            "rrf_score": ranking.score,
                            "matches": [asdict(match) for match in ranking.matches],
                        }
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
        result["relevance"] = relevance.get(frame.timestamp_seconds)
        frame_results.append(result)
    return {
        "video_id": video_id,
        "duration_seconds": duration,
        "heatmap_available": bool(metadata.get("heatmap")),
        "selection_mode": selection_mode,
        "transcript_warning": transcript_warning,
        "embedding_warning": embedding_warning,
        "embedding_model": embedding_model,
        "embedding_usage": asdict(usage) if usage is not None else None,
        "decision_model": decision.model if decision and route == "decision" else None,
        "decision_warning": decision_warning,
        "decision_usage": asdict(decision_usage)
        if decision_usage is not None
        else None,
        "candidate_count": len(candidates),
        "min_similarity": None if route == "decision" else match_floor,
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
