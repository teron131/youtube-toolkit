"""Own video preflight, the choice of ranking route (decision model, Gemini embeddings, local embeddings, or time slots), temporary media lifecycle, and the vision result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from ..artifacts import create_artifact_directory
from ..config import (
    API_EMBEDDING_CUTOFFS,
    DEFAULT_VISION_MAX_FRAMES,
    DEFAULT_VISION_MIN_GAP_SECONDS,
    MAX_VISION_FRAMES,
    DecisionConfig,
    EmbeddingCutoffs,
    decision_config,
    embedding_key,
    local_embedding_cutoffs,
)
from ..embeddings import (
    EmbeddingResult,
    EmbeddingUsage,
    GeminiEmbeddingClient,
    LocalImageEmbeddingClient,
    local_embedding_installed,
)
from ..identity import VideoRef, require_video_id
from ..schemas import TranscriptDocument, VideoMetadata
from ..transcript import fetch_transcript
from ..youtube import YtDlpClient
from ..youtube.captions import extract_sponsorblock_intervals
from .candidates import (
    DecodedFrame,
    StoredFrame,
    collect_candidates,
    select_ranked_frames,
    select_slot_frames,
)
from .collages import render_timestamped_collages
from .decisions import DecisionRanking, rank_frames
from .reranking import rerank_frames
from .text import TextTarget, prepare_targets

MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
SPONSOR_TITLE = re.compile(r"\b(?:sponsor(?:ed|ship)?|advertisement|ad break)\b", re.I)

# A selected frame with its relevance record, which is None when time slots chose it.
type _Pick = tuple[DecodedFrame, dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class _Embedding:
    """An embedding route with the transcript targets already embedded, row for row; texts need no video, so a bad key, quota, or model fails before the download and decoding."""

    embedder: GeminiEmbeddingClient | LocalImageEmbeddingClient
    cutoffs: EmbeddingCutoffs
    warning: str | None
    targets: list[TextTarget]
    texts: EmbeddingResult


def sample_vision(
    video_ref: VideoRef,
    *,
    max_frames: int = DEFAULT_VISION_MAX_FRAMES,
    min_similarity: float | None = None,
    min_gap_seconds: float = DEFAULT_VISION_MIN_GAP_SECONDS,
    local_only: bool = False,
    auto_clean: bool = True,
    metadata: dict[str, Any] | None = None,
    document: TranscriptDocument | None = None,
) -> dict[str, Any]:
    """The first available route ranks the video's keyframes: the decision model with DECISION_API_KEY, which judges frames by their pictures and, when there is timed speech, by whether they show what is being said, then Gemini embeddings with a Gemini key or the local embedding model when its libraries are installed, both of which need timed speech; otherwise time slots spread a local selection across the same keyframes.

    local_only skips both APIs and speech-to-text, so the local model ranks against captions only.
    The transcript is fetched only when a model will rank and no document is given, since time slots never read it; without captions it comes from configured speech-to-text, which ranks frames only when the route reports timing.
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
    if metadata is None:
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
    gemini_key = None if local_only else embedding_key()
    transcript_warning = None
    embedding_warning = None
    targets: list[TextTarget] = []
    if decision or gemini_key or local_embedding_installed():
        fallback = (
            "the decision model judges frames without speech"
            if decision
            else "using visual sampling"
        )
        if document is None:
            try:
                document = fetch_transcript(
                    video_ref,
                    stt="never" if local_only else "fallback",
                    client=client,
                    metadata=metadata,
                )
            # Speech-to-text reports a recording without speech as ValueError, which leaves no speech context, as missing captions do.
            except (RuntimeError, ValueError) as error:
                transcript_warning = f"No timed transcript; {fallback}: {error}"
        if document is not None:
            targets = prepare_targets(document, excluded_intervals)
            if not targets:
                transcript_warning = (
                    f"Too little timed speech to rank against; {fallback}."
                    if document.timed
                    else f"Speech-to-text gave no timestamps; {fallback}."
                )
    elif not local_only:
        embedding_warning = "Set DECISION_API_KEY or EMBEDDING_API_KEY, or install youtube-toolkit[embeddings], for relevance ranking; using local frame selection."
    # Embedding routes match frames to timed speech, so without it they fall back to time slots; the decision model can judge a frame by its image alone.
    embedding = None
    if not decision and targets:
        embedding = _embed_targets(gemini_key, targets)
        embedding_warning = embedding.warning
    cutoffs = API_EMBEDDING_CUTOFFS if embedding is None else embedding.cutoffs
    match_floor = cutoffs.vision_match if min_similarity is None else min_similarity
    decision_ranking = None
    embedding_usage = None
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
            if decision:
                picks, decision_ranking = _rank_by_decisions(
                    candidates,
                    document if targets else None,
                    decision,
                    max_frames=max_frames,
                    min_gap_seconds=min_gap_seconds,
                )
            elif embedding:
                picks, embedding_usage = _rank_by_embeddings(
                    embedding,
                    candidates,
                    max_frames=max_frames,
                    min_similarity=match_floor,
                    min_gap_seconds=min_gap_seconds,
                )
            else:
                picks = [
                    (frame, None)
                    for frame in select_slot_frames(
                        candidates,
                        duration=duration,
                        max_frames=max_frames,
                        min_gap_seconds=min_gap_seconds,
                    )
                ]
            picks.sort(key=lambda pick: pick[0].timestamp_seconds)
            collage_paths = render_timestamped_collages(
                [frame for frame, _ in picks], destination / f"{video_id}_vision"
            )
    except OSError as error:
        raise RuntimeError(
            f"Vision artifact filesystem error under {destination}: {error}"
        ) from error
    return {
        "video_id": video_id,
        "duration_seconds": duration,
        "heatmap_available": bool(metadata.get("heatmap")),
        "selection_mode": "decision"
        if decision
        else "semantic"
        if embedding
        else "visual",
        "transcript_warning": transcript_warning,
        "embedding_warning": embedding_warning,
        "embedding_model": None if embedding is None else embedding.embedder.model,
        "embedding_usage": None if embedding_usage is None else asdict(embedding_usage),
        "decision_model": decision.model if decision else None,
        "decision_warning": None
        if decision_ranking is None
        else decision_ranking.warning,
        "decision_usage": None
        if decision_ranking is None
        else asdict(decision_ranking.usage),
        "candidate_count": len(candidates),
        "min_similarity": None if decision else match_floor,
        "min_gap_seconds": min_gap_seconds,
        "frame_count": len(picks),
        "timestamps_seconds": [frame.timestamp_seconds for frame, _ in picks],
        "directory": str(destination),
        "collage_paths": [str(path) for path in collage_paths],
        "media_dir": None if auto_clean else str(work_dir),
        "frames": [
            {**frame.to_dict(), "relevance": relevance} for frame, relevance in picks
        ],
    }


def _embed_targets(gemini_key: str | None, targets: list[TextTarget]) -> _Embedding:
    if gemini_key:
        embedder = GeminiEmbeddingClient(gemini_key)
        cutoffs, warning = API_EMBEDDING_CUTOFFS, None
    else:
        embedder = LocalImageEmbeddingClient()
        cutoffs, warning = local_embedding_cutoffs(embedder.model)
    texts = embedder.embed([], [target.text for target in targets])
    return _Embedding(embedder, cutoffs, warning, targets, texts)


def _rank_by_decisions(
    candidates: list[StoredFrame],
    document: TranscriptDocument | None,
    config: DecisionConfig,
    *,
    max_frames: int,
    min_gap_seconds: float,
) -> tuple[list[_Pick], DecisionRanking]:
    ranking = rank_frames(candidates, document, config)
    decisions = dict(ranking.frames)
    picks = select_ranked_frames(
        candidates,
        list(decisions),
        max_frames=max_frames,
        min_gap_seconds=min_gap_seconds,
    )
    return [
        (frame, {"selection_rank": rank, **asdict(decisions[index])})
        for rank, (index, frame) in enumerate(picks, start=1)
    ], ranking


def _rank_by_embeddings(
    embedding: _Embedding,
    candidates: list[StoredFrame],
    *,
    max_frames: int,
    min_similarity: float,
    min_gap_seconds: float,
) -> tuple[list[_Pick], EmbeddingUsage]:
    if not candidates:
        return [], embedding.texts.usage
    images = embedding.embedder.embed([frame.image_path for frame in candidates], [])
    rankings = rerank_frames(
        images.images,
        embedding.texts.texts,
        embedding.targets,
        [frame.timestamp_seconds for frame in candidates],
        max_frames=max_frames,
        min_similarity=min_similarity,
        min_gap_seconds=min_gap_seconds,
        duplicate_cosine=embedding.cutoffs.vision_duplicate,
    )
    picks: list[_Pick] = [
        (
            candidates[ranking.index].load(),
            {
                "selection_rank": rank,
                "rrf_score": ranking.score,
                "matches": [asdict(match) for match in ranking.matches],
            },
        )
        for rank, ranking in enumerate(rankings, start=1)
    ]
    return picks, embedding.texts.usage + images.usage


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
