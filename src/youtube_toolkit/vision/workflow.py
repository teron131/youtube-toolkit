"""Vision's entry point, which picks a ranking route for a video's keyframes, or reuses the frames a matching earlier run saved, and writes them as collages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from ..artifacts import artifact_directories, create_artifact_directory
from ..config import (
    API_EMBEDDING_CUTOFFS,
    DEFAULT_VISION_COLUMNS,
    DEFAULT_VISION_MAX_FRAMES,
    DEFAULT_VISION_MIN_GAP_SECONDS,
    DEFAULT_VISION_RESOLUTION,
    DEFAULT_VISION_ROWS,
    GEMINI_EMBEDDING_MODEL,
    MAX_VISION_COLUMNS,
    MAX_VISION_FRAMES,
    MAX_VISION_ROWS,
    VISION_RESOLUTIONS,
    VISION_SOURCE_HEIGHT,
    DecisionConfig,
    EmbeddingCutoffs,
    VisionResolution,
    decision_config,
    gemini_embedding_key,
    local_embedding_cutoffs,
    local_embedding_model,
)
from ..embeddings import (
    EmbeddingResult,
    EmbeddingUsage,
    GeminiEmbeddingClient,
    LocalImageEmbeddingClient,
    local_embedding_installed,
)
from ..ffmpeg_utils import decode_frame
from ..identity import VideoRef, require_video_id
from ..schemas import TranscriptDocument, VideoMetadata
from ..transcript import fetch_transcript
from ..youtube import (
    YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
    YtDlpClient,
    extract_sponsorblock_intervals,
)
from .candidates import (
    CANDIDATE_HEIGHT,
    CANDIDATE_WIDTH,
    DecodedFrame,
    StoredFrame,
    VisualScores,
    collect_candidates,
    select_ranked_frames,
    select_slot_frames,
    store_frame,
)
from .collages import render_collages
from .decisions import DecisionRanking, rank_frames
from .reranking import rerank_frames
from .text import TextTarget, prepare_targets

MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60
MAX_VIDEO_BYTES = 500 * 1024 * 1024
PICKS_FILE = "picks.json"
SPONSOR_TITLE = re.compile(r"\b(?:sponsor(?:ed|ship)?|advertisement|ad break)\b", re.I)

# A selected frame with its relevance record, which is None when time slots chose it.
type _Pick = tuple[DecodedFrame, dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class _Embedding:
    """Texts are embedded before the video downloads, so a bad key, quota, or model fails before the slow work."""

    embedder: GeminiEmbeddingClient | LocalImageEmbeddingClient
    cutoffs: EmbeddingCutoffs
    warning: str | None
    targets: list[TextTarget]
    texts: EmbeddingResult


def sample_vision(
    video_ref: VideoRef,
    *,
    max_frames: int = DEFAULT_VISION_MAX_FRAMES,
    rows: int = DEFAULT_VISION_ROWS,
    columns: int = DEFAULT_VISION_COLUMNS,
    resolution: VisionResolution = DEFAULT_VISION_RESOLUTION,
    timestamp_annotation: bool = True,
    start: float | None = None,
    end: float | None = None,
    min_similarity: float | None = None,
    min_gap_seconds: float = DEFAULT_VISION_MIN_GAP_SECONDS,
    local_only: bool = False,
    auto_clean: bool = True,
    metadata: dict[str, Any] | None = None,
    document: TranscriptDocument | None = None,
    fresh: bool = False,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """Rank the video's keyframes by the first available route and write the picks as collages in a new directory in the working directory's archive.

    Routes in order: the decision model with DECISION_API_KEY, Gemini embeddings with EMBEDDING_API_KEY, the local embedding model when installed, then time slots.
    `start` and `end` in seconds limit the frames, and the speech they are ranked against, to that part of the video; an `end` past the video stops at its end.
    local_only skips both APIs and speech-to-text, so the local model ranks against captions only.
    The transcript is fetched only when a model will rank and no document is given.
    Without captions it comes from speech-to-text, which ranks frames only when its route reports timing.
    min_similarity applies to embedding routes, and None means the cutoff tuned for the ranking model.
    max_frames limits the output, never the candidate search.
    video_path is a downloaded copy of the video from `start` to `end`, at least as tall as the screenshots, to pick from instead of downloading one; vision never deletes it.
    The picked frames are saved, and a later run whose picking options, range, ranking model, and toolkit version match lays them out again under its rows, columns, resolution, and timestamp annotation without downloading, fetching a transcript, or ranking, provided the saved frames are at least that resolution; usage is then None and picks_reused_from names the saved run.
    A given document plays no part in that match; fresh always picks afresh, and so does auto_clean=False, since it keeps the downloaded media.
    """
    if not 1 <= max_frames <= MAX_VISION_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_VISION_FRAMES}")
    if not 1 <= rows <= MAX_VISION_ROWS:
        raise ValueError(f"rows must be between 1 and {MAX_VISION_ROWS}")
    if not 1 <= columns <= MAX_VISION_COLUMNS:
        raise ValueError(f"columns must be between 1 and {MAX_VISION_COLUMNS}")
    width, height = VISION_RESOLUTIONS[resolution]
    if min_similarity is not None and (
        not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1
    ):
        raise ValueError("min_similarity must be finite and between -1 and 1")
    if not math.isfinite(min_gap_seconds) or min_gap_seconds < 0:
        raise ValueError("min_gap_seconds must be finite and non-negative")
    video_id = require_video_id(video_ref)
    decision = None if local_only else decision_config()
    gemini_key = None if local_only else gemini_embedding_key()
    pick_options = {
        "max_frames": max_frames,
        "min_similarity": min_similarity,
        "min_gap_seconds": min_gap_seconds,
        "local_only": local_only,
        "start": start,
        "end": end,
        "ranking_model": _ranking_model(decision, gemini_key),
        "toolkit_version": version("youtube-toolkit"),
    }
    saved = (
        None
        if fresh or not auto_clean
        else _find_saved_picks(video_id, pick_options, height)
    )
    if saved is not None:
        source, saved_result = saved
        return _lay_out_saved_picks(
            source,
            saved_result,
            rows=rows,
            columns=columns,
            frame_size=(width, height),
            timestamp_annotation=timestamp_annotation,
        )
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
    first = 0.0 if start is None else start
    last = duration if end is None else min(end, duration)
    if not 0 <= first < last:
        raise ValueError(
            "A vision range must start before it ends and before the video ends"
        )
    if last - first > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError(
            f"Vision covers at most {MAX_VIDEO_DURATION_SECONDS // 3600} hours of video; narrow it with --start and --end"
        )
    excluded_intervals = _excluded_intervals(metadata, duration)
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
            # Speech density is judged over the whole video, so the range applies after targets are prepared.
            targets = [
                target
                for target in prepare_targets(document, excluded_intervals)
                if target.end_seconds > first and target.start_seconds < last
            ]
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
            if video_path is None:
                video_path = work_dir / f"{video_id}.mp4"
                client.download_silent_video(
                    metadata,
                    video_path,
                    height=VISION_SOURCE_HEIGHT,
                    section=None if start is None and end is None else (first, last),
                    max_bytes=MAX_VIDEO_BYTES,
                    timeout_seconds=YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
                )
            candidates = collect_candidates(
                video_path,
                ffmpeg_path,
                start=first,
                duration=last - first,
                frame_size=(CANDIDATE_WIDTH, CANDIDATE_HEIGHT),
                directory=work_dir / "candidates",
                excluded_intervals=excluded_intervals,
                distinct=True,
            )
            if decision:
                picks, decision_ranking = _rank_by_decisions(
                    candidates,
                    document if targets else None,
                    decision,
                    duration=last - first,
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
                        start=first,
                        duration=last - first,
                        max_frames=max_frames,
                        min_gap_seconds=min_gap_seconds,
                    )
                ]
            picks.sort(key=lambda pick: pick[0].timestamp_seconds)
            # Candidates stay at the size picking was tuned on, so only the picks are decoded again above it.
            if height > CANDIDATE_HEIGHT:
                picks = [
                    (
                        replace(
                            frame,
                            image=decode_frame(
                                video_path,
                                frame.timestamp_seconds - first,
                                ffmpeg_path,
                                width=width,
                                height=height,
                            ),
                        ),
                        relevance,
                    )
                    for frame, relevance in picks
                ]
            frames_directory = destination / "frames"
            frames_directory.mkdir()
            stored = [store_frame(frames_directory, frame) for frame, _ in picks]
            collage_paths = render_collages(
                [frame for frame, _ in picks],
                destination / f"{video_id}_vision",
                rows=rows,
                columns=columns,
                frame_size=(width, height),
                timestamp_annotation=timestamp_annotation,
            )
        result = {
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
            "embedding_usage": None
            if embedding_usage is None
            else asdict(embedding_usage),
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
            "picks_reused_from": None,
            "frames": [
                {
                    **frame.to_dict(),
                    "image_path": str(stored_frame.image_path),
                    "relevance": relevance,
                }
                for (frame, relevance), stored_frame in zip(picks, stored, strict=True)
            ],
        }
        _save_picks(destination, pick_options, max(height, CANDIDATE_HEIGHT), result)
    except OSError as error:
        raise RuntimeError(
            f"Vision artifact filesystem error under {destination}: {error}"
        ) from error
    return result


def _ranking_model(
    decision: DecisionConfig | None, gemini_key: str | None
) -> str | None:
    """The model the route order would try first; None means time slots."""
    if decision:
        return decision.model
    if gemini_key:
        return GEMINI_EMBEDDING_MODEL
    if local_embedding_installed():
        return local_embedding_model()
    return None


def _find_saved_picks(
    video_id: str, options: dict[str, Any], height: int
) -> tuple[Path, dict[str, Any]] | None:
    """The newest saved run whose options match and whose frames are at least height tall, with its result."""
    for directory in artifact_directories("vision", video_id):
        path = directory / PICKS_FILE
        if not path.is_file():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["options"] == options and saved["frame_height"] >= height:
            return directory, saved["result"]
    return None


def _save_picks(
    destination: Path,
    options: dict[str, Any],
    frame_height: int,
    result: dict[str, Any],
) -> None:
    document = {"options": options, "frame_height": frame_height, "result": result}
    (destination / PICKS_FILE).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _lay_out_saved_picks(
    source: Path,
    saved: dict[str, Any],
    *,
    rows: int,
    columns: int,
    frame_size: tuple[int, int],
    timestamp_annotation: bool,
) -> dict[str, Any]:
    destination = None
    try:
        frames = [
            StoredFrame(
                frame["timestamp_seconds"],
                Path(frame["image_path"]),
                VisualScores(**frame["visual_scores"]),
            ).load()
            for frame in saved["frames"]
        ]
        destination = create_artifact_directory("vision", saved["video_id"])
        collage_paths = render_collages(
            frames,
            destination / f"{saved['video_id']}_vision",
            rows=rows,
            columns=columns,
            frame_size=frame_size,
            timestamp_annotation=timestamp_annotation,
        )
    except OSError as error:
        raise RuntimeError(
            f"Vision artifact filesystem error under {destination or source}: {error}"
        ) from error
    return {
        **saved,
        "embedding_usage": None,
        "decision_usage": None,
        "directory": str(destination),
        "collage_paths": [str(path) for path in collage_paths],
        "media_dir": None,
        "picks_reused_from": str(source),
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
    duration: float,
    max_frames: int,
    min_gap_seconds: float,
) -> tuple[list[_Pick], DecisionRanking]:
    ranking = rank_frames(
        candidates, document, config, even_gap_seconds=duration / max_frames
    )
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
