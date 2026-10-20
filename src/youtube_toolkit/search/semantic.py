"""Semantic search ranks filtered candidates by title, then by transcript chunks, entirely within one invocation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..execution import collect_concurrently
from ..schemas import TranscriptDocument
from ..transcript import TranscriptOptions, fetch_transcript
from .embeddings import EmbeddingClient
from .preprocessing import (
    ChunkResult,
    VideoResult,
    candidate_id,
    normalized_upload_date,
    video_fields,
    video_result,
)
from .request import ResolvedRequest, SearchRequest

MAX_CONCURRENT_TRANSCRIPT_FETCHES = 16
TRANSCRIPT_CHUNK_CHARS = 1200
TRANSCRIPT_CHUNK_OVERLAP_CHARS = 200
TITLE_PREFILTER_FACTOR = 3
TITLE_PREFILTER_MIN = 15
TITLE_PREFILTER_MAX = 60
TITLE_MIN_SIMILARITY = 0.3
RESULT_MIN_SIMILARITY = 0.5


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    video_id: str
    chunk_index: int
    text: str
    candidate: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    prepared: PreparedChunk
    similarity: float


def search_transcripts(
    resolved: ResolvedRequest, candidates: list[dict[str, Any]]
) -> list[VideoResult] | list[ChunkResult]:
    """Title, transcript, and chunk-embedding failures become per-result warnings; the search fails only when a query cannot be embedded or no transcript chunk could be."""
    request = resolved.request
    candidates = [
        {
            **candidate,
            "upload_date": normalized_upload_date(candidate.get("upload_date")),
        }
        for candidate in candidates
    ]
    if not candidates:
        raise unprepared_embeddings_error([])
    with EmbeddingClient() as embedder:
        query_embeddings = [
            embedder.embed(query) for query in dict.fromkeys(request.queries)
        ]
        candidates, title_failures = shortlist_by_title(
            request, candidates, query_embeddings, embedder=embedder
        )
        documents, transcript_failures = fetch_semantic_transcripts(candidates)
        chunks = prepare_transcript_chunks(candidates, documents)
        vectors, chunk_failures = embedder.embed([chunk.text for chunk in chunks])
    if not vectors:
        raise unprepared_embeddings_error(
            [*transcript_failures.values(), *chunk_failures.values()]
        )
    scored = score_chunks(chunks, vectors, query_embeddings)

    shared_warnings = [
        *compact_stage_failures("title embedding", title_failures),
        *compact_stage_failures("transcript preparation", transcript_failures),
    ]
    chunk_warnings = chunk_failure_warnings(chunks, chunk_failures)

    def warnings_for(video_id: str) -> list[str]:
        return [*shared_warnings, *chunk_warnings.get(video_id, [])]

    if resolved.return_type == "chunk":
        return [
            chunk_result(chunk, warnings=warnings_for(chunk.prepared.video_id))
            for chunk in scored[: request.limit]
        ]

    video_scores: dict[str, float] = {}
    for chunk in scored:
        video_scores.setdefault(chunk.prepared.video_id, chunk.similarity)
    ranked_candidates = sorted(
        (
            {**candidate, "score": video_scores[video_id]}
            for candidate in candidates
            if (video_id := candidate_id(candidate)) in video_scores
        ),
        key=lambda candidate: candidate["score"],
        reverse=True,
    )
    return [
        video_result(candidate, warnings=warnings_for(str(candidate_id(candidate))))
        for candidate in ranked_candidates[: request.limit]
    ]


def shortlist_by_title(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    query_embeddings: list[list[float]],
    *,
    embedder: EmbeddingClient,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Score titles before fetching transcripts to bound transcript and chunk-embedding work.

    Candidates below the title threshold are dropped unless none pass; untitled or unembedded candidates are dropped only when some title scored.
    """
    titles = {
        video_id: title.strip()
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None
        and isinstance((title := candidate.get("title")), str)
        and title.strip()
    }
    vectors, text_failures = embedder.embed(list(titles.values()))
    failures = {
        video_id: text_failures[title]
        for video_id, title in titles.items()
        if title in text_failures
    }
    if not vectors:
        return candidates, failures
    scored = sorted(
        (
            (best_similarity(query_embeddings, vectors[title]), candidate)
            for candidate in candidates
            if (video_id := candidate_id(candidate)) in titles
            and (title := titles[video_id]) in vectors
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    relevant = [
        candidate for score, candidate in scored if score >= TITLE_MIN_SIMILARITY
    ]
    limit = min(
        len(candidates),
        TITLE_PREFILTER_MAX,
        max(request.limit * TITLE_PREFILTER_FACTOR, TITLE_PREFILTER_MIN),
    )
    return (relevant or [candidate for _score, candidate in scored])[:limit], failures


def fetch_semantic_transcripts(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, TranscriptDocument], dict[str, str]]:
    """Speech-to-text is never used here: a search should stay as cheap to repeat as a YouTube search, and transcribing every uncaptioned candidate would not be."""
    jobs = {
        video_id: (video_id, candidate.get("metadata"))
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None
    }

    def fetch(job: tuple[str, dict[str, Any] | None]) -> TranscriptDocument:
        video_id, metadata = job
        return fetch_transcript(
            video_id, options=TranscriptOptions(allow_stt=False), metadata=metadata
        )

    documents, failures = collect_concurrently(
        jobs, fetch, max_workers=MAX_CONCURRENT_TRANSCRIPT_FETCHES
    )
    return documents, {video_id: str(error) for video_id, error in failures.items()}


def prepare_transcript_chunks(
    candidates: list[dict[str, Any]],
    documents: dict[str, TranscriptDocument],
) -> list[PreparedChunk]:
    prepared: list[PreparedChunk] = []
    for candidate in candidates:
        video_id = candidate_id(candidate)
        if video_id is None or video_id not in documents:
            continue
        prepared.extend(
            PreparedChunk(
                video_id=video_id,
                chunk_index=chunk_index,
                text=text,
                candidate=candidate,
            )
            for chunk_index, text in enumerate(
                split_transcript_chunks(documents[video_id].text)
            )
        )
    return prepared


def score_chunks(
    chunks: list[PreparedChunk],
    vectors: dict[str, list[float]],
    query_embeddings: list[list[float]],
) -> list[ScoredChunk]:
    scored: list[ScoredChunk] = []
    for chunk in chunks:
        vector = vectors.get(chunk.text)
        if vector is None:
            continue
        similarity = best_similarity(query_embeddings, vector)
        if similarity < RESULT_MIN_SIMILARITY:
            continue
        scored.append(ScoredChunk(prepared=chunk, similarity=similarity))
    return sorted(scored, key=lambda chunk: chunk.similarity, reverse=True)


def chunk_result(chunk: ScoredChunk, *, warnings: list[str]) -> ChunkResult:
    result: ChunkResult = {
        "type": "chunk",
        "video_id": chunk.prepared.video_id,
        "chunk_index": chunk.prepared.chunk_index,
        "text": chunk.prepared.text,
        "score": chunk.similarity,
        "video": video_fields(chunk.prepared.candidate),
    }
    if warnings:
        result["warnings"] = warnings
    return result


def chunk_failure_warnings(
    chunks: list[PreparedChunk],
    failures: dict[str, str],
) -> dict[str, list[str]]:
    failed_by_video: dict[str, list[str]] = {}
    for chunk in chunks:
        error = failures.get(chunk.text)
        if error is not None:
            failed_by_video.setdefault(chunk.video_id, []).append(error)
    return {
        video_id: [
            f"{video_id}: {len(failed)} transcript chunk embedding(s) failed: {failed[0]}"
        ]
        for video_id, failed in failed_by_video.items()
    }


def compact_stage_failures(
    stage: str,
    failures: dict[str, str],
) -> list[str]:
    """Stage warnings attach to every result, so only the first three failures are spelled out."""
    if not failures:
        return []
    details = "; ".join(f"{key}: {error}" for key, error in list(failures.items())[:3])
    return [f"{stage} skipped {len(failures)} item(s): {details}"]


def unprepared_embeddings_error(failures: list[str]) -> RuntimeError:
    detail = "; ".join(failures[:3]) or "no transcript chunks were available"
    return RuntimeError(
        f"semantic search could not prepare transcript embeddings: {detail}"
    )


def split_transcript_chunks(
    transcript: str,
    *,
    chunk_chars: int = TRANSCRIPT_CHUNK_CHARS,
    overlap_chars: int = TRANSCRIPT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    clean_transcript = " ".join(transcript.split())
    if not clean_transcript:
        return []
    if chunk_chars <= 0:
        raise ValueError("transcript chunk size must be greater than 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError(
            "transcript chunk overlap must be non-negative and smaller than chunk size"
        )

    chunks: list[str] = []
    start = 0
    while start < len(clean_transcript):
        end = min(len(clean_transcript), start + chunk_chars)
        chunk = clean_transcript[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(clean_transcript):
            break
        start = end - overlap_chars
    return chunks


def best_similarity(query_embeddings: list[list[float]], vector: list[float]) -> float:
    """Score a text by its closest query, so no query outranks another."""
    return max(cosine_similarity(query, vector) for query in query_embeddings)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    norms = math.sqrt(math.sumprod(left, left) * math.sumprod(right, right))
    if not norms:
        return 0.0
    return math.sumprod(left, right) / norms
