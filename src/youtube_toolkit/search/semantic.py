"""Semantic search ranks filtered candidates by title, then by transcript chunks, entirely within one invocation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..config import embedding_concurrency
from ..embeddings import embed_text
from ..execution import collect_concurrently
from ..schemas import TranscriptDocument
from ..transcript import fetch_transcript
from .filters import (
    candidate_id,
    candidate_metadata,
    normalized_upload_date,
    video_result,
)
from .request import SearchRequest, resolve_return_type

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
    """One live transcript chunk carrying its result metadata until embedding completes."""

    video_id: str
    chunk_index: int
    text: str
    candidate: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One prepared transcript chunk after in-memory semantic scoring."""

    prepared: PreparedChunk
    similarity: float


def search_transcripts(
    request: SearchRequest, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Shortlist by title, fetch fresh transcripts, embed chunks, and project chunk or video results.

    Partial title, transcript, or chunk-embedding failures become per-result warnings; the search fails only when no transcript chunk could be embedded.
    """
    candidates = [
        {
            **candidate,
            "upload_date": normalized_upload_date(candidate.get("upload_date")),
        }
        for candidate in candidates
    ]
    if not candidates:
        raise unprepared_embeddings_error([])
    query_embedding = embed_text(request.query)
    candidates, title_failures = shortlist_by_title(
        request, candidates, query_embedding
    )
    documents, transcript_failures = fetch_semantic_transcripts(candidates)
    chunks = prepare_transcript_chunks(candidates, documents)
    vectors, chunk_failures = embed_texts_concurrently(
        {chunk.text: chunk.text for chunk in chunks}
    )
    if not vectors:
        raise unprepared_embeddings_error(
            [*transcript_failures.values(), *chunk_failures.values()]
        )
    scored = score_chunks(chunks, vectors, query_embedding)

    shared_warnings = [
        *compact_stage_failures("title embedding", title_failures),
        *compact_stage_failures("transcript preparation", transcript_failures),
    ]
    chunk_warnings = chunk_failure_warnings(chunks, chunk_failures)

    def warnings_for(video_id: str) -> list[str]:
        return [*shared_warnings, *chunk_warnings.get(video_id, [])]

    if resolve_return_type(request) == "chunk":
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
    query_embedding: list[float],
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
    vectors, failures = embed_texts_concurrently(titles)
    if not vectors:
        return candidates, failures
    scored = sorted(
        (
            (cosine_similarity(query_embedding, vectors[video_id]), candidate)
            for candidate in candidates
            if (video_id := candidate_id(candidate)) in vectors
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
    """Fetch fresh transcripts concurrently while retaining per-video failures."""
    jobs = {
        video_id: video_id
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None
    }
    documents, failures = collect_concurrently(
        jobs, fetch_transcript, max_workers=MAX_CONCURRENT_TRANSCRIPT_FETCHES
    )
    return documents, {video_id: str(error) for video_id, error in failures.items()}


def prepare_transcript_chunks(
    candidates: list[dict[str, Any]],
    documents: dict[str, TranscriptDocument],
) -> list[PreparedChunk]:
    """Split fetched transcripts in candidate order while keeping each source candidate attached."""
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


def embed_texts_concurrently(
    texts: dict[str, str],
) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Embed independent texts concurrently without letting one remote failure stop the stage."""
    if not texts:
        return {}, {}

    def embed(text: str) -> list[float]:
        vector = embed_text(text)
        if not vector:
            raise RuntimeError("Embedding request returned an empty vector")
        return vector

    vectors, failures = collect_concurrently(
        texts,
        embed,
        max_workers=embedding_concurrency(len(texts)),
    )
    return vectors, {key: str(error) for key, error in failures.items()}


def score_chunks(
    chunks: list[PreparedChunk],
    vectors: dict[str, list[float]],
    query_embedding: list[float],
) -> list[ScoredChunk]:
    """Rank embedded chunks by query similarity after applying the result threshold."""
    scored: list[ScoredChunk] = []
    for chunk in chunks:
        vector = vectors.get(chunk.text)
        if vector is None:
            continue
        similarity = cosine_similarity(query_embedding, vector)
        if similarity < RESULT_MIN_SIMILARITY:
            continue
        scored.append(ScoredChunk(prepared=chunk, similarity=similarity))
    return sorted(scored, key=lambda chunk: chunk.similarity, reverse=True)


def chunk_result(chunk: ScoredChunk, *, warnings: list[str]) -> dict[str, Any]:
    """Chunk results carry the passage, its score, and the source video's public metadata."""
    result = {
        "type": "chunk",
        "video_id": chunk.prepared.video_id,
        "chunk_index": chunk.prepared.chunk_index,
        "text": chunk.prepared.text,
        "score": chunk.similarity,
        "video": candidate_metadata(chunk.prepared.candidate),
    }
    if warnings:
        result["warnings"] = warnings
    return result


def chunk_failure_warnings(
    chunks: list[PreparedChunk],
    failures: dict[str, str],
) -> dict[str, list[str]]:
    """Project text-keyed embedding failures into compact per-video warnings."""
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
    """Summarize partial stage failures without flooding every successful result."""
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
    """Split normalized transcript text into overlapping chunks for embedding search."""
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity with C-level dot products, scoring mismatched or zero vectors as zero."""
    if len(left) != len(right):
        return 0.0
    norms = math.sqrt(math.sumprod(left, left) * math.sumprod(right, right))
    if not norms:
        return 0.0
    return math.sumprod(left, right) / norms
