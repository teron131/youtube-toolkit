"""Semantic title, transcript, and chunk search stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from ..embeddings import embed_text
from ..execution import collect_concurrently
from ..model_config import embedding_concurrency
from ..schemas import TranscriptDocument
from ..semantic import cosine_similarity, split_transcript_chunks
from ..transcript import fetch_transcript
from .filters import (
    candidate_id,
    candidate_ids,
    chunk_result,
    rank_by_value,
    video_result,
)
from .request import SearchRequest, SearchReturnType

MAX_CONCURRENT_TRANSCRIPT_FETCHES = 16


@dataclass(slots=True)
class SemanticSearchState:
    """Carry the shared query vector and partial-failure warnings across semantic stages."""

    query_embedding: list[float] | None = None
    warnings: list[str] = field(default_factory=list)


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


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    """Keep prepared transcript chunks, deduplicated vectors, and embedding failures."""

    chunks: list[PreparedChunk]
    vectors: dict[str, list[float]]
    failures: dict[str, str]


@dataclass(frozen=True, slots=True)
class SemanticScores:
    """Carry ranked chunks and per-video scores into result projection."""

    chunks: list[ScoredChunk]
    videos: dict[str, float]


def query_vector(
    state: SemanticSearchState,
    query: str,
) -> list[float]:
    """Embed the query once for both title and transcript scoring."""
    if state.query_embedding is None:
        state.query_embedding = embed_text(query)
    return state.query_embedding


def prefilter_titles_in_memory(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Shortlist fresh candidates by directly scoring their title vectors in memory."""
    title_texts = {
        video_id: title.strip()
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None
        and isinstance((title := candidate.get("title")), str)
        and title.strip()
    }
    vectors, failures = embed_texts_concurrently(title_texts)
    if not vectors:
        return candidates, failures
    scores = {
        video_id: cosine_similarity(query_embedding, vector)
        for video_id, vector in vectors.items()
    }
    return rank_title_candidates(request, candidates, scores), failures


def rank_title_candidates(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Apply title-score ranking, soft thresholding, and prefilter limits consistently."""
    scored_candidates = [
        {
            **candidate,
            "title_score": scores[video_id],
        }
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None and video_id in scores
    ]
    ranked_candidates = rank_by_value(
        scored_candidates,
        value=lambda candidate: candidate.get("title_score"),
    )
    threshold_candidates = [
        candidate
        for candidate in ranked_candidates
        if request.title_min_similarity is None
        or candidate["title_score"] >= request.title_min_similarity
    ]
    if threshold_candidates:
        ranked_candidates = threshold_candidates
    limit = title_prefilter_limit(request, len(candidates))
    return [
        {key: value for key, value in candidate.items() if key != "title_score"}
        for candidate in ranked_candidates[:limit]
    ]


def fetch_semantic_transcripts(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, TranscriptDocument], dict[str, str]]:
    """Fetch directory-cached transcripts concurrently while retaining per-video failures."""
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
    """Split fetched transcript documents while keeping their source candidate metadata attached."""
    candidates_by_id = {
        video_id: candidate
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None
    }
    prepared: list[PreparedChunk] = []
    for video_id, document in documents.items():
        candidate = candidates_by_id.get(video_id)
        if candidate is None:
            continue
        prepared.extend(
            PreparedChunk(
                video_id=video_id,
                chunk_index=chunk_index,
                text=text,
                candidate=candidate,
            )
            for chunk_index, text in enumerate(split_transcript_chunks(document.text))
        )
    return prepared


def embed_chunks_in_memory(
    chunks: list[PreparedChunk],
) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Embed unique transcript chunks once per command and retain content-keyed failures."""
    texts = {chunk_content_key(chunk.text): chunk.text for chunk in chunks}
    return embed_texts_concurrently(texts)


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


def score_chunks_in_memory(
    request: SearchRequest,
    chunks: list[PreparedChunk],
    vectors: dict[str, list[float]],
    query_embedding: list[float],
) -> list[ScoredChunk]:
    """Score available in-memory chunks and apply the public similarity threshold."""
    scored: list[ScoredChunk] = []
    for chunk in chunks:
        vector = vectors.get(chunk_content_key(chunk.text))
        if vector is None:
            continue
        similarity = cosine_similarity(query_embedding, vector)
        if (
            request.result_min_similarity is not None
            and similarity < request.result_min_similarity
        ):
            continue
        scored.append(
            ScoredChunk(
                prepared=chunk,
                similarity=similarity,
            )
        )
    return sorted(
        scored,
        key=lambda chunk: chunk.similarity,
        reverse=True,
    )


def chunk_failure_warnings(
    chunks: list[PreparedChunk],
    failures: dict[str, str],
) -> dict[str, list[str]]:
    """Project content-keyed embedding failures into compact per-video warnings."""
    warnings_by_video_id: dict[str, list[str]] = {}
    failed_by_video: dict[str, list[str]] = {}
    for chunk in chunks:
        error = failures.get(chunk_content_key(chunk.text))
        if error is not None:
            failed_by_video.setdefault(chunk.video_id, []).append(error)
    for video_id, failed in failed_by_video.items():
        warnings_by_video_id[video_id] = [
            f"{video_id}: {len(failed)} transcript chunk embedding(s) failed: {failed[0]}"
        ]
    return warnings_by_video_id


def compact_stage_failures(
    stage: str,
    failures: dict[str, str],
) -> list[str]:
    """Summarize partial stage failures without flooding every successful result."""
    if not failures:
        return []
    details = "; ".join(f"{key}: {error}" for key, error in list(failures.items())[:3])
    return [f"{stage} skipped {len(failures)} item(s): {details}"]


def chunk_content_key(text: str) -> str:
    """Deduplicate identical chunk embedding work within one semantic command."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prefilter_by_title_semantics(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    state: SemanticSearchState,
) -> list[dict[str, Any]]:
    """Score titles before fetching transcripts to bound semantic search work."""
    if not candidates:
        return candidates
    shortlisted, failures = prefilter_titles_in_memory(
        request, candidates, query_vector(state, request.query)
    )
    state.warnings.extend(compact_stage_failures("title embedding", failures))
    return shortlisted


def title_prefilter_limit(
    request: SearchRequest,
    candidate_count: int,
) -> int:
    """Resolve the capped title-prefilter size from request policy and candidate count."""
    requested_limit = max(
        request.limit * request.title_prefilter_factor,
        request.title_prefilter_min,
    )
    return min(
        candidate_count,
        request.title_prefilter_max,
        requested_limit,
    )


def prepare_semantic_index(
    candidates: list[dict[str, Any]],
    documents: dict[str, TranscriptDocument],
) -> SemanticIndex:
    """Embed unique transcript chunks for this search invocation only."""
    chunks = prepare_transcript_chunks(candidates, documents)
    vectors, failures = embed_chunks_in_memory(chunks)
    return SemanticIndex(chunks=chunks, vectors=vectors, failures=failures)


def ensure_semantic_index(
    semantic_index: SemanticIndex,
    transcript_failures: dict[str, str],
) -> None:
    """Require at least one usable vector after transcript preparation."""
    if semantic_index.vectors:
        return
    failures = [*transcript_failures.values(), *semantic_index.failures.values()]
    detail = "; ".join(failures[:3]) or "no transcript chunks were available"
    raise RuntimeError(
        f"semantic search could not prepare transcript embeddings: {detail}"
    )


def score_semantic_index(
    request: SearchRequest,
    semantic_index: SemanticIndex,
    state: SemanticSearchState,
) -> SemanticScores:
    """Rank chunks by similarity and use each video's best chunk as its score."""
    chunks = score_chunks_in_memory(
        request,
        semantic_index.chunks,
        semantic_index.vectors,
        query_vector(state, request.query),
    )
    videos: dict[str, float] = {}
    for scored in chunks:
        video_id = scored.prepared.video_id
        videos[video_id] = max(scored.similarity, videos.get(video_id, float("-inf")))
    return SemanticScores(chunks=chunks, videos=videos)


def project_semantic_results(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    semantic_index: SemanticIndex,
    scores: SemanticScores,
    state: SemanticSearchState,
    transcript_failures: dict[str, str],
    *,
    return_type: SearchReturnType,
) -> list[dict[str, Any]]:
    """Project semantic scores into the requested public result shape."""
    pipeline_warnings = [*state.warnings]
    pipeline_warnings.extend(
        compact_stage_failures("transcript preparation", transcript_failures)
    )
    warnings_by_video_id: dict[str, list[str]] = {}
    embedding_warnings = chunk_failure_warnings(
        semantic_index.chunks,
        semantic_index.failures,
    )
    for video_id in candidate_ids(candidates):
        warnings = warnings_by_video_id.setdefault(video_id, [])
        for warning in pipeline_warnings:
            if warning not in warnings:
                warnings.append(warning)
        for warning in embedding_warnings.get(video_id, []):
            if warning not in warnings:
                warnings.append(warning)

    if return_type == "chunk":
        return [
            chunk_result(
                {
                    "video_id": scored.prepared.video_id,
                    "chunk_index": scored.prepared.chunk_index,
                    "text": scored.prepared.text,
                    "similarity": scored.similarity,
                },
                scored.prepared.candidate,
                warnings=warnings_by_video_id.get(scored.prepared.video_id),
            )
            for scored in scores.chunks[: request.limit]
        ]

    ranked_candidates = rank_by_value(
        [
            {
                **candidate,
                "score": scores.videos[video_id],
            }
            for candidate in candidates
            if (video_id := candidate_id(candidate)) is not None
            and video_id in scores.videos
        ],
        value=lambda candidate: candidate.get("score"),
    )
    return [
        video_result(
            candidate,
            warnings_by_video_id=warnings_by_video_id,
        )
        for candidate in ranked_candidates[: request.limit]
    ]
