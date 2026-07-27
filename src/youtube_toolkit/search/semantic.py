"""Semantic title, transcript, and chunk search stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Any

from ..capabilities import capability_available
from ..embeddings import embed_text
from ..execution import collect_concurrently
from ..model_config import embedding_concurrency
from ..schemas import EmbeddingIndexResult, TranscriptDocument
from ..semantic import cosine_similarity, split_transcript_chunks
from ..transcript import fetch_transcript, stored_transcript_document
from ..youtube.caption import fetch_live_transcript
from .filters import (
    candidate_id,
    candidate_ids,
    chunk_result,
    rank_by_value,
    video_result,
)
from .request import SearchRequest, SearchReturnType

if TYPE_CHECKING:
    from ..storage.connection import LanceConnection

MAX_CONCURRENT_TRANSCRIPT_FETCHES = 16


def _is_database_write_permission_error(error: BaseException) -> bool:
    """Identify storage permission failures only when the storage capability is installed."""
    if not capability_available("storage"):
        return False
    from ..storage.connection import is_database_write_permission_error as is_permission_error

    return is_permission_error(error)


@dataclass(slots=True)
class SemanticSearchState:
    """Carry database availability, the query vector, and degradation warnings across independent stages."""

    database_available: bool = True
    query_embedding: list[float] | None = None
    warnings: list[str] = field(default_factory=list)

    def disable_database(self) -> None:
        """Prevent later stages from retrying a database already known to be unavailable."""
        self.database_available = False
        warning = "Database unavailable; semantic stages continued in memory"
        if warning not in self.warnings:
            self.warnings.append(warning)


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
    """Expose database-ready video IDs and in-memory vectors through one chunk-stage contract."""

    chunks: list[PreparedChunk]
    database_video_ids: frozenset[str]
    vectors: dict[str, list[float]]
    failures: dict[str, str]
    warnings_by_video_id: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class SemanticScores:
    """Carry backend-independent chunk and per-video scores into result projection."""

    chunks: list[ScoredChunk]
    videos: dict[str, float]
    failures: dict[str, str]


def query_vector(
    state: SemanticSearchState,
    query: str,
) -> list[float]:
    """Embed the query once even when multiple stages fall back to memory."""
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
        if (video_id := candidate_id(candidate)) is not None and isinstance((title := candidate.get("title")), str) and title.strip()
    }
    vectors, failures = embed_texts_concurrently(title_texts)
    if not vectors:
        return candidates, failures
    scores = {video_id: cosine_similarity(query_embedding, vector) for video_id, vector in vectors.items()}
    return rank_title_candidates(request, candidates, scores), failures


def rank_title_candidates(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Apply title-score ranking, soft thresholding, and prefilter limits consistently across backends."""
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
    threshold_candidates = [candidate for candidate in ranked_candidates if request.title_min_similarity is None or candidate["title_score"] >= request.title_min_similarity]
    if threshold_candidates:
        ranked_candidates = threshold_candidates
    limit = title_prefilter_limit(request, len(candidates))
    return [{key: value for key, value in candidate.items() if key != "title_score"} for candidate in ranked_candidates[:limit]]


TranscriptFetcher = Callable[[str, dict[str, Any]], TranscriptDocument]


def fetch_transcript_with_database(
    video_id: str,
    candidate: dict[str, Any],
) -> TranscriptDocument:
    """Use an already collected DB transcript or the storage-preserving transcript adapter."""
    if candidate.get("source") == "db" and isinstance(candidate.get("transcript"), str):
        return stored_transcript_document(
            {
                **candidate,
                "video_id": video_id,
            }
        )
    return fetch_transcript(
        video_id,
        auto_index=False,
    )


def fetch_transcript_in_memory(
    video_id: str,
    candidate: dict[str, Any],
) -> TranscriptDocument:
    """Use collected DB text when present or fetch captions/STT without touching storage."""
    if candidate.get("source") == "db" and isinstance(candidate.get("transcript"), str):
        return stored_transcript_document(
            {
                **candidate,
                "video_id": video_id,
            }
        )
    return fetch_live_transcript(video_id)


def fetch_transcripts_concurrently(
    candidates: list[dict[str, Any]],
    fetch_document: TranscriptFetcher,
) -> tuple[dict[str, TranscriptDocument], dict[str, Exception]]:
    """Fetch candidate transcripts through one selected adapter while retaining per-video failures."""
    jobs = {
        video_id: (
            video_id,
            candidate,
        )
        for candidate in candidates
        if (video_id := candidate_id(candidate)) is not None
    }
    return collect_concurrently(
        jobs,
        lambda job: fetch_document(*job),
        max_workers=MAX_CONCURRENT_TRANSCRIPT_FETCHES,
    )


def fetch_semantic_transcripts(
    candidates: list[dict[str, Any]],
    state: SemanticSearchState,
) -> tuple[dict[str, TranscriptDocument], dict[str, str]]:
    """Run the transcript stage DB-first, retrying only permission-denied items through the live adapter."""
    fetch_document = fetch_transcript_with_database if state.database_available else fetch_transcript_in_memory
    documents, failures = fetch_transcripts_concurrently(candidates, fetch_document)
    database_failures = {video_id: error for video_id, error in failures.items() if _is_database_write_permission_error(error)}
    if database_failures:
        state.disable_database()
        retry_candidates = [candidate for candidate in candidates if candidate_id(candidate) in database_failures]
        retry_documents, retry_failures = fetch_transcripts_concurrently(
            retry_candidates,
            fetch_transcript_in_memory,
        )
        documents.update(retry_documents)
        failures = {video_id: error for video_id, error in failures.items() if video_id not in database_failures}
        failures.update(retry_failures)
    return documents, {video_id: str(error) for video_id, error in failures.items()}


def prepare_transcript_chunks(
    candidates: list[dict[str, Any]],
    documents: dict[str, TranscriptDocument],
) -> list[PreparedChunk]:
    """Split fetched transcript documents while keeping their source candidate metadata attached."""
    candidates_by_id = {video_id: candidate for candidate in candidates if (video_id := candidate_id(candidate)) is not None}
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
        if request.result_min_similarity is not None and similarity < request.result_min_similarity:
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
        warnings_by_video_id[video_id] = [f"{video_id}: {len(failed)} transcript chunk embedding(s) failed: {failed[0]}"]
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


def prefilter_titles_with_database(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    """Stage and score candidate titles through their LanceDB-owned indexes."""
    from ..storage import get_connection, store_search_candidates

    with get_connection() as connection:
        if any(candidate.get("source") != "db" for candidate in candidates):
            store_search_candidates(
                connection,
                session_id,
                request.query,
                candidates,
            )
        scores = title_semantic_scores(
            connection,
            request,
            request.query,
            candidates,
            session_id=session_id,
        )
    if not scores:
        return []
    return rank_title_candidates(request, candidates, scores)


def prefilter_by_title_semantics(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    state: SemanticSearchState,
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    """Run the title stage through LanceDB when available and otherwise score titles in memory."""
    video_ids = candidate_ids(candidates)
    if not request.query.strip() or not video_ids:
        return candidates
    database_candidates: list[dict[str, Any]] = []
    if state.database_available:
        try:
            database_candidates = prefilter_titles_with_database(
                request,
                candidates,
                session_id=session_id,
            )
        except Exception as error:
            if not _is_database_write_permission_error(error):
                raise
            state.disable_database()

    if not database_candidates:
        memory_candidates, failures = prefilter_titles_in_memory(
            request,
            candidates,
            query_vector(state, request.query),
        )
        state.warnings.extend(compact_stage_failures("title embedding", failures))
        return memory_candidates
    return database_candidates


def title_semantic_scores(
    connection: LanceConnection,
    request: SearchRequest,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    session_id: str,
) -> dict[str, float]:
    """Score external search-session titles and stored DB titles with their owned embedding indexes."""
    from ..storage import (
        index_search_candidate_titles,
        index_video_title_embeddings,
        search_candidate_title_scores,
        search_video_title_scores,
    )

    scores: dict[str, float] = {}
    external_ids = candidate_ids([candidate for candidate in candidates if candidate.get("source") != "db"])
    if external_ids:
        index_search_candidate_titles(connection, session_id)
        scores.update(
            search_candidate_title_scores(
                connection,
                session_id,
                query,
                limit=title_prefilter_limit(request, len(external_ids)),
                min_similarity=request.title_min_similarity,
            )
        )

    db_ids = candidate_ids([candidate for candidate in candidates if candidate.get("source") == "db"])
    if db_ids:
        index_video_title_embeddings(connection, db_ids)
        scores.update(
            search_video_title_scores(
                connection,
                query,
                db_ids,
                limit=title_prefilter_limit(request, len(db_ids)),
                min_similarity=request.title_min_similarity,
            )
        )
    return scores


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


def index_transcripts(candidates: list[dict[str, Any]]) -> list[EmbeddingIndexResult]:
    """Run the database chunk-index option for transcripts already prepared by the prior stage."""
    from ..storage import auto_index_transcript_chunks, get_connection

    results: list[EmbeddingIndexResult] = []
    for candidate in candidates:
        video_id = candidate_id(candidate)
        if video_id is None:
            continue
        try:
            with get_connection() as connection:
                results.append(auto_index_transcript_chunks(connection, video_id))
        except Exception as error:
            if _is_database_write_permission_error(error):
                raise
            results.append(
                {
                    "status": "failed",
                    "video_id": video_id,
                    "error": str(error),
                }
            )
    return results


def semantic_ready_ids(
    candidates: list[dict[str, Any]],
    index_results: list[EmbeddingIndexResult],
) -> set[str]:
    """Combine newly indexed and already stored chunk IDs that can satisfy semantic search."""
    from ..storage import get_connection
    from ..storage.connection import filter_in_values

    video_ids = candidate_ids(candidates)
    indexed_ids = {str(result["video_id"]) for result in index_results if result.get("status") == "indexed" and result.get("video_id")}
    if not video_ids:
        return indexed_ids
    with get_connection() as connection:
        stored_ids = {
            row["video_id"]
            for row in connection.rows(
                "video_transcript_chunks",
                where_sql=f"video_id IN ({filter_in_values(video_ids)}) AND embedding IS NOT NULL",
                columns=["video_id"],
            )
        }
    return indexed_ids | stored_ids


def index_warnings(
    index_results: list[EmbeddingIndexResult],
    ready_ids: set[str],
) -> dict[str, list[str]]:
    """Return per-video warnings for candidates that failed or skipped transcript indexing."""
    warnings: dict[str, list[str]] = {}
    for result in index_results:
        video_id = result.get("video_id")
        if not isinstance(video_id, str) or video_id in ready_ids:
            continue
        if result.get("status") not in {"failed", "skipped"}:
            continue
        detail = result.get("error") or result.get("detail") or result.get("status", "")
        warnings.setdefault(video_id, []).append(f"{video_id}: {detail}")
    return warnings


def index_chunks_with_database(
    candidates: list[dict[str, Any]],
    documents: dict[str, TranscriptDocument],
) -> tuple[list[EmbeddingIndexResult], set[str]]:
    """Index prepared transcripts and return the candidate IDs searchable through LanceDB."""
    index_candidates = [candidate for candidate in candidates if candidate.get("source") == "db" or candidate_id(candidate) in documents]
    index_results = index_transcripts(index_candidates)
    return index_results, semantic_ready_ids(
        index_candidates,
        index_results,
    )


def prepare_semantic_index(
    candidates: list[dict[str, Any]],
    documents: dict[str, TranscriptDocument],
    state: SemanticSearchState,
) -> SemanticIndex:
    """Run the chunk-index stage with independent database and in-memory implementations."""
    chunks = prepare_transcript_chunks(candidates, documents)
    index_results: list[EmbeddingIndexResult] = []
    database_video_ids: set[str] = set()
    if state.database_available:
        try:
            index_results, database_video_ids = index_chunks_with_database(
                candidates,
                documents,
            )
        except Exception as error:
            if not _is_database_write_permission_error(error):
                raise
            state.disable_database()

    memory_chunks = [chunk for chunk in chunks if chunk.video_id not in database_video_ids]
    vectors, failures = embed_chunks_in_memory(memory_chunks)
    return SemanticIndex(
        chunks=chunks,
        database_video_ids=frozenset(database_video_ids),
        vectors=vectors,
        failures=failures,
        warnings_by_video_id=index_warnings(
            index_results,
            database_video_ids,
        ),
    )


def ensure_semantic_index(
    semantic_index: SemanticIndex,
    transcript_failures: dict[str, str],
) -> None:
    """Require at least one database-ready video or in-memory vector after transcript preparation."""
    if semantic_index.database_video_ids or semantic_index.vectors:
        return
    failures = [*transcript_failures.values(), *semantic_index.failures.values()]
    detail = "; ".join(failures[:3]) or "no transcript chunks were available"
    raise RuntimeError(f"semantic search could not prepare transcript embeddings: {detail}")


def ensure_memory_vectors(
    chunks: list[PreparedChunk],
    vectors: dict[str, list[float]],
) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Complete an in-memory index only for chunk contents not already embedded by the index stage."""
    missing_texts = {key: chunk.text for chunk in chunks if (key := chunk_content_key(chunk.text)) not in vectors}
    embedded, failures = embed_texts_concurrently(missing_texts)
    return {
        **vectors,
        **embedded,
    }, failures


def score_chunks_with_database(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    video_ids: set[str],
    *,
    return_type: SearchReturnType,
) -> tuple[list[ScoredChunk], dict[str, float]]:
    """Score the database-ready portion of a semantic index through LanceDB."""
    from ..storage import get_connection, search_transcript_chunk_scores, search_transcript_chunks

    with get_connection() as connection:
        video_scores = search_transcript_chunk_scores(
            connection,
            request.query,
            sorted(video_ids),
            min_similarity=request.result_min_similarity,
        )
        if return_type != "chunk":
            return [], video_scores
        rows = search_transcript_chunks(
            connection,
            request.query,
            sorted(video_ids),
            limit=request.limit,
            columns=["video_id", "chunk_index", "text"],
            min_similarity=request.result_min_similarity,
        )
    candidates_by_id = {video_id: candidate for candidate in candidates if (video_id := candidate_id(candidate)) is not None}
    scored_chunks = [
        ScoredChunk(
            prepared=PreparedChunk(
                video_id=row["video_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                candidate=candidates_by_id[row["video_id"]],
            ),
            similarity=row["similarity"],
        )
        for row in rows
        if row["video_id"] in candidates_by_id
    ]
    return scored_chunks, video_scores


def score_semantic_index(
    request: SearchRequest,
    candidates: list[dict[str, Any]],
    semantic_index: SemanticIndex,
    state: SemanticSearchState,
    *,
    return_type: SearchReturnType,
) -> SemanticScores:
    """Run DB and memory scoring options against the same prepared-index contract."""
    database_video_ids = set(semantic_index.database_video_ids)
    database_chunks: list[ScoredChunk] = []
    video_scores: dict[str, float] = {}
    if database_video_ids and state.database_available:
        try:
            database_chunks, video_scores = score_chunks_with_database(
                request,
                candidates,
                database_video_ids,
                return_type=return_type,
            )
        except Exception as error:
            if not _is_database_write_permission_error(error):
                raise
            state.disable_database()
            database_video_ids.clear()
            video_scores.clear()
            database_chunks.clear()

    scored_database_video_ids = {chunk.prepared.video_id for chunk in database_chunks} if return_type == "chunk" else set(video_scores)
    memory_chunks = [chunk for chunk in semantic_index.chunks if chunk.video_id not in scored_database_video_ids]
    vectors, failures = ensure_memory_vectors(
        memory_chunks,
        semantic_index.vectors,
    )
    scored_memory_chunks = score_chunks_in_memory(
        request,
        memory_chunks,
        vectors,
        query_vector(state, request.query),
    )
    for scored in scored_memory_chunks:
        video_id = scored.prepared.video_id
        video_scores[video_id] = max(
            scored.similarity,
            video_scores.get(video_id, float("-inf")),
        )
    return SemanticScores(
        chunks=sorted(
            [*database_chunks, *scored_memory_chunks],
            key=lambda chunk: chunk.similarity,
            reverse=True,
        ),
        videos=video_scores,
        failures={
            **semantic_index.failures,
            **failures,
        },
    )


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
    """Project backend-independent semantic scores into the requested public result shape."""
    pipeline_warnings = [*state.warnings]
    pipeline_warnings.extend(compact_stage_failures("transcript preparation", transcript_failures))
    warnings_by_video_id = {video_id: [*warnings] for video_id, warnings in semantic_index.warnings_by_video_id.items()}
    embedding_warnings = chunk_failure_warnings(
        semantic_index.chunks,
        scores.failures,
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
            if (video_id := candidate_id(candidate)) is not None and video_id in scores.videos
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
