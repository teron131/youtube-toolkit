"""Public storage API for YouTube toolkit state."""

from __future__ import annotations

from ..model_config import (
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)
from .blacklist import (
    delete_blacklist_channel,
    list_blacklist_channel_identities,
    upsert_blacklist_channel,
)
from .candidates import index_search_candidate_titles, search_candidate_title_scores, store_search_candidates
from .connection import (
    DATABASE_FILENAME,
    STORAGE_DIRNAME,
    database_schema_health,
    get_connection,
    resolve_storage_dir,
)
from .embeddings import EmbeddingSearchError, embed_text
from .search import get_stored_video, search_stored_videos
from .semantic import (
    TRANSCRIPT_CHUNK_CHARS,
    TRANSCRIPT_CHUNK_OVERLAP_CHARS,
    analyze_transcript_chunk_search,
    auto_index_transcript_chunks,
    auto_index_video_embedding,
    auto_index_video_title_embedding,
    explain_transcript_chunk_search,
    index_transcript_chunks,
    index_video_embedding,
    index_video_title_embedding,
    index_video_title_embeddings,
    list_video_ids_for_embedding_index,
    search_transcript_chunk_scores,
    search_transcript_chunks,
    search_video_title_scores,
    search_video_titles,
    split_transcript_chunks,
)
from .videos import store_channel_listing, upsert_video, upsert_video_summary
from .watchlist import (
    delete_watchlist_channel,
    list_missing_watchlist_video_ids,
    list_recent_watchlist_video_payloads,
    list_watchlist_channel_refs,
    list_watchlist_video_ids_missing_embeddings,
    store_watchlist_channel_listing,
    upsert_watchlist_channel,
)

__all__ = [
    "DATABASE_FILENAME",
    "DEFAULT_EMBEDDING_DIMENSIONALITY",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "STORAGE_DIRNAME",
    "TRANSCRIPT_CHUNK_CHARS",
    "TRANSCRIPT_CHUNK_OVERLAP_CHARS",
    "EmbeddingSearchError",
    "analyze_transcript_chunk_search",
    "auto_index_transcript_chunks",
    "auto_index_video_embedding",
    "auto_index_video_title_embedding",
    "database_schema_health",
    "delete_blacklist_channel",
    "delete_watchlist_channel",
    "embed_text",
    "explain_transcript_chunk_search",
    "get_connection",
    "get_stored_video",
    "index_search_candidate_titles",
    "index_transcript_chunks",
    "index_video_embedding",
    "index_video_title_embedding",
    "index_video_title_embeddings",
    "list_blacklist_channel_identities",
    "list_missing_watchlist_video_ids",
    "list_recent_watchlist_video_payloads",
    "list_video_ids_for_embedding_index",
    "list_watchlist_channel_refs",
    "list_watchlist_video_ids_missing_embeddings",
    "resolve_storage_dir",
    "search_candidate_title_scores",
    "search_stored_videos",
    "search_transcript_chunk_scores",
    "search_transcript_chunks",
    "search_video_title_scores",
    "search_video_titles",
    "split_transcript_chunks",
    "store_channel_listing",
    "store_search_candidates",
    "store_watchlist_channel_listing",
    "upsert_blacklist_channel",
    "upsert_video",
    "upsert_video_summary",
    "upsert_watchlist_channel",
]
