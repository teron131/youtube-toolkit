"""Public storage API for repo-local YouTube toolkit state."""

from __future__ import annotations

from ..model_config import (
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)
from .catalog import (
    list_blacklist_channel_identities,
    list_missing_watchlist_video_ids,
    list_watchlist_channel_inputs,
    store_channel_listing,
    upsert_blacklist_channel,
    upsert_video,
    upsert_video_summary,
    upsert_watchlist_channel,
)
from .connection import (
    DATABASE_FILENAME,
    STORAGE_DIRNAME,
    find_repo_root,
    get_connection,
    resolve_storage_dir,
)
from .search import search_stored_videos
from .semantic import (
    EmbeddingSearchError,
    embed_text_with_gemini,
    index_video_embedding,
)

__all__ = [
    "DATABASE_FILENAME",
    "DEFAULT_EMBEDDING_DIMENSIONALITY",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "STORAGE_DIRNAME",
    "EmbeddingSearchError",
    "embed_text_with_gemini",
    "find_repo_root",
    "get_connection",
    "index_video_embedding",
    "list_blacklist_channel_identities",
    "list_missing_watchlist_video_ids",
    "list_watchlist_channel_inputs",
    "resolve_storage_dir",
    "search_stored_videos",
    "store_channel_listing",
    "upsert_blacklist_channel",
    "upsert_video",
    "upsert_video_summary",
    "upsert_watchlist_channel",
]
