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
from .connection import (
    DATABASE_FILENAME,
    STORAGE_DIRNAME,
    database_schema_health,
    get_connection,
    resolve_storage_dir,
)
from .search import get_stored_video, search_stored_videos
from .semantic import (
    EmbeddingSearchError,
    auto_index_video_embedding,
    embed_text_with_gemini,
    index_video_embedding,
    list_video_ids_for_embedding_index,
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
    "EmbeddingSearchError",
    "auto_index_video_embedding",
    "database_schema_health",
    "delete_blacklist_channel",
    "delete_watchlist_channel",
    "embed_text_with_gemini",
    "get_connection",
    "get_stored_video",
    "index_video_embedding",
    "list_blacklist_channel_identities",
    "list_missing_watchlist_video_ids",
    "list_recent_watchlist_video_payloads",
    "list_video_ids_for_embedding_index",
    "list_watchlist_channel_refs",
    "list_watchlist_video_ids_missing_embeddings",
    "resolve_storage_dir",
    "search_stored_videos",
    "store_channel_listing",
    "store_watchlist_channel_listing",
    "upsert_blacklist_channel",
    "upsert_video",
    "upsert_video_summary",
    "upsert_watchlist_channel",
]
