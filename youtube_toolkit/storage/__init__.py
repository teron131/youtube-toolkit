"""Public storage API for YouTube toolkit state."""

from __future__ import annotations

from ..model_config import (
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)
from .catalog import (
    delete_blacklist_channel,
    delete_watchlist_channel,
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
    get_connection,
    resolve_storage_dir,
)
from .search import get_stored_video, search_stored_videos
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
    "delete_blacklist_channel",
    "delete_watchlist_channel",
    "embed_text_with_gemini",
    "get_connection",
    "get_stored_video",
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
