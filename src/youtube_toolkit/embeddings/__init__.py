"""Embedding clients for semantic search and vision, through an API or a local model, plus the result and error types they share."""

from .gemini import GeminiEmbeddingClient
from .local import (
    LocalImageEmbeddingClient,
    LocalTextEmbeddingClient,
    local_embedding_downloaded,
    local_embedding_installed,
)
from .openai import EmbeddingClient
from .results import EmbeddingError, EmbeddingResult, EmbeddingUsage

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "EmbeddingResult",
    "EmbeddingUsage",
    "GeminiEmbeddingClient",
    "LocalImageEmbeddingClient",
    "LocalTextEmbeddingClient",
    "local_embedding_downloaded",
    "local_embedding_installed",
]
