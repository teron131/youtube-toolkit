"""Generate embedding vectors from the configured endpoint and translate failures for semantic search."""

from __future__ import annotations

from openai import APIError, OpenAI

from .config import EmbeddingConfig, embedding_config_from_env


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding API cannot return a usable vector."""


def embed_text(
    text: str,
    *,
    config: EmbeddingConfig | None = None,
) -> list[float]:
    """Request float embeddings with a 60-second timeout and two retries."""
    try:
        config = config or embedding_config_from_env()
        with OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=60,
            max_retries=2,
        ) as client:
            response = client.embeddings.create(
                model=config.model,
                input=text,
                dimensions=config.output_dimensionality,
                encoding_format="float",
            )
        values = response.data[0].embedding if response.data else None
        if not isinstance(values, list):
            raise EmbeddingError("Embedding response did not include values")
        return [float(value) for value in values]
    except (APIError, ValueError) as error:
        raise EmbeddingError(str(error)) from error
