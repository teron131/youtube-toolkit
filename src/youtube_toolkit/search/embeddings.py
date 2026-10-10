"""Embedding endpoint access for semantic search, reusing one connection pool across a workflow's requests and translating failures."""

from __future__ import annotations

from typing import Self

from openai import APIError, OpenAI

from ..config import EmbeddingConfig, embedding_config_from_env


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding API cannot return a usable vector."""


class EmbeddingClient:
    """One configured embedding endpoint shared by concurrent requests until closed.

    A fresh SDK client per text would open a new TLS connection for every request; sharing the pool keeps connections alive between requests.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        try:
            self._config = config or embedding_config_from_env()
        except ValueError as error:
            raise EmbeddingError(str(error)) from error
        self._client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            timeout=60,
            max_retries=2,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._client.close()

    def embed(self, text: str) -> list[float]:
        """Request one float embedding with a 60-second timeout and two retries."""
        try:
            response = self._client.embeddings.create(
                model=self._config.model,
                input=text,
                dimensions=self._config.output_dimensionality,
                encoding_format="float",
            )
            values = response.data[0].embedding if response.data else None
            if not isinstance(values, list):
                raise EmbeddingError("Embedding response did not include values")
            return [float(value) for value in values]
        except (APIError, ValueError) as error:
            raise EmbeddingError(str(error)) from error
