"""Own embedding transport, bounded batches, and per-item failures while sharing one connection pool within a search."""

from __future__ import annotations

from typing import Self, overload

from openai import APIError, OpenAI

from ..config import EmbeddingConfig, embedding_concurrency, embedding_config_from_env
from ..execution import collect_concurrently


class EmbeddingError(RuntimeError):
    """Covers invalid embedding configuration as well as failed or empty API responses."""


class EmbeddingClient:
    """A fresh SDK client per text would open a new TLS connection for every request; one shared client keeps connections alive between them."""

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

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(
        self, text: list[str]
    ) -> tuple[dict[str, list[float]], dict[str, str]]: ...

    def embed(
        self, text: str | list[str]
    ) -> list[float] | tuple[dict[str, list[float]], dict[str, str]]:
        """A list never raises for individual texts: each distinct text is requested once, and failures come back as messages beside the successful vectors."""
        if isinstance(text, list):
            if not text:
                return {}, {}
            jobs = {item: item for item in text}
            vectors, failures = collect_concurrently(
                jobs, self.embed, max_workers=embedding_concurrency(len(jobs))
            )
            return (
                {item: vectors[item] for item in jobs if item in vectors},
                {item: str(failures[item]) for item in jobs if item in failures},
            )

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
            if not values:
                raise EmbeddingError("Embedding request returned an empty vector")
            return [float(value) for value in values]
        except (APIError, ValueError) as error:
            raise EmbeddingError(str(error)) from error
