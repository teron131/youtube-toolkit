"""Environment settings, defaults, and validation shared by retrieval clients, search, and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlparse

DEFAULT_SEARCH_MIN_VIEWS = 10_000
DEFAULT_SEARCH_MIN_DURATION = "3:00"
DEFAULT_SEARCH_MAX_DURATION = "8:00:00"
DEFAULT_YOUTUBE_SEARCH_LIMIT = 10
DEFAULT_CHANNEL_SEARCH_LIMIT = 30

DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONALITY = 1536
DEFAULT_EMBEDDING_CONCURRENCY = 16
MAX_EMBEDDING_CONCURRENCY = 50


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Validated embedding endpoint configuration including the expected output dimensionality."""

    model: str
    api_key: str
    base_url: str
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONALITY

    def __post_init__(self) -> None:
        """Normalize endpoint fields once so request code can treat config as validated."""
        model = self.model.strip()
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        if not model:
            raise ValueError("model must not be empty")
        if not api_key:
            raise ValueError("api_key must not be empty")
        if urlparse(base_url).scheme not in {"http", "https"}:
            raise ValueError("base_url must use http or https")
        if self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be greater than 0")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)


def embedding_config_from_env() -> EmbeddingConfig:
    """Resolve embedding config from EMBEDDING_* variables, requiring only the API key."""
    api_key = config_value("EMBEDDING_API_KEY")
    if api_key is None:
        raise ValueError("Missing model config value: EMBEDDING_API_KEY")
    return EmbeddingConfig(
        model=config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        api_key=api_key,
        base_url=config_value("EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL,
        output_dimensionality=config_int_value(
            "EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONALITY
        ),
    )


def embedding_concurrency(job_count: int) -> int:
    """Clamp embedding worker count to the configured limit and the amount of pending work."""
    configured_concurrency = config_int_value(
        "EMBEDDING_CONCURRENCY", DEFAULT_EMBEDDING_CONCURRENCY
    )
    return min(
        job_count,
        max(
            1,
            min(configured_concurrency, MAX_EMBEDDING_CONCURRENCY),
        ),
    )


def config_value(name: str) -> str | None:
    """Read one embedding setting from the process environment, treating blank values as unset."""
    value = os.environ.get(name, "").strip()
    return value or None


def config_int_value(
    name: str,
    default: int,
) -> int:
    """Read a positive integer embedding setting with the provided default."""
    value = config_value(name)
    if value is None:
        return default
    if not value.isdigit():
        raise ValueError(f"Model config value must be a positive integer: {name}")
    return int(value)
