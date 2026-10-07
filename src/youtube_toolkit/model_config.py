"""Optional semantic search reads embedding API configuration from the process environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlparse

EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONALITY = 1536
DEFAULT_EMBEDDING_CONCURRENCY = 16
MAX_EMBEDDING_CONCURRENCY = 50
MODEL_CONFIG_KEYS = (
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_CONCURRENCY",
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Validated model API endpoint configuration for embedding requests."""

    model: str
    api_key: str
    base_url: str

    def __post_init__(self) -> None:
        """Normalize endpoint fields once so model callers can treat config as validated."""
        model = self.model.strip()
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        if not model:
            raise ValueError("model must not be empty")
        if not api_key:
            raise ValueError("api_key must not be empty")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError("base_url must use http or https")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)

    def endpoint(self, path: str) -> str:
        """Join an API path to the normalized base URL without leaking slash policy to callers."""
        return f"{self.base_url}/{path.lstrip('/')}"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig(ModelConfig):
    """Embedding-specific model configuration including the expected output dimensionality."""

    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONALITY

    def __post_init__(self) -> None:
        """Preserve base model validation while enforcing embedding vector dimensionality."""
        ModelConfig.__post_init__(self)
        if self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be greater than 0")


def embedding_config_from_env(
    *,
    model: str | None = None,
    base_url: str | None = None,
    output_dimensionality: int | None = None,
) -> EmbeddingConfig:
    """Resolve embedding config while letting tests or specialized callers override non-secret endpoint fields."""
    api_key = required_config_value("EMBEDDING_API_KEY")
    return EmbeddingConfig(
        model=model or config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        api_key=api_key,
        base_url=base_url or config_value("EMBEDDING_BASE_URL") or EMBEDDING_BASE_URL,
        output_dimensionality=output_dimensionality
        or config_int_value("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONALITY),
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
    """Read one supported embedding setting from the process environment."""
    if name not in MODEL_CONFIG_KEYS:
        raise ValueError(f"Unsupported model config key: {name}")
    value = os.environ.get(name, "").strip()
    return value or None


def required_config_value(name: str) -> str:
    """Read a model config value and raise the public missing-config error when absent."""
    value = config_value(name)
    if value is None:
        raise ValueError(f"Missing model config value: {name}")
    return value


def config_int_value(
    name: str,
    default: int,
) -> int:
    """Read a positive integer model setting with the provided default."""
    value = config_value(name)
    if value is None:
        return default
    if not value.isdigit():
        raise ValueError(f"Model config value must be a positive integer: {name}")
    return int(value)
