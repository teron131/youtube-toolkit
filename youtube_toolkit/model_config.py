"""Shared model API configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse

EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONALITY = 768


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model: str
    api_key: str
    base_url: str

    def __post_init__(self) -> None:
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
        return f"{self.base_url}/{path.lstrip('/')}"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig(ModelConfig):
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONALITY

    def __post_init__(self) -> None:
        ModelConfig.__post_init__(self)
        if self.output_dimensionality <= 0:
            raise ValueError("output_dimensionality must be greater than 0")


def embedding_config_from_env(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key_env: str = "GEMINI_API_KEY",
    base_url: str = EMBEDDING_BASE_URL,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONALITY,
) -> EmbeddingConfig:
    api_key = env_value(api_key_env)
    if api_key is None:
        raise ValueError(f"Missing API key: set {api_key_env} in env or .env")
    return EmbeddingConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        output_dimensionality=output_dimensionality,
    )


def env_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    if value:
        return value

    repo_root = find_repo_root(Path.cwd()) or find_repo_root(Path(__file__))
    if repo_root is None:
        return None
    return dotenv_values(repo_root / ".env").get(name)


def find_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def dotenv_values(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        value = raw_value.strip().strip("'\"")
        if key.strip():
            values[key.strip()] = value
    return values
