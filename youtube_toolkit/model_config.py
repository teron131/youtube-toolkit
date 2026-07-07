"""Shared model API configuration from environment and app-local settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlparse

APP_STATE_DIRNAME = "youtube-toolkit"
CONFIG_FILENAME = "config.env"
EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONALITY = 768
DEFAULT_EMBEDDING_AUTO_INDEX = True
DEFAULT_EMBEDDING_CONCURRENCY = 16
MAX_EMBEDDING_CONCURRENCY = 50
MODEL_CONFIG_KEYS = (
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_AUTO_INDEX",
    "EMBEDDING_CONCURRENCY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
)
LLM_REQUIRED_CONFIG = (
    ("LLM_API_KEY", "API key"),
    ("LLM_BASE_URL", "base URL"),
    ("LLM_MODEL", "model"),
)


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
    model: str | None = None,
    base_url: str | None = None,
    output_dimensionality: int | None = None,
) -> EmbeddingConfig:
    api_key = required_config_value("EMBEDDING_API_KEY")
    return EmbeddingConfig(
        model=model or config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        api_key=api_key,
        base_url=base_url or config_value("EMBEDDING_BASE_URL") or EMBEDDING_BASE_URL,
        output_dimensionality=output_dimensionality or config_int_value("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONALITY),
    )


def embedding_auto_index_enabled() -> bool:
    return config_bool_value("EMBEDDING_AUTO_INDEX", DEFAULT_EMBEDDING_AUTO_INDEX)


def embedding_concurrency(job_count: int) -> int:
    configured_concurrency = config_int_value("EMBEDDING_CONCURRENCY", DEFAULT_EMBEDDING_CONCURRENCY)
    return min(
        job_count,
        max(
            1,
            min(configured_concurrency, MAX_EMBEDDING_CONCURRENCY),
        ),
    )


def llm_config_from_env() -> ModelConfig:
    return ModelConfig(
        model=required_config_value("LLM_MODEL"),
        api_key=required_config_value("LLM_API_KEY"),
        base_url=required_config_value("LLM_BASE_URL"),
    )


def config_value(name: str) -> str | None:
    if name not in MODEL_CONFIG_KEYS:
        raise ValueError(f"Unsupported model config key: {name}")
    return env_or_app_config_value(name)


def env_or_app_config_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return app_config_values().get(name)


def required_config_value(name: str) -> str:
    value = config_value(name)
    if value is None:
        raise ValueError(f"Missing model config value: {name}")
    return value


def config_int_value(
    name: str,
    default: int,
) -> int:
    value = config_value(name)
    if value is None:
        return default
    if not value.isdigit():
        raise ValueError(f"Model config value must be a positive integer: {name}")
    return int(value)


def config_bool_value(
    name: str,
    default: bool,
) -> bool:
    value = config_value(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Model config value must be a boolean: {name}")


def app_config_path() -> Path:
    return Path.home() / "Library" / "Application Support" / APP_STATE_DIRNAME / CONFIG_FILENAME


def app_config_values() -> dict[str, str]:
    return dotenv_values(app_config_path())


def write_model_config(updates: Mapping[str, str | int | None]) -> Path:
    values = app_config_values()
    for key, raw_value in updates.items():
        if key not in MODEL_CONFIG_KEYS:
            raise ValueError(f"Unsupported model config key: {key}")
        value = str(raw_value).strip() if raw_value is not None else ""
        if "\n" in value or "\r" in value:
            raise ValueError(f"Model config value cannot contain newlines: {key}")
        if value:
            values[key] = value
        else:
            values.pop(key, None)

    file_path = app_config_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{key}={values[key]}\n" for key in MODEL_CONFIG_KEYS if values.get(key))
    file_path.write_text(text, encoding="utf-8")
    file_path.chmod(0o600)
    return file_path


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
