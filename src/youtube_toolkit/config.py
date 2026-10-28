"""App-specific settings, environment overrides, and defaults shared by retrieval clients, search, and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

from dotenv.parser import parse_stream
from platformdirs import user_config_path

DEFAULT_SEARCH_MIN_VIEWS = 10_000
DEFAULT_SEARCH_MIN_DURATION = "3:00"
DEFAULT_SEARCH_MAX_DURATION = "8:00:00"
DEFAULT_YOUTUBE_SEARCH_LIMIT = 10
DEFAULT_CHANNEL_SEARCH_LIMIT = 30
DEFAULT_LAST_VIDEOS = 10
MAX_CONCURRENT_METADATA_FETCHES = 8
DEFAULT_VISION_MAX_FRAMES = 96
MAX_VISION_FRAMES = 96
DEFAULT_VISION_MIN_SIMILARITY = 0.3
DEFAULT_VISION_MIN_GAP_SECONDS = 10.0

DEFAULT_STT_PROVIDER = "xai"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"
DEFAULT_GEMINI_STT_MODEL = "gemini-3.5-transcribe"
DEFAULT_OPENAI_STT_MODEL = "gpt-transcribe"
# Vercel AI Gateway serves many providers' models under one key; this is its id for xAI's speech-to-text.
DEFAULT_VERCEL_STT_MODEL = "spacexai/grok-stt"
DEFAULT_XAI_STT_MODEL = "grok-voice-transcribe-2.0"
# A Hugging Face ASR model ID; STT_LOCAL_MODEL can name any other model that runs in the transformers automatic-speech-recognition pipeline.
DEFAULT_LOCAL_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

DEFAULT_EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONALITY = 768
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
    """Resolve EMBEDDING_* settings from environment overrides and saved app configuration, requiring only the API key."""
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
    configured = config_int_value(
        "EMBEDDING_CONCURRENCY", DEFAULT_EMBEDDING_CONCURRENCY
    )
    return min(job_count, max(1, min(configured, MAX_EMBEDDING_CONCURRENCY)))


def stt_provider_name() -> str:
    """Name the cloud speech-to-text provider chosen with STT_PROVIDER."""
    return config_value("STT_PROVIDER") or DEFAULT_STT_PROVIDER


def vision_embedding_key() -> str | None:
    """Reuse the generic embedding key only when its model and endpoint identify Google Gemini."""
    key = config_value("GEMINI_API_KEY")
    if key:
        return key
    model = config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    base_url = config_value("EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL
    if (
        model == "gemini-embedding-2"
        and urlparse(base_url).hostname == "generativelanguage.googleapis.com"
    ):
        return config_value("EMBEDDING_API_KEY")
    return None


def local_stt_model() -> str:
    """Return the Hugging Face ASR model ID chosen with STT_LOCAL_MODEL for local speech-to-text."""
    return config_value("STT_LOCAL_MODEL") or DEFAULT_LOCAL_STT_MODEL


def config_value(name: str) -> str | None:
    """Prefer a nonblank environment override, then the app's saved setting, without modifying the process environment."""
    value = os.environ.get(name, "").strip()
    return value or read_config().get(name, "").strip() or None


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


def config_path() -> Path:
    """Locate per-user settings outside the installed package and research directories so reinstalls preserve keys."""
    return user_config_path("youtube-toolkit", appauthor=False, roaming=True) / ".env"


def read_config() -> dict[str, str]:
    """Read dotenv settings literally without exporting them, expanding variables, or accepting a partially parsed file."""
    path = config_path()
    settings = {}
    try:
        with path.open(encoding="utf-8") as stream:
            for binding in parse_stream(stream):
                if binding.error or (
                    binding.key is not None
                    and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", binding.key)
                ):
                    raise ValueError(
                        f"Invalid YouTube Toolkit configuration: {path} (line {binding.original.line})"
                    )
                if binding.key is not None and binding.value is not None:
                    settings[binding.key] = binding.value
    except FileNotFoundError:
        return {}
    except UnicodeError as error:
        raise ValueError(f"Invalid YouTube Toolkit configuration: {path}") from error
    return settings


def save_config(settings: dict[str, str]) -> Path:
    """Replace the app's config atomically with a user-only file; never write shell configuration or export settings."""
    path = config_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".config-",
        delete_on_close=False,
    ) as stream:
        for name, value in settings.items():
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            stream.write(f"{name}='{escaped}'\n")
        # Windows requires the temporary file to be closed before replacement.
        stream.close()
        temporary = Path(stream.name)
        temporary.chmod(0o600)
        temporary.replace(path)
    return path
