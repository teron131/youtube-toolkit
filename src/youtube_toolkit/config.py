"""App settings: the per-user settings file, environment overrides, and every feature's defaults.

The vision limits live here because the CLI parser needs them without the vision extra installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

from dotenv.parser import parse_stream
from platformdirs import user_config_path

DEFAULT_VISION_MAX_FRAMES = 96
MAX_VISION_FRAMES = 96
DEFAULT_VISION_MIN_GAP_SECONDS = 10.0

DEFAULT_STT_PROVIDER = "xai"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"
DEFAULT_GEMINI_STT_MODEL = "gemini-3.5-transcribe"
DEFAULT_OPENAI_STT_MODEL = "gpt-transcribe"
# Vercel AI Gateway's ID for xAI's speech-to-text.
DEFAULT_VERCEL_STT_MODEL = "spacexai/grok-stt"
DEFAULT_XAI_STT_MODEL = "grok-voice-transcribe-2.0"
# STT_LOCAL_MODEL can name any model the transformers speech-recognition pipeline runs.
DEFAULT_LOCAL_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

DEFAULT_EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL
DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_EMBEDDING_CONCURRENCY = 16
MAX_EMBEDDING_CONCURRENCY = 50
# EMBEDDING_LOCAL_MODEL can name any sentence-transformers model, though vision needs one that also embeds images.
DEFAULT_LOCAL_EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-omni-small-retrieval"
# The SDK appends /v1/systemone to this root.
DEFAULT_DECISION_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
DEFAULT_DECISION_MODEL = "perplexity/pplx-decider-flash-v1.2"
DEFAULT_LLM_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_LLM_MODEL = "openai/gpt-6-luna"
# Real keys never look like a variable name.
KEY_REFERENCE = re.compile(r"[A-Z_][A-Z0-9_]*_KEY")


@dataclass(frozen=True, slots=True)
class EmbeddingCutoffs:
    """Cosine cutoffs only hold on the similarity scale of the model they were measured on."""

    vision_match: float
    vision_duplicate: float
    search_title: float
    search_result: float


API_EMBEDDING_CUTOFFS = EmbeddingCutoffs(
    vision_match=0.3, vision_duplicate=0.96, search_title=0.3, search_result=0.5
)
LOCAL_EMBEDDING_CUTOFFS = EmbeddingCutoffs(
    vision_match=0.059, vision_duplicate=0.96, search_title=-0.049, search_result=0.045
)


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model: str
    api_key: str
    base_url: str
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS

    def __post_init__(self) -> None:
        model = self.model.strip()
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        if not model:
            raise ValueError("model must not be empty")
        if not api_key:
            raise ValueError("api_key must not be empty")
        check_http_url("base_url", base_url)
        if self.dimensions <= 0:
            raise ValueError("embedding dimensions must be greater than 0")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)


def embedding_config() -> EmbeddingConfig:
    """Raises ValueError when EMBEDDING_API_KEY is unset."""
    api_key = config_value("EMBEDDING_API_KEY")
    if api_key is None:
        raise ValueError("Missing model config value: EMBEDDING_API_KEY")
    return EmbeddingConfig(
        model=config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        api_key=api_key,
        base_url=config_value("EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL,
        dimensions=config_int_value(
            "EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS
        ),
    )


def embedding_concurrency(job_count: int) -> int:
    configured = config_int_value(
        "EMBEDDING_CONCURRENCY", DEFAULT_EMBEDDING_CONCURRENCY
    )
    return min(job_count, max(1, min(configured, MAX_EMBEDDING_CONCURRENCY)))


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    base_url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        check_http_url("DECISION_BASE_URL", self.base_url)

    @property
    def host(self) -> str:
        return urlparse(self.base_url).netloc


def decision_config() -> DecisionConfig | None:
    """None when DECISION_API_KEY is unset."""
    api_key = config_value("DECISION_API_KEY")
    if api_key is None:
        return None
    return DecisionConfig(
        base_url=config_value("DECISION_BASE_URL") or DEFAULT_DECISION_BASE_URL,
        api_key=api_key,
        model=config_value("DECISION_MODEL") or DEFAULT_DECISION_MODEL,
    )


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        check_http_url("LLM_BASE_URL", self.base_url)

    @property
    def host(self) -> str:
        return urlparse(self.base_url).netloc


def llm_config() -> LLMConfig | None:
    """None when LLM_API_KEY is unset."""
    api_key = config_value("LLM_API_KEY")
    if api_key is None:
        return None
    return LLMConfig(
        base_url=config_value("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        api_key=api_key,
        model=config_value("LLM_MODEL") or DEFAULT_LLM_MODEL,
    )


def stt_provider_name() -> str:
    return config_value("STT_PROVIDER") or DEFAULT_STT_PROVIDER


def embedding_key() -> str | None:
    """None unless EMBEDDING_* name Gemini Embedding 2 on Google's API, the only embedding API here that takes images."""
    model = config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    base_url = config_value("EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL
    if (
        model == GEMINI_EMBEDDING_MODEL
        and urlparse(base_url).hostname == "generativelanguage.googleapis.com"
    ):
        return config_value("EMBEDDING_API_KEY")
    return None


def local_embedding_model() -> str:
    return config_value("EMBEDDING_LOCAL_MODEL") or DEFAULT_LOCAL_EMBEDDING_MODEL


def local_embedding_cutoffs(model: str) -> tuple[EmbeddingCutoffs, str | None]:
    """The second value is a warning for a model the cutoffs were not tuned on."""
    if model == DEFAULT_LOCAL_EMBEDDING_MODEL:
        return LOCAL_EMBEDDING_CUTOFFS, None
    return (
        LOCAL_EMBEDDING_CUTOFFS,
        f"Similarity cutoffs were measured on {DEFAULT_LOCAL_EMBEDDING_MODEL}, not {model}; matches may be too strict or too loose.",
    )


def local_stt_model() -> str:
    return config_value("STT_LOCAL_MODEL") or DEFAULT_LOCAL_STT_MODEL


def config_value(name: str) -> str | None:
    """A nonblank environment variable wins over the saved setting; an API key setting that names another variable takes that variable's value, raising ValueError when it is unset."""
    value = _setting(name)
    if (
        value is None
        or not name.endswith("_API_KEY")
        or not KEY_REFERENCE.fullmatch(value)
    ):
        return value
    referenced = _setting(value)
    if referenced is None:
        raise ValueError(f"{name} names {value}, which is not set")
    return referenced


def environment_overrides(names: Iterable[str]) -> list[str]:
    return [name for name in names if _environment(name)]


def _setting(name: str) -> str | None:
    return _environment(name) or read_config().get(name, "").strip() or None


def _environment(name: str) -> str:
    return os.environ.get(name, "").strip()


def config_int_value(
    name: str,
    default: int,
) -> int:
    value = config_value(name)
    return default if value is None else parse_positive_int(name, value)


def check_http_url(name: str, url: str) -> None:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"{name} must use http or https")


def parse_positive_int(name: str, value: str) -> int:
    """Only ASCII digits count, since int() also accepts signs, spaces, and digits from other scripts."""
    if not (value.isascii() and value.isdecimal()) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def config_path() -> Path:
    return user_config_path("youtube-toolkit", appauthor=False, roaming=True) / ".env"


def read_config() -> dict[str, str]:
    """Reads values literally, without expanding variables, and raises ValueError on any malformed line."""
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
    """Replaces the file atomically, readable only by the user."""
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
