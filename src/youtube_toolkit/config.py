"""App settings: the per-user settings file and nonblank environment overrides, plus the embedding, decision, and speech-to-text model IDs, setting defaults, and the similarity cutoffs measured for each embedding route.

The vision command's limits also live here because the vision package needs its optional extra to import, while the CLI parser needs them on every run.
"""

from __future__ import annotations

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
# A Hugging Face model ID; EMBEDDING_LOCAL_MODEL can name any other model that loads in sentence-transformers, and vision also needs it to embed images.
DEFAULT_LOCAL_EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-omni-small-retrieval"
# TypeSafe's System One endpoint on Vercel AI Gateway; DECISION_URL can name any host that accepts the same request, such as OpenRouter's https://openrouter.ai/api/alpha/decisions.
DEFAULT_DECISION_URL = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
# DECISION_MODEL can name another model that reads images, though vision's score cutoff was measured on this one.
DEFAULT_DECISION_MODEL = "perplexity/pplx-decider-flash-v1.2"
# An API key setting holding a name of this shape, such as GEMINI_API_KEY, uses that variable's value; real keys never look like this.
KEY_REFERENCE = re.compile(r"[A-Z_][A-Z0-9_]*_KEY")


@dataclass(frozen=True, slots=True)
class EmbeddingCutoffs:
    """Cosine cutoffs only hold on the similarity scale of the model they were measured on."""

    vision_match: float
    vision_duplicate: float
    search_title: float
    search_result: float


# Measured on gemini-embedding-2, the default for both API routes; other API models reuse them.
# A 0.95 duplicate cutoff already merged distinct slides that share a template too often.
API_EMBEDDING_CUTOFFS = EmbeddingCutoffs(
    vision_match=0.3, vision_duplicate=0.96, search_title=0.3, search_result=0.5
)
# Measured on DEFAULT_LOCAL_EMBEDDING_MODEL over four graded test videos and set conservatively, as the most lenient value any one video needed, because dropping a useful frame or passage costs more than keeping an extra one.
# The match and search cutoffs pass the share of pairs the API cutoff passes on each video; the search pairs are keyword-to-paragraph transcript pairs, since no labeled search queries exist.
# The duplicate cutoff is the lowest at which no video lost more distinct frames than with Gemini's 0.96; real-world footage set it, and it leaves more repeats than Gemini.
# Other local models reuse all four with a warning.
LOCAL_EMBEDDING_CUTOFFS = EmbeddingCutoffs(
    vision_match=0.059, vision_duplicate=0.96, search_title=-0.049, search_result=0.045
)


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


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        if urlparse(self.url).scheme not in {"http", "https"}:
            raise ValueError("DECISION_URL must use http or https")

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc


def decision_config() -> DecisionConfig | None:
    """None when DECISION_API_KEY is unset, which leaves vision to the embedding routes."""
    api_key = config_value("DECISION_API_KEY")
    if api_key is None:
        return None
    return DecisionConfig(
        url=config_value("DECISION_URL") or DEFAULT_DECISION_URL,
        api_key=api_key,
        model=config_value("DECISION_MODEL") or DEFAULT_DECISION_MODEL,
    )


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


def local_embedding_model() -> str:
    """Return the Hugging Face model ID chosen with EMBEDDING_LOCAL_MODEL for local embeddings."""
    return config_value("EMBEDDING_LOCAL_MODEL") or DEFAULT_LOCAL_EMBEDDING_MODEL


def local_embedding_cutoffs(model: str) -> tuple[EmbeddingCutoffs, str | None]:
    """Return the local cutoffs and, for a model they were not measured on, a warning to show with its results."""
    if model == DEFAULT_LOCAL_EMBEDDING_MODEL:
        return LOCAL_EMBEDDING_CUTOFFS, None
    return (
        LOCAL_EMBEDDING_CUTOFFS,
        f"Similarity cutoffs were measured on {DEFAULT_LOCAL_EMBEDDING_MODEL}, not {model}; matches may be too strict or too loose.",
    )


def local_stt_model() -> str:
    """Return the Hugging Face ASR model ID chosen with STT_LOCAL_MODEL for local speech-to-text."""
    return config_value("STT_LOCAL_MODEL") or DEFAULT_LOCAL_STT_MODEL


def config_value(name: str) -> str | None:
    """Prefer a nonblank environment override, then the app's saved setting, without modifying the process environment.

    An API key setting that names another variable resolves to that variable's value, and raises when it is unset rather than leaving the feature unconfigured.
    """
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


def _setting(name: str) -> str | None:
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
