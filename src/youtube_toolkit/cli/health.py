"""CLI readiness checks, optional dependency guards, and local environment reporting."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import shutil
from typing import Literal, TypedDict

from ..artifacts import artifact_root
from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    config_value,
    embedding_config_from_env,
    local_stt_model,
)
from ..stt import (
    cloud_provider,
    cloud_stt_configured,
    local_model_downloaded,
    local_stt_installed,
)

VISION_MODULES = ("numpy", "PIL", "cv2", "scenedetect")


class EmbeddingHealthResult(TypedDict):
    """Embedding readiness without exposing secrets or sending API requests."""

    ok: bool
    model: str
    status: Literal["configured", "missing", "invalid"]
    detail: str


class DoctorReport(TypedDict):
    """Runtime, artifact location, and optional capability readiness for the doctor command."""

    yt_dlp_version: str
    artifacts: str
    vision: bool
    ffmpeg: bool
    embedding: EmbeddingHealthResult
    stt: str


def run_doctor(args: argparse.Namespace) -> DoctorReport:
    """Inspect the current working directory and optional retrieval capabilities without creating files."""
    del args
    import yt_dlp.version

    return {
        "yt_dlp_version": yt_dlp.version.__version__,
        "artifacts": str(artifact_root()),
        "vision": vision_available(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "embedding": embedding_api_health(),
        "stt": stt_health(),
    }


def embedding_api_health() -> EmbeddingHealthResult:
    """Validate environment configuration for optional semantic search."""
    model = config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if config_value("EMBEDDING_API_KEY") is None:
        return {
            "ok": False,
            "model": model,
            "status": "missing",
            "detail": "Set EMBEDDING_API_KEY for semantic search",
        }
    try:
        embedding_config_from_env()
    except ValueError as error:
        return {"ok": False, "model": model, "status": "invalid", "detail": str(error)}
    return {
        "ok": True,
        "model": model,
        "status": "configured",
        "detail": f"Configured ({model})",
    }


def stt_health() -> str:
    """Describe which speech-to-text route videos without captions would take, without sending API requests or loading a model."""
    provider = cloud_provider()
    if cloud_stt_configured():
        return f"Configured ({provider.label} API); `youtube stt-setup --cloud-test` sends a billed test"
    if not local_stt_installed():
        return f"Set {provider.api_key_env}, or install youtube-toolkit[stt] and ffmpeg for local models"
    model = local_stt_model()
    if not local_model_downloaded(model):
        return f"Local {model} downloads on first use, or now with `youtube stt-setup`"
    return f"Configured (local {model}); `youtube stt-setup` checks that it runs"


def vision_available() -> bool:
    """Check that every vision dependency can be located without loading the packages."""
    try:
        return all(find_spec(module) is not None for module in VISION_MODULES)
    except (ImportError, ValueError):
        return False


def require_vision() -> None:
    """Stop the vision command with installation guidance when dependencies are missing."""
    if vision_available():
        return
    raise RuntimeError(
        "vision requires the vision capability. Install youtube-toolkit[vision]."
    )
