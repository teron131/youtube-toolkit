"""CLI readiness checks, optional dependency guards, and local environment reporting.

Checks inspect configuration and installed packages only: they send no API requests, load no models, create no files, and report no secret values.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib.util import find_spec
import shutil

from .. import config
from ..artifacts import artifact_root
from ..embeddings import local_embedding_downloaded, local_embedding_installed
from ..stt import (
    cloud_provider,
    cloud_stt_configured,
    local_model_downloaded,
    local_stt_installed,
)

VISION_MODULES = ("numpy", "PIL", "cv2", "yake")


def run_doctor(args: argparse.Namespace) -> str:
    del args
    import yt_dlp.version

    return "\n".join(
        [
            "YouTube Toolkit",
            f"  yt-dlp     {yt_dlp.version.__version__}",
            f"  config     {config.config_path()}",
            f"  artifacts  {artifact_root()}",
            f"  vision     {_checked(vision_health)}",
            f"  ffmpeg     {'available' if shutil.which('ffmpeg') else 'not installed'}",
            f"  embedding  {_checked(embedding_health)}",
            f"  decision   {_checked(decision_health)}",
            f"  stt        {_checked(stt_health)}",
        ]
    )


def embedding_health() -> str:
    if config.config_value("EMBEDDING_API_KEY") is None:
        if not local_embedding_installed():
            return "Set EMBEDDING_API_KEY with `youtube setup --embeddings`, or install youtube-toolkit[embeddings] for local models"
        model = config.local_embedding_model()
        if not local_embedding_downloaded(model):
            return f"Local {model} downloads on first use"
        return f"Configured (local {model})"
    try:
        embedding = config.embedding_config_from_env()
    except ValueError as error:
        return str(error)
    return f"Configured ({embedding.model})"


def vision_health() -> str:
    """Name the route `youtube vision` would rank with, in its order of preference."""
    if not vision_available():
        return "not installed"
    if decision := config.decision_config():
        return f"available; ranks with {decision.model} at {decision.host}"
    if config.vision_embedding_key():
        return "available; ranks with Gemini embeddings"
    if local_embedding_installed():
        return f"available; ranks with local {config.local_embedding_model()}"
    return "available; picks by time slots until `youtube setup --decisions` or `--embeddings` saves a key, or youtube-toolkit[embeddings] is installed"


def decision_health() -> str:
    if decision := config.decision_config():
        return f"Configured ({decision.model} at {decision.host})"
    return "Set DECISION_API_KEY with `youtube setup --decisions`"


def stt_health() -> str:
    provider = cloud_provider()
    if cloud_stt_configured():
        return f"Configured ({provider.label} API); `youtube stt-setup --cloud-test` sends a billed test"
    if not local_stt_installed():
        return f"Set {provider.api_key_env} with `youtube setup --stt`, or install youtube-toolkit[stt] and ffmpeg for local models"
    model = config.local_stt_model()
    if not local_model_downloaded(model):
        return f"Local {model} downloads on first use, or now with `youtube stt-setup`"
    return f"Configured (local {model}); `youtube stt-setup` checks that it runs"


def _checked(check: Callable[[], str]) -> str:
    """A key setting that names an unset variable reports on its own line instead of stopping the whole report."""
    try:
        return check()
    except ValueError as error:
        return str(error)


def vision_available() -> bool:
    try:
        return all(find_spec(module) is not None for module in VISION_MODULES)
    except (ImportError, ValueError):
        return False


def require_vision() -> None:
    if vision_available():
        return
    raise RuntimeError(
        "vision requires the vision capability. Install youtube-toolkit[vision]."
    )
