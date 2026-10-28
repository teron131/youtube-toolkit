"""CLI readiness checks, optional dependency guards, and local environment reporting.

Checks inspect configuration and installed packages only: they send no API requests, load no models, create no files, and report no secret values.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import shutil

from .. import config
from ..artifacts import artifact_root
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
            f"  vision     {'available' if vision_available() else 'not installed'}",
            f"  ffmpeg     {'available' if shutil.which('ffmpeg') else 'not installed'}",
            f"  embedding  {embedding_health()}",
            f"  stt        {stt_health()}",
        ]
    )


def embedding_health() -> str:
    if config.config_value("EMBEDDING_API_KEY") is None:
        return "Set EMBEDDING_API_KEY with `youtube setup --embeddings` for semantic search"
    try:
        embedding = config.embedding_config_from_env()
    except ValueError as error:
        return str(error)
    return f"Configured ({embedding.model})"


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
