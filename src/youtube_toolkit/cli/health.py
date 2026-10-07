"""Read-only runtime and environment checks for directory-cached retrieval."""

from __future__ import annotations

import argparse
import shutil
from typing import Literal, TypedDict

from ..cache import cache_root
from ..capabilities import capability_available
from ..model_config import (
    DEFAULT_EMBEDDING_MODEL,
    config_value,
    embedding_config_from_env,
)


class ModelApiHealthResult(TypedDict):
    """Embedding readiness without exposing secrets or sending API requests."""

    ok: bool
    model: str
    status: Literal["configured", "missing", "invalid"]
    detail: str


def run_doctor(args: argparse.Namespace) -> dict[str, object]:
    """Inspect the current working directory and optional retrieval capabilities without creating files."""
    del args
    import yt_dlp.version

    return {
        "yt_dlp_version": yt_dlp.version.__version__,
        "cache": str(cache_root()),
        "vision": capability_available("vision"),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "embedding": embedding_api_health(),
    }


def embedding_api_health() -> ModelApiHealthResult:
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
