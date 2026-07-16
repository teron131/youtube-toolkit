"""CLI health checks and live model API verification."""

from __future__ import annotations

import argparse
from typing import Literal, NotRequired, TypedDict

from pydantic import SecretStr

from ..model_config import (
    DEFAULT_EMBEDDING_MODEL,
    LLM_REQUIRED_CONFIG,
    ModelConfig,
    config_value,
    embedding_config_from_env,
    llm_config_from_env,
)
from ..storage import DATABASE_FILENAME, database_schema_health, embed_text, resolve_storage_dir
from ..storage.connection import DatabaseSchemaHealthResult
from ..watchlist_schedule import DEFAULT_WATCHLIST_SCHEDULE_LABEL, watchlist_schedule_status


class DoctorResult(TypedDict):
    """Structured health payload rendered by the doctor command."""

    yt_dlp_version: str
    database: str
    database_schema: DatabaseSchemaHealthResult
    watchlist_schedule: WatchlistScheduleHealthResult
    model_apis: ModelApisHealthResult


class ModelApisHealthResult(TypedDict):
    """Health payload for every configured model API role."""

    embedding: ModelApiHealthResult
    llm: ModelApiHealthResult


class ModelApiHealthResult(TypedDict):
    """Configuration readiness for one model API role without exposing secrets."""

    ok: bool
    model: str
    status: Literal["configured", "missing", "invalid"]
    detail: str


class ModelApisVerificationResult(TypedDict):
    """Live verification payload for every configured model API role."""

    embedding: ModelApiVerificationResult
    llm: ModelApiVerificationResult


class ModelApiVerificationResult(TypedDict):
    """Result of one live model API smoke request."""

    ok: bool
    status: Literal["skipped", "passed", "failed"]
    detail: str


class WatchlistScheduleHealthResult(TypedDict):
    """Doctor-facing LaunchAgent status with plist detail when installed."""

    label: str
    ok: bool
    status: Literal["unsupported", "not_installed", "not_loaded", "running", "idle", "failed"]
    detail: str
    plist_path: NotRequired[str]


def run_doctor(args: argparse.Namespace) -> DoctorResult:
    """Collect runtime, storage, schedule, and model health for the doctor command."""
    del args
    import yt_dlp.version

    return {
        "yt_dlp_version": yt_dlp.version.__version__,
        "database": str(resolve_storage_dir() / DATABASE_FILENAME),
        "database_schema": database_schema_health(),
        "watchlist_schedule": watchlist_schedule_health(),
        "model_apis": model_api_health(),
    }


def watchlist_schedule_health(
    *,
    label: str = DEFAULT_WATCHLIST_SCHEDULE_LABEL,
) -> WatchlistScheduleHealthResult:
    """Normalize LaunchAgent status into the doctor command's health vocabulary."""
    try:
        status = watchlist_schedule_status(label=label)
    except RuntimeError as error:
        return {
            "label": label,
            "ok": False,
            "status": "unsupported",
            "detail": str(error),
        }
    return {
        "label": status["label"],
        "plist_path": status["plist_path"],
        "ok": status["ok"],
        "status": status["status"],
        "detail": status["detail"],
    }


def model_api_health() -> ModelApisHealthResult:
    """Collect configuration health for every model API role."""
    return {
        "embedding": embedding_api_health(),
        "llm": llm_api_health(),
    }


def embedding_api_health() -> ModelApiHealthResult:
    """Validate embedding configuration without sending a live embedding request."""
    model = config_value("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if config_value("EMBEDDING_API_KEY") is None:
        return {
            "ok": False,
            "model": model,
            "status": "missing",
            "detail": "Embedding API key is missing",
        }
    try:
        embedding_config_from_env()
    except ValueError as error:
        return {
            "ok": False,
            "model": model,
            "status": "invalid",
            "detail": str(error),
        }
    return {
        "ok": True,
        "model": model,
        "status": "configured",
        "detail": "Embedding API is configured",
    }


def llm_api_health() -> ModelApiHealthResult:
    """Validate LLM configuration without sending a live chat request."""
    model = config_value("LLM_MODEL") or "not configured"
    missing = [label for key, label in LLM_REQUIRED_CONFIG if config_value(key) is None]
    if missing:
        return {
            "ok": False,
            "model": model,
            "status": "missing",
            "detail": f"LLM {', '.join(missing)} missing",
        }
    try:
        llm_config_from_env()
    except ValueError as error:
        return {
            "ok": False,
            "model": model,
            "status": "invalid",
            "detail": str(error),
        }
    return {
        "ok": True,
        "model": model,
        "status": "configured",
        "detail": "LLM API is configured",
    }


def verify_model_apis() -> ModelApisVerificationResult:
    """Live smoke requests cover every model API role."""
    return {
        "embedding": verify_embedding_api(),
        "llm": verify_llm_api(),
    }


def skipped_model_verification() -> ModelApisVerificationResult:
    """Skipped verification keeps the setup payload shape stable without live requests."""
    return {
        "embedding": {
            "ok": False,
            "status": "skipped",
            "detail": "Verification skipped",
        },
        "llm": {
            "ok": False,
            "status": "skipped",
            "detail": "Verification skipped",
        },
    }


def verify_embedding_api() -> ModelApiVerificationResult:
    """One embedding smoke request is collapsed into setup status vocabulary."""
    try:
        embedding = embed_text("Hello")
    except Exception as error:
        return {
            "ok": False,
            "status": "failed",
            "detail": f"Embedding verification failed: {error}",
        }
    return {
        "ok": True,
        "status": "passed",
        "detail": f"Embedding API returned {len(embedding)} dimensions",
    }


def verify_llm_api() -> ModelApiVerificationResult:
    """One LLM smoke request is collapsed into setup status vocabulary."""
    try:
        config = llm_config_from_env()
        text = call_llm_hello(config)
    except Exception as error:
        return {
            "ok": False,
            "status": "failed",
            "detail": f"LLM verification failed: {error}",
        }
    return {
        "ok": True,
        "status": "passed",
        "detail": f"LLM API returned {len(text)} characters",
    }


def call_llm_hello(config: ModelConfig) -> str:
    """Call the configured chat model with the setup smoke-test prompt."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    response = ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
    ).invoke([HumanMessage(content="Hello")])
    return str(getattr(response, "content", "")).strip()
