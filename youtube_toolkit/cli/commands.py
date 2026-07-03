"""CLI command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
from typing import Any, Literal, NotRequired, TypedDict

from ..model_config import (
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    LLM_REQUIRED_CONFIG,
    ModelConfig,
    config_value,
    embedding_config_from_env,
    llm_config_from_env,
    write_model_config,
)
from ..schemas import ChannelRemovalResult, WatchlistSyncResult
from ..storage import DATABASE_FILENAME, database_schema_health, embed_text_with_gemini, resolve_storage_dir
from ..storage.connection import DatabaseSchemaHealthResult
from ..watchlist_schedule import (
    DEFAULT_WATCHLIST_SCHEDULE_LABEL,
    ScheduledSyncResult,
    WatchlistScheduleInstallResult,
    WatchlistScheduleOptions,
    WatchlistScheduleStatusResult,
    WatchlistScheduleUninstallResult,
    install_watchlist_schedule,
    scheduled_sync_watchlist,
    uninstall_watchlist_schedule,
    watchlist_schedule_status,
)
from ..youtube import YtDlpClient
from ..youtube.caption import fetch_transcript
from ..youtube.channels import (
    ChannelOptions,
    add_blacklist_channels,
    add_watchlist_channels,
    fetch_channel_videos,
    list_recent_watchlist_videos,
    list_watchlist_channels,
    remove_blacklist_channels,
    remove_watchlist_channels,
    sync_watchlist,
)
from ..youtube.search import SearchOptions, search_videos


class DoctorResult(TypedDict):
    yt_dlp_version: str
    database: str
    database_schema: DatabaseSchemaHealthResult
    watchlist_schedule: WatchlistScheduleHealthResult
    model_apis: ModelApisHealthResult


class ModelSetupResult(TypedDict):
    config: str
    model_apis: ModelApisHealthResult
    verification: ModelApisVerificationResult


class ModelApisHealthResult(TypedDict):
    embedding: ModelApiHealthResult
    llm: ModelApiHealthResult


class ModelApiHealthResult(TypedDict):
    ok: bool
    model: str
    status: Literal["configured", "missing", "invalid"]
    detail: str


class ModelApisVerificationResult(TypedDict):
    embedding: ModelApiVerificationResult
    llm: ModelApiVerificationResult


class ModelApiVerificationResult(TypedDict):
    ok: bool
    status: Literal["skipped", "passed", "failed"]
    detail: str


class WatchlistScheduleHealthResult(TypedDict):
    label: str
    ok: bool
    status: Literal["unsupported", "not_installed", "not_loaded", "running", "idle", "failed"]
    detail: str
    plist_path: NotRequired[str]


class VideoCommands:
    @staticmethod
    def transcript(args: argparse.Namespace) -> dict[str, Any]:
        return asdict(fetch_transcript(args.video_ref))

    @staticmethod
    def metadata(args: argparse.Namespace) -> dict[str, Any]:
        return YtDlpClient().metadata(args.video_ref)


class ChannelCommands:
    @staticmethod
    def videos(args: argparse.Namespace) -> dict[str, Any]:
        return fetch_channel_videos(args.channel_ref, options=channel_options(args))


class WatchlistCommands:
    @staticmethod
    def add(args: argparse.Namespace) -> list[str]:
        return add_watchlist_channels(args.channel_refs)

    @staticmethod
    def list(args: argparse.Namespace) -> list[str]:
        del args
        return list_watchlist_channels()

    @staticmethod
    def remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
        return remove_watchlist_channels(args.channel_refs)

    @staticmethod
    def sync(args: argparse.Namespace) -> WatchlistSyncResult:
        return sync_watchlist(options=channel_options(args))

    @staticmethod
    def scheduled_sync(args: argparse.Namespace) -> ScheduledSyncResult:
        return scheduled_sync_watchlist(options=channel_options(args))

    @staticmethod
    def recent(args: argparse.Namespace) -> list[dict[str, Any]]:
        return list_recent_watchlist_videos(
            channel_refs=args.channel_refs,
            days=args.days,
        )

    @staticmethod
    def install_schedule(args: argparse.Namespace) -> WatchlistScheduleInstallResult:
        return install_watchlist_schedule(
            options=WatchlistScheduleOptions(
                minutes=args.minutes,
                last=args.last,
                days=args.days,
                label=args.label,
                load=not args.no_load,
            )
        )

    @staticmethod
    def schedule_status(args: argparse.Namespace) -> WatchlistScheduleStatusResult:
        return watchlist_schedule_status(label=args.label)

    @staticmethod
    def uninstall_schedule(args: argparse.Namespace) -> WatchlistScheduleUninstallResult:
        return uninstall_watchlist_schedule(label=args.label)


class BlacklistCommands:
    @staticmethod
    def add(args: argparse.Namespace) -> list[str]:
        return add_blacklist_channels(args.channel_refs)

    @staticmethod
    def remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
        return remove_blacklist_channels(args.channel_refs)


class SetupCommands:
    @staticmethod
    def models(args: argparse.Namespace) -> ModelSetupResult:
        interactive = any(
            getattr(args, name) is None
            for name in (
                "embedding_api_key",
                "embedding_base_url",
                "embedding_model",
                "embedding_dimensions",
                "llm_api_key",
                "llm_base_url",
                "llm_model",
            )
        )
        updates = {
            "EMBEDDING_API_KEY": configured_secret(args.embedding_api_key, "Embedding API key", "EMBEDDING_API_KEY"),
            "EMBEDDING_BASE_URL": configured_text(args.embedding_base_url, "Embedding base URL", "EMBEDDING_BASE_URL", EMBEDDING_BASE_URL),
            "EMBEDDING_MODEL": configured_text(args.embedding_model, "Embedding model", "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            "EMBEDDING_DIMENSIONS": configured_text(
                args.embedding_dimensions,
                "Embedding dimensions",
                "EMBEDDING_DIMENSIONS",
                str(DEFAULT_EMBEDDING_DIMENSIONALITY),
            ),
            "LLM_API_KEY": configured_secret(args.llm_api_key, "LLM API key", "LLM_API_KEY"),
            "LLM_BASE_URL": configured_text(args.llm_base_url, "LLM base URL", "LLM_BASE_URL", None),
            "LLM_MODEL": configured_text(args.llm_model, "LLM model", "LLM_MODEL", None),
        }
        config_path = write_model_config(updates)
        should_verify = args.verify
        if should_verify is None and interactive:
            should_verify = confirm("Verify now by sending 'Hello' to the model APIs?", default=False)
        if should_verify:
            verification = {
                "embedding": verify_embedding_api(),
                "llm": verify_llm_api(),
            }
        else:
            verification = {
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
        return {
            "config": str(config_path),
            "model_apis": model_api_health(),
            "verification": verification,
        }


def run_search(args: argparse.Namespace) -> list[dict[str, Any]]:
    return search_videos(
        args.query,
        options=SearchOptions(
            limit=args.limit,
            sort=args.sort,
            min_views=args.min_views,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            include_channel=tuple(args.include_channel),
            exclude_channel=tuple(args.exclude_channel),
            include_title=tuple(args.include_title),
            exclude_title=tuple(args.exclude_title),
        ),
    )


def run_doctor(args: argparse.Namespace) -> DoctorResult:
    del args
    import yt_dlp

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
    return {
        "embedding": embedding_api_health(),
        "llm": llm_api_health(),
    }


def embedding_api_health() -> ModelApiHealthResult:
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


def verify_embedding_api() -> ModelApiVerificationResult:
    try:
        embedding = embed_text_with_gemini("Hello")
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
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    response = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    ).invoke([HumanMessage(content="Hello")])
    return str(getattr(response, "content", "")).strip()


def confirm(
    prompt: str,
    *,
    default: bool,
) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def configured_secret(
    value: str | None,
    label: str,
    key: str,
) -> str | None:
    if value is not None:
        return value
    existing = config_value(key)
    prompt = f"{label} [{'configured' if existing else 'empty'}; leave blank to keep]: "
    return getpass.getpass(prompt).strip() or existing


def configured_text(
    value: str | int | None,
    label: str,
    key: str,
    default: str | None,
) -> str | None:
    if value is not None:
        return str(value)
    existing = config_value(key)
    fallback = existing or default
    if fallback:
        prompt = f"{label} [{fallback}]: "
        return input(prompt).strip() or fallback
    return input(f"{label}: ").strip() or None


def channel_options(args: argparse.Namespace) -> ChannelOptions:
    return ChannelOptions(last=args.last, days=args.days)
