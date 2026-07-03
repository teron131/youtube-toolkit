"""CLI command handlers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import getpass
import sys
from typing import Any, Literal, NotRequired, TypedDict

from ..model_config import (
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    LLM_REQUIRED_CONFIG,
    ModelConfig,
    app_config_path,
    config_value,
    embedding_auto_index_enabled,
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
    settings: ModelSettingsResult
    verification: ModelApisVerificationResult


class ModelSettingsResult(TypedDict):
    auto_embeddings: bool


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
        if setup_interactive_terminal() and setup_args_missing(args):
            return run_model_setup_menu(args)

        draft = prompt_model_setup(args)
        return save_model_setup(
            draft,
            verify=args.verify if args.verify is not None else False,
        )


@dataclass(slots=True)
class ModelSetupDraft:
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str | None
    embedding_dimensions: str | None
    auto_embeddings: bool
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str | None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ModelSetupDraft:
        return cls(
            embedding_api_key=args.embedding_api_key if args.embedding_api_key is not None else config_value("EMBEDDING_API_KEY"),
            embedding_base_url=setup_text_value(args.embedding_base_url, "EMBEDDING_BASE_URL", EMBEDDING_BASE_URL),
            embedding_model=setup_text_value(args.embedding_model, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dimensions=setup_text_value(args.embedding_dimensions, "EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONALITY)),
            auto_embeddings=args.auto_embeddings if args.auto_embeddings is not None else embedding_auto_index_enabled(),
            llm_api_key=args.llm_api_key if args.llm_api_key is not None else config_value("LLM_API_KEY"),
            llm_base_url=setup_text_value(args.llm_base_url, "LLM_BASE_URL", None),
            llm_model=setup_text_value(args.llm_model, "LLM_MODEL", None),
        )

    def updates(self) -> dict[str, str | None]:
        return {
            "EMBEDDING_API_KEY": self.embedding_api_key,
            "EMBEDDING_BASE_URL": self.embedding_base_url,
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_DIMENSIONS": self.embedding_dimensions,
            "EMBEDDING_AUTO_INDEX": "true" if self.auto_embeddings else "false",
            "LLM_API_KEY": self.llm_api_key,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_MODEL": self.llm_model,
        }


@dataclass(frozen=True, slots=True)
class MenuItem:
    label: str
    action: str


def run_model_setup_menu(args: argparse.Namespace) -> ModelSetupResult:
    draft = ModelSetupDraft.from_args(args)
    verification = skipped_model_verification()
    selected_index = 0
    message: str | None = None
    while True:
        action, selected_index, message = select_menu_item(
            title="Setup",
            help_text="Use Up/Down to move, Enter to select, q to quit.",
            items=setup_menu_items(draft),
            selected_index=selected_index,
            message=message,
        )
        if action == "embedding":
            configure_embedding(draft)
            message = None
        elif action == "llm":
            configure_llm(draft)
            message = None
        elif action == "settings":
            message = configure_model_settings(draft)
        elif action == "verify":
            save_model_setup(draft, verify=False)
            verification = verify_model_apis()
            message = setup_verification_text(verification)
        elif action == "save":
            clear_setup_menu()
            return save_model_setup(draft, verify=args.verify if args.verify is not None else False, verification=verification)
        elif action == "quit":
            clear_setup_menu()
            return model_setup_result(draft, skipped_model_verification())


def select_menu_item(
    *,
    title: str,
    help_text: str,
    items: tuple[MenuItem, ...],
    selected_index: int,
    message: str | None,
) -> tuple[str, int, str | None]:
    while True:
        render_menu(
            title=title,
            help_text=help_text,
            items=items,
            selected_index=selected_index,
            message=message,
        )
        key = read_setup_key()
        if key in {"up", "k"}:
            selected_index = (selected_index - 1) % len(items)
            message = None
            continue
        if key in {"down", "j"}:
            selected_index = (selected_index + 1) % len(items)
            message = None
            continue
        if key == "q":
            return "quit", selected_index, None
        if key == "enter":
            return items[selected_index].action, selected_index, message
        message = help_text


def render_menu(
    *,
    title: str,
    help_text: str,
    items: tuple[MenuItem, ...],
    selected_index: int,
    message: str | None,
) -> None:
    lines = [
        f"\x1b[2J\x1b[H{title}",
        help_text,
        "",
    ]
    for index, item in enumerate(items):
        marker = ">" if index == selected_index else " "
        lines.append(f"{marker} {item.label}")
    if message:
        lines.extend(("", message))
    print("\n".join(lines), end="", flush=True)


def clear_setup_menu() -> None:
    print("\x1b[2J\x1b[H", end="", flush=True)


def setup_menu_items(draft: ModelSetupDraft) -> tuple[MenuItem, ...]:
    return (
        MenuItem(f"Embedding API  {setup_section_status(draft.embedding_api_key, draft.embedding_model)}", "embedding"),
        MenuItem(f"LLM API        {setup_section_status(draft.llm_api_key and draft.llm_base_url, draft.llm_model)}", "llm"),
        MenuItem(f"Settings       auto embeddings {setup_bool_text(draft.auto_embeddings)}", "settings"),
        MenuItem("Verify         send Hello to configured APIs", "verify"),
        MenuItem("Save and exit", "save"),
        MenuItem("Quit without saving", "quit"),
    )


def read_setup_key() -> str:
    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        key = sys.stdin.read(1)
        if key == "\x1b":
            key += sys.stdin.read(2)
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_settings)

    if key == "\x1b[A":
        return "up"
    if key == "\x1b[B":
        return "down"
    if key in {"\r", "\n"}:
        return "enter"
    if key in {"j", "k", "q"}:
        return key
    if key == "\x03":
        raise KeyboardInterrupt
    return "unknown"


def setup_section_status(
    configured: object,
    model: str | None,
) -> str:
    status = "configured" if configured else "missing"
    return f"{status} ({model})" if model else status


def setup_bool_text(value: bool) -> str:
    return "on" if value else "off"


def configure_embedding(draft: ModelSetupDraft) -> None:
    print("\nEmbedding API")
    draft.embedding_api_key = prompt_secret("API key", draft.embedding_api_key)
    draft.embedding_base_url = prompt_text("Base URL", draft.embedding_base_url)
    draft.embedding_model = prompt_text("Model", draft.embedding_model)
    draft.embedding_dimensions = prompt_text("Dimensions", draft.embedding_dimensions)


def configure_llm(draft: ModelSetupDraft) -> None:
    print("\nLLM API")
    draft.llm_api_key = prompt_secret("API key", draft.llm_api_key)
    draft.llm_base_url = prompt_text("Base URL", draft.llm_base_url)
    draft.llm_model = prompt_text("Model", draft.llm_model)


def configure_model_settings(draft: ModelSetupDraft) -> str | None:
    selected_index = 0
    message: str | None = None
    while True:
        action, selected_index, message = select_menu_item(
            title="Settings",
            help_text="Use Up/Down to move, Enter to select, q to go back.",
            items=settings_menu_items(draft),
            selected_index=selected_index,
            message=message,
        )
        if action in {"back", "quit"}:
            return None
        if action == "auto_embeddings":
            draft.auto_embeddings = not draft.auto_embeddings
            message = f"Auto embeddings {setup_bool_text(draft.auto_embeddings)}."


def settings_menu_items(draft: ModelSetupDraft) -> tuple[MenuItem, ...]:
    return (
        MenuItem(f"Auto-index transcript embeddings {setup_bool_text(draft.auto_embeddings)}", "auto_embeddings"),
        MenuItem("Back", "back"),
    )


def prompt_secret(
    label: str,
    existing: str | None,
) -> str | None:
    state = "configured" if existing else "empty"
    return getpass.getpass(f"{label} [{state}; blank keeps current]: ").strip() or existing


def prompt_text(
    label: str,
    existing: str | None,
) -> str | None:
    if existing:
        return input(f"{label} [{existing}]: ").strip() or existing
    return input(f"{label}: ").strip() or None


def prompt_model_setup(args: argparse.Namespace) -> ModelSetupDraft:
    interactive = setup_args_missing(args)
    return ModelSetupDraft(
        embedding_api_key=configured_secret(args.embedding_api_key, "Embedding API key", "EMBEDDING_API_KEY"),
        embedding_base_url=configured_text(args.embedding_base_url, "Embedding base URL", "EMBEDDING_BASE_URL", EMBEDDING_BASE_URL),
        embedding_model=configured_text(args.embedding_model, "Embedding model", "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dimensions=configured_text(
            args.embedding_dimensions,
            "Embedding dimensions",
            "EMBEDDING_DIMENSIONS",
            str(DEFAULT_EMBEDDING_DIMENSIONALITY),
        ),
        auto_embeddings=configured_auto_embeddings(args.auto_embeddings, interactive=interactive),
        llm_api_key=configured_secret(args.llm_api_key, "LLM API key", "LLM_API_KEY"),
        llm_base_url=configured_text(args.llm_base_url, "LLM base URL", "LLM_BASE_URL", None),
        llm_model=configured_text(args.llm_model, "LLM model", "LLM_MODEL", None),
    )


def save_model_setup(
    draft: ModelSetupDraft,
    *,
    verify: bool,
    verification: ModelApisVerificationResult | None = None,
) -> ModelSetupResult:
    config_path = write_model_config(draft.updates())
    return model_setup_result(
        draft,
        verify_model_apis() if verify else (verification or skipped_model_verification()),
        config_path=str(config_path),
    )


def model_setup_result(
    draft: ModelSetupDraft,
    verification: ModelApisVerificationResult,
    *,
    config_path: str | None = None,
) -> ModelSetupResult:
    return {
        "config": config_path or str(app_config_path()),
        "model_apis": model_api_health(),
        "settings": {
            "auto_embeddings": draft.auto_embeddings,
        },
        "verification": verification,
    }


def setup_verification_text(verification: ModelApisVerificationResult) -> str:
    return "\n".join(
        [
            "",
            "Verification",
            f"  embedding  {verification['embedding']['status']} - {verification['embedding']['detail']}",
            f"  llm        {verification['llm']['status']} - {verification['llm']['detail']}",
        ]
    )


def verify_model_apis() -> ModelApisVerificationResult:
    return {
        "embedding": verify_embedding_api(),
        "llm": verify_llm_api(),
    }


def skipped_model_verification() -> ModelApisVerificationResult:
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


def setup_args_missing(args: argparse.Namespace) -> bool:
    return any(
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


def setup_interactive_terminal() -> bool:
    return sys.stdin.isatty()


def setup_text_value(
    value: str | int | None,
    key: str,
    default: str | None,
) -> str | None:
    if value is not None:
        return str(value)
    return config_value(key) or default


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


def configured_auto_embeddings(
    value: bool | None,
    *,
    interactive: bool,
) -> bool:
    if value is not None:
        return value
    if interactive:
        return confirm("Auto-index transcript embeddings after sync?", default=True)
    return embedding_auto_index_enabled()


def channel_options(args: argparse.Namespace) -> ChannelOptions:
    return ChannelOptions(last=args.last, days=args.days)
