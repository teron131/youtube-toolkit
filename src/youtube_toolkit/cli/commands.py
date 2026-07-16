"""CLI command handlers, setup persistence, and runtime health probes."""

from __future__ import annotations

import argparse
import builtins
from dataclasses import asdict, dataclass
import getpass
from pathlib import Path
import sys
from typing import Any, Literal, NotRequired, TypedDict, cast

from pydantic import SecretStr

from ..codex_config import CodexSetupResult, configure_codex_permissions
from ..identity import require_video_id
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
from ..schemas import ChannelRemovalResult, EmbeddingIndexResult, WatchlistSyncResult
from ..search import ChannelTarget, SearchRequest, search as run_search_request
from ..search.request import SearchSource
from ..storage import (
    DATABASE_FILENAME,
    database_schema_health,
    embed_text,
    get_connection,
    index_video_embedding,
    list_video_ids_for_embedding_index,
    resolve_storage_dir,
    upsert_video,
)
from ..storage.connection import DatabaseSchemaHealthResult
from ..storage.videos import delete_video_record, edit_video_content, get_video_record
from ..storage.viewer import serve_database_viewer
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


class DoctorResult(TypedDict):
    """Structured health payload rendered by the doctor command."""

    yt_dlp_version: str
    database: str
    database_schema: DatabaseSchemaHealthResult
    watchlist_schedule: WatchlistScheduleHealthResult
    model_apis: ModelApisHealthResult


class ModelSetupResult(TypedDict):
    """Structured setup payload after config persistence and optional verification."""

    config: str
    model_apis: ModelApisHealthResult
    settings: ModelSettingsResult
    verification: ModelApisVerificationResult


class ModelSettingsResult(TypedDict):
    """Persisted model behavior switches exposed by setup and doctor output."""

    auto_embeddings: bool


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


class IndexCommandResult(TypedDict):
    """Embedding index command result split into indexed, skipped, and failed videos."""

    videos: int
    indexed: list[EmbeddingIndexResult]
    skipped: list[EmbeddingIndexResult]
    failed: list[EmbeddingIndexResult]


DEFAULT_YOUTUBE_SEARCH_LIMIT = 10
DEFAULT_CHANNEL_SEARCH_LIMIT = 30
DATABASE_REFRESH_METADATA_FIELDS = (
    "channel_id",
    "channel_handle",
    "title",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
)


class DatabaseCommands:
    """Handlers for inspecting and administering the active database."""

    @staticmethod
    def view(args: argparse.Namespace) -> None:
        """Serve the active database until the user stops the viewer."""
        serve_database_viewer(
            resolve_storage_dir() / DATABASE_FILENAME,
            port=args.port,
            open_browser=not args.no_open,
        )

    @staticmethod
    def get(args: argparse.Namespace) -> dict[str, Any]:
        """Return one complete locally stored video record."""
        video_id = require_video_id(args.video_ref)
        with get_connection() as connection:
            return get_video_record(connection, video_id)

    @staticmethod
    def refresh(args: argparse.Namespace) -> dict[str, Any]:
        """Refresh provider-owned metadata through yt-dlp and report the stored facts."""
        requested_video_id = require_video_id(args.video_ref)
        metadata = YtDlpClient().metadata(args.video_ref)
        fetched_video_id = require_video_id(str(metadata.get("id") or ""))
        if fetched_video_id != requested_video_id:
            raise ValueError(f"yt-dlp returned video {fetched_video_id} for requested video {requested_video_id}")
        with get_connection() as connection:
            stored_video = upsert_video(connection, metadata)
        if stored_video is None:
            raise RuntimeError(f"Could not store refreshed metadata for video {requested_video_id}")
        return {
            "video_id": requested_video_id,
            "metadata": {field: stored_video[field] for field in DATABASE_REFRESH_METADATA_FIELDS},
            "metadata_fetched_at": stored_video["metadata_fetched_at"],
        }

    @staticmethod
    def edit(args: argparse.Namespace) -> dict[str, Any]:
        """Translate explicit agent-owned content fields into one validated edit."""
        video_id = require_video_id(args.video_ref)
        values = {}
        if args.summary is not None:
            values["summary"] = args.summary
        if args.transcript_file is not None:
            values["transcript"] = read_text_input(args.transcript_file)
        clear_fields = [field.replace("-", "_") for field in args.clear]
        with get_connection() as connection:
            return edit_video_content(
                connection,
                video_id,
                values,
                clear_fields=clear_fields,
                dry_run=args.dry_run,
            )

    @staticmethod
    def delete(args: argparse.Namespace) -> dict[str, Any]:
        """Require explicit confirmation before cascading one stored video deletion."""
        video_id = require_video_id(args.video_ref)
        if not args.dry_run and not args.yes:
            raise ValueError("Video deletion requires --yes; use --dry-run to inspect it first")
        with get_connection() as connection:
            return delete_video_record(
                connection,
                video_id,
                dry_run=args.dry_run,
            )


class VideoCommands:
    """Handlers for retrieving video metadata and transcripts."""

    @staticmethod
    def transcript(args: argparse.Namespace) -> dict[str, Any]:
        return asdict(fetch_transcript(args.video_ref))

    @staticmethod
    def metadata(args: argparse.Namespace) -> dict[str, Any]:
        return YtDlpClient().metadata(args.video_ref)


def read_text_input(path: str) -> str:
    """Read UTF-8 command text from a file or standard input while preserving the original failure cause."""
    try:
        if path == "-":
            return sys.stdin.read()
        return Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Could not read text input {path}: {error}") from error


class ChannelCommands:
    """Handlers for direct channel listing commands."""

    @staticmethod
    def videos(args: argparse.Namespace) -> dict[str, Any]:
        return fetch_channel_videos(args.channel_ref, options=channel_options(args))


class SearchCommands:
    """Handlers for the public mixed-source search command."""

    @staticmethod
    def search(args: argparse.Namespace) -> list[dict[str, Any]]:
        """Translate parsed CLI arguments into the public search request contract."""
        raw_sources = tuple(dict.fromkeys(args.source or ["youtube"]))
        limit = args.limit
        if limit is None:
            limit = DEFAULT_CHANNEL_SEARCH_LIMIT if raw_sources == ("channel",) else DEFAULT_YOUTUBE_SEARCH_LIMIT
        sources = cast(tuple[SearchSource, ...], raw_sources)
        return run_search_request(
            SearchRequest(
                query=args.query,
                extra_queries=tuple(getattr(args, "extra_queries", [])),
                sources=sources,
                mode=args.mode,
                return_type=args.return_type,
                channels=tuple(ChannelTarget.from_ref(channel) for channel in args.channels),
                limit=limit,
                sort=args.sort,
                min_views=args.min_views,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                uploaded_after=getattr(args, "uploaded_after", None),
                uploaded_within_days=getattr(args, "uploaded_within_days", None),
                include_channel=tuple(args.include_channel),
                exclude_channel=tuple(args.exclude_channel),
                include_title=tuple(args.include_title),
                exclude_title=tuple(args.exclude_title),
            )
        )


class WatchlistCommands:
    """Handlers for watchlist membership, sync, recency, and scheduler commands."""

    @staticmethod
    def add(args: argparse.Namespace) -> builtins.list[str]:
        return add_watchlist_channels(args.channel_refs)

    @staticmethod
    def list(args: argparse.Namespace) -> builtins.list[str]:
        del args
        return list_watchlist_channels()

    @staticmethod
    def remove(args: argparse.Namespace) -> builtins.list[ChannelRemovalResult]:
        """Remove watchlist membership without deleting catalog data."""
        return remove_watchlist_channels(args.channel_refs)

    @staticmethod
    def sync(args: argparse.Namespace) -> WatchlistSyncResult:
        return sync_watchlist(options=channel_options(args))

    @staticmethod
    def scheduled_sync(args: argparse.Namespace) -> ScheduledSyncResult:
        """The hidden scheduler entrypoint uses scheduled default arguments."""
        return scheduled_sync_watchlist(options=channel_options(args))

    @staticmethod
    def recent(args: argparse.Namespace) -> builtins.list[dict[str, Any]]:
        """Recent watchlist output is storage-only and does not mutate catalog state."""
        return list_recent_watchlist_videos(
            channel_refs=args.channel_refs,
            days=args.days,
        )

    @staticmethod
    def install_schedule(args: argparse.Namespace) -> WatchlistScheduleInstallResult:
        """Install the LaunchAgent that runs hidden watchlist scheduled-sync."""
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
        """Inspect the LaunchAgent state for the configured watchlist schedule label."""
        return watchlist_schedule_status(label=args.label)

    @staticmethod
    def uninstall_schedule(args: argparse.Namespace) -> WatchlistScheduleUninstallResult:
        """Unload and remove the LaunchAgent for the configured watchlist schedule label."""
        return uninstall_watchlist_schedule(label=args.label)


class BlacklistCommands:
    """Handlers for search blacklist membership commands."""

    @staticmethod
    def add(args: argparse.Namespace) -> list[str]:
        return add_blacklist_channels(args.channel_refs)

    @staticmethod
    def remove(args: argparse.Namespace) -> list[ChannelRemovalResult]:
        """Remove blacklist membership without deleting catalog data."""
        return remove_blacklist_channels(args.channel_refs)


class IndexCommands:
    """Handlers for embedding index maintenance commands."""

    @staticmethod
    def videos(args: argparse.Namespace) -> IndexCommandResult:
        """Index explicit videos or all eligible stored videos when no refs are provided."""
        video_ids = [require_video_id(video_ref) for video_ref in args.video_refs]
        indexed: list[EmbeddingIndexResult] = []
        skipped: list[EmbeddingIndexResult] = []
        failed: list[EmbeddingIndexResult] = []

        if not video_ids:
            with get_connection() as connection:
                video_ids = list_video_ids_for_embedding_index(
                    connection,
                    require_transcript=not args.include_without_transcript,
                )

        for video_id in video_ids:
            with get_connection() as connection:
                result = index_one_video(connection, video_id)
            if result["status"] == "indexed":
                indexed.append(result)
            elif result["status"] == "skipped":
                skipped.append(result)
            else:
                failed.append(result)

        return {
            "videos": len(video_ids),
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
        }


class SetupCommands:
    """Handlers for persistent model API and host integration setup."""

    @staticmethod
    def models(args: argparse.Namespace) -> ModelSetupResult:
        """Model setup either opens the terminal menu or saves supplied flags directly."""
        if sys.stdin.isatty() and setup_args_missing(args):
            return run_model_setup_menu(args)

        draft = prompt_model_setup(args)
        return save_model_setup(
            draft,
            verify=args.verify if args.verify is not None else False,
        )

    @staticmethod
    def codex(args: argparse.Namespace) -> CodexSetupResult:
        """Preview or apply the active storage writable root in Codex config."""
        return configure_codex_permissions(apply=args.yes)


def index_one_video(
    connection: Any,
    video_id: str,
) -> EmbeddingIndexResult:
    """Index one stored video and convert storage outcomes into command result rows."""
    try:
        payload = index_video_embedding(connection, video_id)
    except Exception as error:
        return {
            "status": "failed",
            "video_id": video_id,
            "error": str(error),
        }
    if payload is None:
        return {
            "status": "skipped",
            "video_id": video_id,
            "detail": "Video is not stored",
        }
    return {
        "status": "indexed",
        "video_id": payload["video_id"],
        "dimensions": payload["dimensions"],
    }


@dataclass(slots=True)
class ModelSetupDraft:
    """Mutable setup draft shared by direct flags and the interactive menu before persistence."""

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
        """Seed an interactive draft from explicit flags, stored config, and setup defaults."""
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
        """Project the draft into persisted config keys, including boolean normalization."""
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
    """Selectable terminal menu row mapped to an internal setup action."""

    label: str
    action: str


def run_model_setup_menu(args: argparse.Namespace) -> ModelSetupResult:
    """Drive the terminal setup menu until the user saves or quits."""
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
    """Handle one reusable raw-key terminal menu loop."""
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
    """Redraw the full terminal menu in place."""
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
    """Clear the terminal after the setup menu exits."""
    print("\x1b[2J\x1b[H", end="", flush=True)


def setup_menu_items(draft: ModelSetupDraft) -> tuple[MenuItem, ...]:
    """The top-level setup menu mirrors the current draft state."""
    return (
        MenuItem(f"Embedding API  {setup_section_status(draft.embedding_api_key, draft.embedding_model)}", "embedding"),
        MenuItem(f"LLM API        {setup_section_status(draft.llm_api_key and draft.llm_base_url, draft.llm_model)}", "llm"),
        MenuItem(f"Settings       auto embeddings {setup_bool_text(draft.auto_embeddings)}", "settings"),
        MenuItem("Verify         send Hello to configured APIs", "verify"),
        MenuItem("Save and exit", "save"),
        MenuItem("Quit without saving", "quit"),
    )


def read_setup_key() -> str:
    """Raw terminal input is translated into the setup menu's supported navigation keys."""
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
    """Summarize whether a setup section has enough config to be useful."""
    status = "configured" if configured else "missing"
    return f"{status} ({model})" if model else status


def setup_bool_text(value: bool) -> str:
    """Render setup booleans in the menu's compact on/off vocabulary."""
    return "on" if value else "off"


def configure_embedding(draft: ModelSetupDraft) -> None:
    """Prompt for embedding API fields and mutate the setup draft."""
    print("\nEmbedding API")
    draft.embedding_api_key = prompt_secret("API key", draft.embedding_api_key)
    draft.embedding_base_url = prompt_text("Base URL", draft.embedding_base_url)
    draft.embedding_model = prompt_text("Model", draft.embedding_model)
    draft.embedding_dimensions = prompt_text("Dimensions", draft.embedding_dimensions)


def configure_llm(draft: ModelSetupDraft) -> None:
    """Prompt for LLM API fields and mutate the setup draft."""
    print("\nLLM API")
    draft.llm_api_key = prompt_secret("API key", draft.llm_api_key)
    draft.llm_base_url = prompt_text("Base URL", draft.llm_base_url)
    draft.llm_model = prompt_text("Model", draft.llm_model)


def configure_model_settings(draft: ModelSetupDraft) -> str | None:
    """Toggle non-secret model settings from a nested setup menu."""
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
    """Setup setting rows mirror the current draft state."""
    return (
        MenuItem(f"Auto-index transcript embeddings {setup_bool_text(draft.auto_embeddings)}", "auto_embeddings"),
        MenuItem("Back", "back"),
    )


def prompt_secret(
    label: str,
    existing: str | None,
) -> str | None:
    """Prompt for a secret while preserving the current value on blank input."""
    state = "configured" if existing else "empty"
    return getpass.getpass(f"{label} [{state}; blank keeps current]: ").strip() or existing


def prompt_text(
    label: str,
    existing: str | None,
) -> str | None:
    """Prompt for non-secret setup text while preserving the current value on blank input."""
    if existing:
        return input(f"{label} [{existing}]: ").strip() or existing
    return input(f"{label}: ").strip() or None


def prompt_model_setup(args: argparse.Namespace) -> ModelSetupDraft:
    """Collect missing setup values from prompts when direct flags are incomplete."""
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
    """Persist model setup and attach either live or supplied verification results."""
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
    """The public setup payload combines persisted state with verification results."""
    return {
        "config": config_path or str(app_config_path()),
        "model_apis": model_api_health(),
        "settings": {
            "auto_embeddings": draft.auto_embeddings,
        },
        "verification": verification,
    }


def setup_verification_text(verification: ModelApisVerificationResult) -> str:
    """Render verification results for the interactive setup menu."""
    return "\n".join(
        [
            "",
            "Verification",
            f"  embedding  {verification['embedding']['status']} - {verification['embedding']['detail']}",
            f"  llm        {verification['llm']['status']} - {verification['llm']['detail']}",
        ]
    )


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


def setup_args_missing(args: argparse.Namespace) -> bool:
    """Setup needs prompts when at least one setup flag is absent."""
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


def setup_text_value(
    value: str | int | None,
    key: str,
    default: str | None,
) -> str | None:
    """Resolve a setup value from explicit input, persisted config, then default."""
    if value is not None:
        return str(value)
    return config_value(key) or default


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


def confirm(
    prompt: str,
    *,
    default: bool,
) -> bool:
    """Prompt for a yes/no answer with a default used on blank input."""
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
    """Resolve a secret from explicit setup input, stored config, or an interactive prompt."""
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
    """Resolve non-secret setup text from explicit input, stored config, default, or prompt."""
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
    """Resolve the auto-embedding switch from flags, prompt, or persisted config."""
    if value is not None:
        return value
    if interactive:
        return confirm("Auto-index transcript embeddings after sync?", default=True)
    return embedding_auto_index_enabled()


def channel_options(args: argparse.Namespace) -> ChannelOptions:
    """Translate shared CLI window flags into channel workflow options."""
    return ChannelOptions(last=args.last, days=args.days)
