"""Persistent model configuration and interactive CLI setup workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import sys
from typing import TYPE_CHECKING, TypedDict

from ..model_config import (
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    app_config_path,
    config_value,
    embedding_auto_index_enabled,
    write_model_config,
)
from .health import (
    ModelApisHealthResult,
    ModelApisVerificationResult,
    model_api_health,
    skipped_model_verification,
    verify_model_apis,
)

if TYPE_CHECKING:
    from ..codex_config import CodexSetupResult


class ModelSetupResult(TypedDict):
    """Structured setup payload after config persistence and optional verification."""

    config: str
    model_apis: ModelApisHealthResult
    settings: ModelSettingsResult
    verification: ModelApisVerificationResult


class ModelSettingsResult(TypedDict):
    """Persisted model behavior switches exposed by setup output."""

    auto_embeddings: bool


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
        from ..codex_config import configure_codex_permissions

        return configure_codex_permissions(apply=args.yes)


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
