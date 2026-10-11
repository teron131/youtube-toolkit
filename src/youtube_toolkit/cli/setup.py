"""Interactive app configuration keeps API keys out of shell history and saves only after every prompt succeeds."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from urllib.parse import urlparse
import warnings

from .. import config
from ..config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_CONCURRENCY,
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_STT_MODEL,
    DEFAULT_STT_PROVIDER,
)
from ..stt.cloud import PROVIDERS


def run_setup(args: argparse.Namespace) -> dict[str, object]:
    """Prompt for app-scoped keys and settings without contacting providers or changing shell configuration.

    Enter keeps a saved value, and '-' removes it so the existing default applies.
    Environment overrides are neither copied into the file nor modified.
    Cancellation and validation errors leave the existing configuration intact.
    """
    if not sys.stdin.isatty():
        raise ValueError(
            "Run `youtube setup` in an interactive terminal to enter keys securely"
        )
    settings = config.read_config()
    configure_all = not (args.embeddings or args.stt)
    print(
        f"YouTube Toolkit settings: {config.config_path()}\n"
        "Enter keeps saved values; '-' removes a saved value.\n"
        "Keys are hidden. Environment variables override saved settings.",
        file=sys.stderr,
    )
    try:
        if configure_all or args.embeddings:
            _setup_embeddings(settings)
        if configure_all or args.stt:
            _setup_stt(settings)
    except (EOFError, KeyboardInterrupt) as error:
        raise ValueError("Setup cancelled; no settings were saved") from error
    except getpass.GetPassWarning as error:
        raise ValueError(
            "Hidden key entry is unavailable; no settings were saved"
        ) from error
    path = config.save_config(settings)
    overridden = [name for name in settings if os.environ.get(name, "").strip()]
    if overridden:
        print(
            f"Environment variables currently override: {', '.join(overridden)}",
            file=sys.stderr,
        )
    return {"config_path": str(path), "saved_settings": list(settings)}


def _setup_embeddings(settings: dict[str, str]) -> None:
    """Collect the embedding endpoint and key together, validating saved overrides before writing."""
    _prompt(settings, "EMBEDDING_API_KEY", secret=True)
    _prompt(settings, "EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    _prompt(settings, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    _prompt(settings, "EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONALITY))
    _prompt(settings, "EMBEDDING_CONCURRENCY", str(DEFAULT_EMBEDDING_CONCURRENCY))
    base_url = settings.get("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    if urlparse(base_url).scheme not in {"http", "https"}:
        raise ValueError(
            "EMBEDDING_BASE_URL must use http or https; no settings were saved"
        )
    for name in ("EMBEDDING_DIMENSIONS", "EMBEDDING_CONCURRENCY"):
        value = settings.get(name)
        if value is not None and (
            not value.isascii() or not value.isdecimal() or int(value) <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer; no settings were saved"
            )


def _setup_stt(settings: dict[str, str]) -> None:
    """Use the cloud owner's provider catalog so setup asks for the same key as transcription."""
    print(f"Speech-to-text providers: {', '.join(PROVIDERS)}", file=sys.stderr)
    _prompt(settings, "STT_PROVIDER", DEFAULT_STT_PROVIDER)
    name = settings.get("STT_PROVIDER", DEFAULT_STT_PROVIDER)
    if name not in PROVIDERS:
        raise ValueError(
            f"STT_PROVIDER must be one of {', '.join(PROVIDERS)}; no settings were saved"
        )
    _prompt(settings, PROVIDERS[name].api_key_env, secret=True)
    _prompt(settings, "STT_LOCAL_MODEL", DEFAULT_LOCAL_STT_MODEL)


def _prompt(
    settings: dict[str, str], name: str, default: str = "", *, secret: bool = False
) -> None:
    """Display only a secret's presence; blank input preserves saved state without copying environment values."""
    current = settings.get(name, "")
    label = current or default
    if secret:
        label = "saved" if current else "not saved"
    prompt = f"{name} [{label}]: "
    if secret:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass(prompt, stream=sys.stderr).strip()
    else:
        print(prompt, end="", file=sys.stderr, flush=True)
        value = input().strip()
    if value == "-":
        settings.pop(name, None)
    elif value:
        settings[name] = value
