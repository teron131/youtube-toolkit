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
    DEFAULT_DECISION_MODEL,
    DEFAULT_DECISION_URL,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_CONCURRENCY,
    DEFAULT_EMBEDDING_DIMENSIONALITY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_LOCAL_STT_MODEL,
    DEFAULT_STT_PROVIDER,
)
from ..stt.cloud import PROVIDERS


def run_setup(args: argparse.Namespace) -> dict[str, object]:
    """Environment overrides are neither copied into the file nor modified."""
    if not sys.stdin.isatty():
        raise ValueError(
            "Run `youtube setup` in an interactive terminal to enter keys securely"
        )
    settings = config.read_config()
    configure_all = not (args.embeddings or args.decisions or args.stt)
    print(
        f"YouTube Toolkit settings: {config.config_path()}\n"
        "Enter keeps saved values; '-' removes a saved value.\n"
        "Keys are hidden. A key prompt also takes a variable name, such as GEMINI_API_KEY, to use that variable's value.\n"
        "Environment variables override saved settings.",
        file=sys.stderr,
    )
    try:
        if configure_all or args.embeddings:
            _setup_embeddings(settings)
        if configure_all or args.decisions:
            _setup_decisions(settings)
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
    print(
        "\nEmbeddings for semantic search and vision: the API when its key is set, otherwise the local model.",
        file=sys.stderr,
    )
    _prompt(settings, "EMBEDDING_API_KEY", secret=True)
    _prompt(settings, "EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    _prompt(settings, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    _prompt(settings, "EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONALITY))
    _prompt(settings, "EMBEDDING_CONCURRENCY", str(DEFAULT_EMBEDDING_CONCURRENCY))
    _prompt(settings, "EMBEDDING_LOCAL_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
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


def _setup_decisions(settings: dict[str, str]) -> None:
    print(
        "\nDecision models rank vision frames ahead of embeddings, through any host that takes TypeSafe's System One requests: Vercel AI Gateway by default, or OpenRouter at https://openrouter.ai/api/alpha/decisions.",
        file=sys.stderr,
    )
    _prompt(settings, "DECISION_API_KEY", secret=True)
    _prompt(settings, "DECISION_URL", DEFAULT_DECISION_URL)
    _prompt(settings, "DECISION_MODEL", DEFAULT_DECISION_MODEL)
    url = settings.get("DECISION_URL", DEFAULT_DECISION_URL)
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("DECISION_URL must use http or https; no settings were saved")


def _setup_stt(settings: dict[str, str]) -> None:
    print(
        f"\nSpeech-to-text for videos without captions: the provider's API when its key is set, otherwise the local model. Providers: {', '.join(PROVIDERS)}.",
        file=sys.stderr,
    )
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
    current = settings.get(name, "")
    label = current or default
    if secret and not config.KEY_REFERENCE.fullmatch(current):
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
