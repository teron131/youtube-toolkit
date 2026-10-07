"""Disposable JSON retrieval cache scoped to the caller's current working directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

REQUEST_MAX_AGE_SECONDS = 15 * 60


def cache_root() -> Path:
    """Resolve the task-local cache without creating directories or consulting global settings."""
    return Path.cwd() / ".cache" / "youtube-toolkit"


def read_cache(
    namespace: str, key: dict[str, Any], *, max_age: float | None = None
) -> dict[str, Any] | None:
    """Treat absent, expired, or malformed cache entries as misses; filesystem access errors remain visible."""
    path = _entry_path(namespace, key)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict) or entry.get("key") != key:
        return None
    fetched_at = entry.get("fetched_at")
    if not isinstance(fetched_at, int | float) or (
        max_age is not None and time.time() - fetched_at >= max_age
    ):
        return None
    payload = entry.get("payload")
    return payload if isinstance(payload, dict) else None


def write_cache(namespace: str, key: dict[str, Any], payload: dict[str, Any]) -> None:
    """Replace a complete entry atomically so parallel retrievals never expose partially written JSON."""
    path = _entry_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"key": key, "fetched_at": time.time(), "payload": payload},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _entry_path(namespace: str, key: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_root() / namespace / f"{digest}.json"
