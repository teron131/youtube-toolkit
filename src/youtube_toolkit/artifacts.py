"""Preserve retrieval documents in the caller's working directory, with unique timestamped filenames and optional request context."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


def artifact_root() -> Path:
    """Resolve local artifact storage without creating directories or consulting global settings."""
    return Path.cwd() / ".cache" / "youtube-toolkit"


def write_artifact(
    namespace: str,
    identifier: str,
    request: dict[str, Any] | None,
    payload: dict[str, Any],
) -> Path:
    """Save each retrieval independently, preserving prior copies even when filenames collide.

    A missing request saves the payload itself as a complete document, without a request or fetch-time wrapper.
    """
    fetched_at = datetime.now(UTC)
    document = (
        {"request": request, "fetched_at": fetched_at.isoformat(), "payload": payload}
        if request is not None
        else payload
    )
    content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    directory = artifact_root() / namespace
    directory.mkdir(parents=True, exist_ok=True)
    stems = _artifact_stems(identifier, fetched_at)
    while True:
        path = directory / f"{next(stems)}.json"
        try:
            output = path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError:
            continue
        try:
            with output:
                output.write(content)
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path


def create_artifact_directory(namespace: str, identifier: str) -> Path:
    """Reserve a separate output directory so repeated visual extractions retain their prior collages."""
    directory = artifact_root() / namespace
    directory.mkdir(parents=True, exist_ok=True)
    stems = _artifact_stems(identifier, datetime.now(UTC))
    while True:
        path = directory / next(stems)
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path


def _artifact_stems(identifier: str, fetched_at: datetime) -> Iterator[str]:
    """Keep identifiers readable and path-safe, with numbered suffixes for same-time collisions."""
    label = re.sub(r"[^\w.@-]+", "_", identifier).strip(" ._")[:48] or "request"
    stem = f"{fetched_at.strftime('%Y-%m-%dT%H%M%S.%fZ')}-{label}"
    yield stem
    counter = 2
    while True:
        yield f"{stem}-{counter}"
        counter += 1
