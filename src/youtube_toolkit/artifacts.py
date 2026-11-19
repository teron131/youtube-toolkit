"""Preserve retrieval documents in the caller's working directory, with unique timestamped filenames and optional request context."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


def artifact_root() -> Path:
    return Path.cwd() / ".cache" / "youtube-toolkit"


def write_artifact(
    namespace: str,
    identifier: str,
    request: dict[str, Any] | None,
    payload: dict[str, Any],
) -> Path:
    """A None request saves the payload alone, without the request and fetched_at envelope."""
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


def artifact_directories(namespace: str, identifier: str) -> list[Path]:
    """Directories create_artifact_directory made for identifier, newest first; one suffixed after a same-microsecond collision is left out."""
    return sorted(
        (artifact_root() / namespace).glob(f"*Z-{_artifact_label(identifier)}"),
        reverse=True,
    )


def _artifact_stems(identifier: str, fetched_at: datetime) -> Iterator[str]:
    """Yield the UTC fetch time joined to the filename-safe identifier, then that stem suffixed `-2`, `-3`, and so on.

    `SEARCH QUERY?` yields `YYYY-MM-DDTHHMMSS.ffffffZ-SEARCH_QUERY`.
    """
    stem = f"{fetched_at.strftime('%Y-%m-%dT%H%M%S.%fZ')}-{_artifact_label(identifier)}"
    yield stem
    counter = 2
    while True:
        yield f"{stem}-{counter}"
        counter += 1


def _artifact_label(identifier: str) -> str:
    return re.sub(r"[^\w.@-]+", "_", identifier).strip(" ._")[:48] or "request"
