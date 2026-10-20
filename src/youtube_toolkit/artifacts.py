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


def _artifact_stems(identifier: str, fetched_at: datetime) -> Iterator[str]:
    """`@Open AI/videos` fetched at 09:05:07.123456 UTC yields `2026-10-20T090507.123456Z-@Open_AI_videos`, then that stem with `-2`, `-3`, and so on."""
    label = re.sub(r"[^\w.@-]+", "_", identifier).strip(" ._")[:48] or "request"
    stem = f"{fetched_at.strftime('%Y-%m-%dT%H%M%S.%fZ')}-{label}"
    yield stem
    counter = 2
    while True:
        yield f"{stem}-{counter}"
        counter += 1
