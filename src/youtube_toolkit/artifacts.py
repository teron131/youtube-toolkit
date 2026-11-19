"""Everything the toolkit keeps lives under `.youtube-toolkit/` in the caller's working directory, which ignores itself in Git.

- `records/NAMESPACE/` holds each raw fetch as timestamped JSON, kept as history and never read back.
- `videos/TITLE_ID/` holds a video's part files and the results later runs reuse; the title comes first so folders sort by it, and the folder is found by the video ID at its end, so it outlives a change of title.
- `tmp/` holds media needed only while a command runs.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from uuid import uuid4

from .filename_utils import normalize_filename_component

ROOT_NAME = ".youtube-toolkit"
# The title in a video's folder name is cut to this many characters so paths stay short; even in a four-byte script that and the ID stay under Linux's 255-byte name limit.
FOLDER_TITLE_LENGTH = 60
KEY_HASH_LENGTH = 12
# A run killed before its cleanup, as by a timeout's SIGTERM, leaves its folder in tmp; no command runs this long, so older folders are such leftovers.
STALE_TEMPORARY_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class Cached[T]:
    """A result reused from an earlier run or made by this one, with the saved entry that holds it."""

    value: T
    reused: bool
    path: Path


def artifact_root() -> Path:
    return Path.cwd() / ROOT_NAME


def write_record(
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
    directory = _root() / "records" / namespace
    directory.mkdir(parents=True, exist_ok=True)
    stems = _record_stems(identifier, fetched_at)
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


def video_directory(video_id: str, title: str | None) -> Path:
    """`videos/TITLE_ID`, or `videos/ID` for a video without a title, made on first use."""
    videos = _root() / "videos"
    videos.mkdir(exist_ok=True)
    existing = [*videos.glob(f"*_{video_id}"), videos / video_id]
    for directory in existing:
        if directory.is_dir():
            return directory
    label = normalize_filename_component(title or "")[:FOLDER_TITLE_LENGTH].rstrip("_")
    directory = videos / (f"{label}_{video_id}" if label else video_id)
    directory.mkdir(exist_ok=True)
    return directory


def reuse(
    directory: Path,
    label: str,
    key: dict[str, Any],
    *,
    fresh: bool,
    compute: Callable[[], dict[str, Any]],
) -> Cached[dict[str, Any]]:
    """The data saved in directory as `LABEL-HASH.json` for this exact key, unless it is missing or `fresh`, in which case compute's data is saved there and returned."""
    path = directory / f"{normalize_filename_component(label)}-{key_hash(key)}.json"
    if not fresh and path.is_file():
        return Cached(json.loads(path.read_text(encoding="utf-8"))["data"], True, path)
    data = compute()
    save_json(
        path, {"key": key, "saved_at": datetime.now(UTC).isoformat(), "data": data}
    )
    return Cached(data, False, path)


def key_hash(key: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[
        :KEY_HASH_LENGTH
    ]


def save_json(path: Path, data: object) -> None:
    save_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def save_text(path: Path, text: str) -> None:
    """Written under a unique name beside path and then renamed, so a file under its final name is complete even when two runs save it at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{uuid4().hex}.part")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)


@contextmanager
def temporary_directory(prefix: str) -> Generator[Path]:
    """Removed when the block ends, failure included, and leftovers of killed runs go first."""
    directory = _root() / "tmp"
    directory.mkdir(exist_ok=True)
    cutoff = time.time() - STALE_TEMPORARY_SECONDS
    for leftover in directory.iterdir():
        if leftover.stat().st_mtime < cutoff:
            shutil.rmtree(leftover, ignore_errors=True)
    with TemporaryDirectory(prefix=prefix, dir=directory) as path:
        yield Path(path)


def _root() -> Path:
    """Made on first use with a `.gitignore` of `*`, as pytest and Ruff do for their folders, so downloaded media never reaches a commit."""
    root = artifact_root()
    root.mkdir(exist_ok=True)
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    return root


def _record_stems(identifier: str, fetched_at: datetime) -> Iterator[str]:
    """Yield the UTC fetch time joined to the filename-safe identifier, then that stem suffixed `-2`, `-3`, and so on.

    `SEARCH QUERY?` yields `YYYY-MM-DDTHHMMSS.ffffffZ-SEARCH_QUERY`.
    """
    stem = f"{fetched_at.strftime('%Y-%m-%dT%H%M%S.%fZ')}-{_record_label(identifier)}"
    yield stem
    counter = 2
    while True:
        yield f"{stem}-{counter}"
        counter += 1


def _record_label(identifier: str) -> str:
    return re.sub(r"[^\w.@-]+", "_", identifier).strip(" ._")[:48] or "request"
