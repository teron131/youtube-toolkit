"""What every rewrite request shares: a header with the video's title, description, and repeated names, items numbered with bracketed IDs that replies are matched by, neighbouring text as context, and the loop that runs batches in parallel."""

from __future__ import annotations

from collections.abc import Callable
import re

from ..execution import collect_concurrently
from ..names import repeated_names
from ..prompts import HEADER
from ..schemas import TranscriptDocument

# On Vercel AI Gateway the slowest request sets a round's pace, so more requests in flight shorten long videos without slowing short ones.
MAX_WORKERS = 64
# A reply that is empty, misses an ID, or fails its step's checks is asked for again, up to this many times in all; the SDK already retries transport failures.
ATTEMPTS = 3
# A reply line opens with its item's ID, so a merged or split item shows as a missing ID or a line without one instead of shifting every later line onto the wrong item.
# In dialogue the model merges or splits short lines: plain lines broke the count in 9 of 26 and 12 of 60 first replies for two Japanese videos and 1 of 37 for an English one, and with IDs none of those 123 missed one.
LINE_ID = re.compile(r"\s*\[LINE(\d+)\]\s*(.*)")
MAX_NAMES = 60


class RewriteError(RuntimeError):
    pass


def request_header(document: TranscriptDocument) -> str:
    """Title, description, and the names said more than once, most frequent first, which every request opens with."""
    names = repeated_names(document.text)[:MAX_NAMES]
    metadata = document.metadata
    return HEADER.format(
        title=metadata.title or "",
        description=metadata.description or "",
        names=", ".join(names),
    )


def request(
    header: str,
    lines: list[str],
    bounds: tuple[int, int],
    items: list[str],
    *,
    context: int,
) -> str:
    """The numbered `items` between up to `context` of `lines` before `bounds` and after it."""
    start, end = bounds
    before = "\n".join(lines[max(0, start - context) : start])
    after = "\n".join(lines[end : end + context])
    return f"{header}\n\n<before>\n{before}\n</before>\n\n{numbered(items)}\n\n<after>\n{after}\n</after>"


def numbered(lines: list[str]) -> str:
    block = "\n".join(f"[LINE{number}] {line}" for number, line in enumerate(lines, 1))
    return f'<lines count="{len(lines)}">\n{block}\n</lines>'


def replies(reply: str, count: int) -> dict[int, str]:
    """Reply text by ID from 1 to `count`: a line without an ID continues the one before it, and lines before the first ID or under an ID out of range, such as a label or a context line, are dropped."""
    parts: dict[int, list[str]] = {}
    current = None
    for line in reply.splitlines():
        if match := LINE_ID.fullmatch(line):
            current = int(match[1]) if 1 <= int(match[1]) <= count else None
            line = match[2]
        if current is not None and line.strip():
            parts.setdefault(current, []).append(line.strip())
    return {number: " ".join(texts) for number, texts in parts.items()}


def run_batches[Result](
    batches: list[tuple[int, int]],
    job: Callable[[int, int], Result],
    *,
    task: str,
    unit: str,
    total: int,
) -> list[Result]:
    """Runs every (start, end) batch in parallel and returns their results in batch order; the earliest batch that fails raises, named by its 1-based span."""
    results, failures = collect_concurrently(
        dict(enumerate(batches)), lambda bounds: job(*bounds), max_workers=MAX_WORKERS
    )
    if failures:
        index, error = min(failures.items())
        start, end = batches[index]
        raise RewriteError(
            f"{task} failed for {unit} {start + 1}-{end} of {total}: {error}"
        ) from error
    return [results[index] for index in range(len(batches))]


def fixed_batches(count: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(start + size, count)) for start in range(0, count, size)]
