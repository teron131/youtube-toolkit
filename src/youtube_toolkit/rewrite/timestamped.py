"""Caption lines corrected or translated under their own IDs in parallel batches, with neighbouring lines as context, so every reply keeps the timestamps of the line it came from and no word moves to another line's time; untimed speech goes through here sentence by sentence."""

from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher

from ..llm import ChatModel
from ..prompts import CORRECT_LINES, TRANSLATE_LINES
from ..schemas import TranscriptSegment
from .requests import (
    ATTEMPTS,
    RewriteError,
    fixed_batches,
    replies,
    request,
    run_batches,
)

# Ten sentences per request gave the model enough of the speech while an hour of video fit in about two rounds of workers, and the test videos run 1.4 to 2.5 caption lines per sentence, so 20 lines carry about as much speech.
LINES_PER_BATCH = 20
# A caption line is often a fragment, and its neighbours show the sentence it belongs to.
CONTEXT_LINES = 4
# On sentences of a 32-minute English and a 19-minute Japanese video, corrected text kept a similarity of at least 0.94 and 0.74 to its own sentence in 95 of 100 and fell below this only where the model had moved a neighbour's text or guessed at a misheard name, while text set against the next sentence passed it in 14 of 616.
MIN_LINE_SIMILARITY = 0.5


def rewrite_lines(
    llm: ChatModel, header: str, lines: list[TranscriptSegment], language: str | None
) -> list[TranscriptSegment]:
    texts = [line.text for line in lines]
    system = TRANSLATE_LINES.format(language=language) if language else CORRECT_LINES

    def rewrite(start: int, end: int) -> list[TranscriptSegment]:
        user = request(
            header, texts, (start, end), texts[start:end], context=CONTEXT_LINES
        )
        best: dict[int, str] = {}
        for _ in range(ATTEMPTS):
            best = max(best, replies(llm.invoke(system, user), end - start), key=len)
            if len(best) == end - start:
                break
        if language and not best:
            raise RewriteError("The model returned no translation")
        return _aligned(lines[start:end], best, translated=bool(language))

    batches = run_batches(
        fixed_batches(len(texts), LINES_PER_BATCH),
        rewrite,
        task="Rewriting",
        unit="lines",
        total=len(texts),
    )
    return [segment for batch in batches for segment in batch]


def _aligned(
    lines: list[TranscriptSegment], texts: dict[int, str], *, translated: bool
) -> list[TranscriptSegment]:
    """Each line takes the reply under its ID; a missing ID means the model merged that line into a neighbour, so the entry before it, or after it for a batch's first line, spans both.

    A cleaned-up copy keeps the original wherever a reply is too unlike its line to be a correction of it, such as a neighbour's text moved onto it, and keeps a missing line apart when the neighbour's reply reads as that neighbour alone, which means the line was dropped rather than merged.
    """
    groups: list[tuple[int, int, int]] = []
    pending = None
    for number in range(1, len(lines) + 1):
        if number in texts:
            groups.append((pending or number, number, number))
            pending = None
        elif groups:
            first, _, owner = groups[-1]
            groups[-1] = (first, number, owner)
        else:
            pending = pending or number
    if not groups:
        return lines
    segments: list[TranscriptSegment] = []
    for first, last, owner in groups:
        span = lines[first - 1 : last]
        text = texts[owner]
        merged = TranscriptSegment(span[0].start_seconds, span[-1].end_seconds, text)
        if translated:
            segments.append(merged)
            continue
        own = _similarity(lines[owner - 1].text, text)
        if len(span) > 1 and _similarity(
            " ".join(line.text for line in span), text
        ) >= max(own, MIN_LINE_SIMILARITY):
            segments.append(merged)
            continue
        segments.extend(
            replace(line, text=text)
            if number == owner and own >= MIN_LINE_SIMILARITY
            else line
            for number, line in enumerate(span, first)
        )
    return segments


def _similarity(source: str, rewritten: str) -> float:
    return SequenceMatcher(None, source, rewritten, autojunk=False).ratio()
