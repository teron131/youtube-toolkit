"""Caption lines regrouped into sentences at their end marks, for the article and for untimed speech, where no line's own timing has to survive the regrouping."""

from __future__ import annotations

from collections.abc import Sequence
import re

from ..schemas import TranscriptChunk

# Captions without punctuation never end a sentence, so this caps how many lines one sentence gathers; a punctuated 77-minute video had 9 sentences cut mid-way at 8 lines and none at 16.
MAX_LINES_PER_SENTENCE = 16
# A full stop, exclamation mark, or question mark, also behind a closing quote or bracket, ends a sentence before whitespace unless a lowercase letter follows, as in a `page.close` broken across caption lines; the Chinese and Japanese marks end one with no space after them.
# YouTube captions write a censored word as `[ __ ]` and drop the end mark after it, so the marker ends a sentence when a capital letter follows.
SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019)\]]))\s+(?=[^\sa-z]|$)"
    r"|(?<=[\u3002\uff01\uff1f])"
    r"|(?<=\[ __ \])\s+(?=[A-Z])"
)


def split_sentences(chunks: Sequence[TranscriptChunk]) -> list[TranscriptChunk]:
    """A sentence runs from the start of the line it begins in to the end of the line it ends in, so untimed text yields untimed sentences."""
    sentences: list[TranscriptChunk] = []
    parts: list[str] = []
    start = 0.0
    lines = 0
    for position, chunk in enumerate(chunks):
        pieces = [
            piece for piece in SENTENCE_BOUNDARY.split(chunk.text.strip()) if piece
        ]
        following = (
            chunks[position + 1].text.strip() if position + 1 < len(chunks) else ""
        )
        for index, piece in enumerate(pieces):
            if not parts:
                start, lines = chunk.start_seconds, 1
            elif index == 0:
                lines += 1
            parts.append(piece)
            text = " ".join(parts)
            # Every piece but a line's last ends at a sentence boundary, though a quotation can hold several sentences.
            boundary = index < len(pieces) - 1 or _ends_sentence(piece, following)
            if (
                boundary and not _inside_quote(text)
            ) or lines == MAX_LINES_PER_SENTENCE:
                sentences.append(TranscriptChunk(start, chunk.end_seconds, text))
                parts = []
    if parts:
        sentences.append(
            TranscriptChunk(start, chunks[-1].end_seconds, " ".join(parts))
        )
    return sentences


def _ends_sentence(line: str, following: str) -> bool:
    joint = SENTENCE_BOUNDARY.search(f"{line} {following}", len(line))
    return joint is not None and joint.start() == len(line)


def _inside_quote(text: str) -> bool:
    return text.count('"') % 2 == 1 or text.count("\u201c") > text.count("\u201d")
