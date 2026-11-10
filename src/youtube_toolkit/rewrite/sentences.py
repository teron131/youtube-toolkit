"""Caption lines regrouped into sentences, for rewrites where no line's own timing has to survive."""

from __future__ import annotations

from collections.abc import Sequence
import re

from ..schemas import TranscriptSegment

# Captions without punctuation never end a sentence, so this caps how many lines one sentence gathers.
MAX_LINES_PER_SENTENCE = 16
# A Latin end mark, also behind a closing quote or bracket, ends a sentence before whitespace unless a lowercase letter follows, which keeps a `page.close` broken across caption lines whole.
# The Chinese and Japanese marks end one with no space after them.
# YouTube captions write a censored word as `[ __ ]` and drop the end mark after it, so the marker ends a sentence when a capital letter follows.
SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019)\]]))\s+(?=[^\sa-z]|$)"
    r"|(?<=[\u3002\uff01\uff1f])"
    r"|(?<=\[ __ \])\s+(?=[A-Z])"
)


def split_sentences(segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
    """A sentence runs from the start of the line it begins in to the end of the line it ends in, so untimed text yields untimed sentences."""
    sentences: list[TranscriptSegment] = []
    parts: list[str] = []
    start = 0.0
    lines = 0
    for position, segment in enumerate(segments):
        pieces = [
            piece for piece in SENTENCE_BOUNDARY.split(segment.text.strip()) if piece
        ]
        following = (
            segments[position + 1].text.strip() if position + 1 < len(segments) else ""
        )
        for index, piece in enumerate(pieces):
            if not parts:
                start, lines = segment.start_seconds, 1
            elif index == 0:
                lines += 1
            parts.append(piece)
            text = " ".join(parts)
            # A quotation can hold several sentences, so a boundary inside one does not end the sentence.
            boundary = index < len(pieces) - 1 or _ends_sentence(piece, following)
            if (
                boundary and not _inside_quote(text)
            ) or lines == MAX_LINES_PER_SENTENCE:
                sentences.append(TranscriptSegment(start, segment.end_seconds, text))
                parts = []
    if parts:
        sentences.append(
            TranscriptSegment(start, segments[-1].end_seconds, " ".join(parts))
        )
    return sentences


def _ends_sentence(line: str, following: str) -> bool:
    joint = SENTENCE_BOUNDARY.search(f"{line} {following}", len(line))
    return joint is not None and joint.start() == len(line)


def _inside_quote(text: str) -> bool:
    return text.count('"') % 2 == 1 or text.count("\u201c") > text.count("\u201d")
