"""Turn a cloud provider's JSON reply into a transcription, grouping word timings into caption-like lines."""

from __future__ import annotations

import math
import re
from typing import Any

from ..schemas import SpeechTranscription, TranscriptSegment, normalize_text

# Word timings become caption-like lines, which end at sentence punctuation, before a pause, or once they span LINE_MAX_SECONDS.
LINE_ENDINGS = (".", "?", "!", "\u3002", "\uff1f", "\uff01")
LINE_PAUSE_SECONDS = 1.0
LINE_MAX_SECONDS = 10.0
# Words that meet at a CJK character join without a space, as the language is written.
CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def parse_transcription(
    response: dict[str, Any], *, language_key: str
) -> SpeechTranscription | None:
    """Per-channel transcripts are joined when there is no single text, and stay untimed.

    xAI and ElevenLabs time each word, Vercel AI Gateway times segments when its model does, and OpenAI reports no timing.
    OpenAI reports languages as a list of entries, and the first entry's code is used.
    """
    text = normalize_text(response.get("text"))
    transcripts = response.get("transcripts")
    if text is None and isinstance(transcripts, list):
        channel_texts = [
            channel_text
            for channel in transcripts
            if isinstance(channel, dict)
            and (channel_text := normalize_text(channel.get("text"))) is not None
        ]
        text = " ".join(channel_texts) or None
    if text is None:
        return None
    language = response.get(language_key)
    if isinstance(language, list) and language and isinstance(language[0], dict):
        language = language[0].get("code")
    return SpeechTranscription(
        text=text,
        segments=_word_lines(response.get("words"))
        or _gateway_segments(response.get("segments")),
        language=normalize_text(language),
    )


def _word_lines(words: Any) -> list[TranscriptSegment]:
    """ElevenLabs also lists spacing and audio events, and its timings can be null; only timed words count."""
    if not isinstance(words, list):
        return []
    timed = []
    for word in words:
        if not isinstance(word, dict) or word.get("type", "word") != "word":
            continue
        start, end = _seconds(word.get("start")), _seconds(word.get("end"))
        word_text = normalize_text(word.get("text"))
        if start is not None and end is not None and word_text:
            timed.append((start, end, word_text))
    return _lines(timed)


def _gateway_segments(segments: Any) -> list[TranscriptSegment]:
    if not isinstance(segments, list):
        return []
    return [
        TranscriptSegment(start_seconds=start, end_seconds=end, text=segment_text)
        for segment in segments
        if isinstance(segment, dict)
        and (start := _seconds(segment.get("startSecond"))) is not None
        and (end := _seconds(segment.get("endSecond"))) is not None
        and (segment_text := normalize_text(segment.get("text")))
    ]


def _lines(words: list[tuple[float, float, str]]) -> list[TranscriptSegment]:
    lines: list[TranscriptSegment] = []
    line: list[tuple[float, float, str]] = []
    for index, (start, end, word) in enumerate(words):
        line.append((start, end, word))
        next_start = words[index + 1][0] if index + 1 < len(words) else math.inf
        if (
            word.endswith(LINE_ENDINGS)
            or next_start - end >= LINE_PAUSE_SECONDS
            or end - line[0][0] >= LINE_MAX_SECONDS
        ):
            lines.append(
                TranscriptSegment(
                    start_seconds=line[0][0],
                    end_seconds=end,
                    text=_join_words([text for _, _, text in line]),
                )
            )
            line = []
    return lines


def _join_words(words: list[str]) -> str:
    joined = words[0]
    for word in words[1:]:
        joined += word if CJK.match(joined[-1]) or CJK.match(word[0]) else f" {word}"
    return joined


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None
