"""Speech-to-text response schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import normalize_number, normalize_text


@dataclass(frozen=True, slots=True)
class SpeechTranscription:
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    word_count: int | None = None

    @classmethod
    def from_xai_stt(cls, response: dict[str, Any]) -> SpeechTranscription:
        text = normalize_text(response.get("text"))
        if text is None:
            raise ValueError("xAI STT response did not include transcript text")

        words = response.get("words")
        return cls(
            text=text,
            language=normalize_text(response.get("language")),
            duration_seconds=normalize_number(response.get("duration")),
            word_count=len(words) if isinstance(words, list) else None,
        )

    @classmethod
    def from_elevenlabs_stt(cls, response: dict[str, Any]) -> SpeechTranscription:
        text = normalize_text(response.get("text")) or extract_multichannel_text(response)
        if text is None:
            raise ValueError("ElevenLabs STT response did not include transcript text")

        words = response.get("words")
        return cls(
            text=text,
            language=normalize_text(response.get("language_code")),
            duration_seconds=extract_max_word_end(words),
            word_count=count_transcribed_words(words),
        )


def extract_multichannel_text(response: dict[str, Any]) -> str | None:
    transcripts = response.get("transcripts")
    if not isinstance(transcripts, list):
        return None
    transcript_texts: list[str] = []
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            continue
        transcript_text = normalize_text(transcript.get("text"))
        if transcript_text:
            transcript_texts.append(transcript_text)
    return " ".join(transcript_texts) or None


def count_transcribed_words(words: Any) -> int | None:
    if not isinstance(words, list):
        return None
    return sum(1 for word in words if isinstance(word, dict) and word.get("type", "word") == "word")


def extract_max_word_end(words: Any) -> float | None:
    if not isinstance(words, list):
        return None
    word_ends: list[float] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        end = normalize_number(word.get("end"))
        if end is not None:
            word_ends.append(end)
    return max(word_ends) if word_ends else None
