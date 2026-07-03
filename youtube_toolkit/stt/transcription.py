"""Speech-to-text transcription result and provider response parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import normalize_text


@dataclass(frozen=True, slots=True)
class SpeechTranscription:
    text: str
    language: str | None = None

    @classmethod
    def from_xai_stt(cls, response: dict[str, Any]) -> SpeechTranscription:
        text = normalize_text(response.get("text"))
        if text is None:
            raise ValueError("xAI STT response did not include transcript text")

        return cls(
            text=text,
            language=normalize_text(response.get("language")),
        )

    @classmethod
    def from_elevenlabs_stt(cls, response: dict[str, Any]) -> SpeechTranscription:
        text = normalize_text(response.get("text")) or extract_multichannel_text(response)
        if text is None:
            raise ValueError("ElevenLabs STT response did not include transcript text")

        return cls(
            text=text,
            language=normalize_text(response.get("language_code")),
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
