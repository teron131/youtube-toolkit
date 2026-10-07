"""Speech-to-text transcription owns provider requests, response parsing, and caller-facing result shapes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal

from ..schemas import normalize_text
from . import elevenlabs, xai

DEFAULT_STT_LANGUAGE = "en"
DEFAULT_STT_PROVIDER = "xai"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"
SpeechToTextProvider = Literal["xai", "elevenlabs"]


@dataclass(frozen=True, slots=True)
class SpeechTranscription:
    """Provider-neutral speech-to-text result used by transcript documents and plain-text STT callers."""

    text: str
    language: str | None = None

    @classmethod
    def from_xai_stt(cls, response: dict[str, Any]) -> SpeechTranscription:
        """Translate xAI STT JSON into the provider-neutral transcription result."""
        text = normalize_text(response.get("text"))
        if text is None:
            raise ValueError("xAI STT response did not include transcript text")

        return cls(
            text=text,
            language=normalize_text(response.get("language")),
        )

    @classmethod
    def from_elevenlabs_stt(cls, response: dict[str, Any]) -> SpeechTranscription:
        """Translate ElevenLabs STT JSON, including multichannel output, into one transcription result."""
        text = normalize_text(response.get("text"))
        if text is None:
            transcripts = response.get("transcripts")
            if isinstance(transcripts, list):
                transcript_texts: list[str] = []
                for transcript in transcripts:
                    if not isinstance(transcript, dict):
                        continue
                    transcript_text = normalize_text(transcript.get("text"))
                    if transcript_text is not None:
                        transcript_texts.append(transcript_text)
                text = " ".join(transcript_texts) or None
        if text is None:
            raise ValueError("ElevenLabs STT response did not include transcript text")

        return cls(
            text=text,
            language=normalize_text(response.get("language_code")),
        )


@dataclass(frozen=True, slots=True)
class SpeechToTextOptions:
    """Options for selecting an STT provider and request behavior."""

    provider: SpeechToTextProvider = DEFAULT_STT_PROVIDER
    language: str | None = DEFAULT_STT_LANGUAGE
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300


def transcribe_audio(
    audio_path: Path,
    *,
    options: SpeechToTextOptions | None = None,
) -> str:
    """Return only transcript text for callers that do not need provider metadata."""
    return transcribe_audio_details(audio_path, options=options).text


def transcribe_audio_details(
    audio_path: Path,
    *,
    options: SpeechToTextOptions | None = None,
) -> SpeechTranscription:
    """Preserve normalized provider metadata for transcript-document workflows."""
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    options = options or SpeechToTextOptions()
    if options.provider == "xai":
        response = xai.xai_stt(
            audio_path,
            api_key=_provider_api_key(
                explicit_api_key=options.api_key,
                env_var="XAI_API_KEY",
                provider_name="xAI",
            ),
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
        return SpeechTranscription.from_xai_stt(response)
    if options.provider == "elevenlabs":
        response = elevenlabs.elevenlabs_stt(
            audio_path,
            api_key=_provider_api_key(
                explicit_api_key=options.api_key,
                env_var="ELEVENLABS_API_KEY",
                provider_name="ElevenLabs",
            ),
            model=options.model or DEFAULT_ELEVENLABS_STT_MODEL,
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
        return SpeechTranscription.from_elevenlabs_stt(response)
    raise ValueError(f"Unsupported speech-to-text provider: {options.provider}")


def _provider_api_key(
    *,
    explicit_api_key: str | None,
    env_var: str,
    provider_name: str,
) -> str:
    api_key = explicit_api_key or os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"{env_var} is required for {provider_name} speech-to-text")
    return api_key
