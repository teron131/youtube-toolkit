"""Speech-to-text transcription owns provider requests, response parsing, and caller-facing result shapes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Literal
import urllib.error
import urllib.request
from uuid import uuid4

from ..execution import retry_transient
from ..schemas import normalize_text

DEFAULT_STT_LANGUAGE = "en"
DEFAULT_STT_PROVIDER = "xai"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"
STT_REQUEST_ATTEMPTS = 3
STT_RETRY_BACKOFF_SECONDS = 0.5
AUDIO_CONTENT_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/x-m4a",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}

SpeechToTextProvider = Literal["xai", "elevenlabs"]


@dataclass(frozen=True, slots=True)
class SpeechTranscription:
    """Provider-neutral speech-to-text result used by storage and plain-text STT callers."""

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
    """Preserve normalized provider metadata for storage and transcript-document workflows."""
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    options = options or SpeechToTextOptions()
    response = _request_provider_response(
        audio_path,
        options=options,
    )
    if options.provider == "xai":
        return SpeechTranscription.from_xai_stt(response)
    if options.provider == "elevenlabs":
        return SpeechTranscription.from_elevenlabs_stt(response)
    raise ValueError(f"Unsupported speech-to-text provider: {options.provider}")


def _request_provider_response(
    audio_path: Path,
    *,
    options: SpeechToTextOptions,
) -> dict[str, Any]:
    if options.provider == "xai":
        from . import xai

        return xai.xai_stt(
            audio_path,
            api_key=_provider_api_key(
                explicit_api_key=options.api_key,
                env_var="XAI_API_KEY",
                provider_name="xAI",
            ),
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
    if options.provider == "elevenlabs":
        from . import elevenlabs

        return elevenlabs.elevenlabs_stt(
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


def run_json_request(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    provider_name: str,
) -> dict[str, Any]:
    """Run one provider request with bounded retries for transient and malformed responses."""

    def request_payload() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS API endpoint.
            raw_response = response.read()
        response_data = json.loads(raw_response)
        if not isinstance(response_data, dict):
            raise json.JSONDecodeError("STT response was not a JSON object", "", 0)
        return response_data

    try:
        return retry_transient(
            request_payload,
            attempts=STT_REQUEST_ATTEMPTS,
            backoff_seconds=STT_RETRY_BACKOFF_SECONDS,
            retryable_errors=(urllib.error.URLError, json.JSONDecodeError),
            sleep=time.sleep,
        )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{provider_name} STT request failed ({error.code}): {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{provider_name} STT request failed: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{provider_name} STT response was not valid JSON") from error


def encode_multipart_form(
    *,
    fields: dict[str, str | None],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """Build the multipart payload shared by providers without leaking boundary details to adapters."""
    boundary = f"----youtube-toolkit-{uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                f"{value}\r\n".encode(),
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode(),
            f"Content-Type: {AUDIO_CONTENT_TYPES.get(file_path.suffix.casefold(), 'application/octet-stream')}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
