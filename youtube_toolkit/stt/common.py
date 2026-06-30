"""Speech-to-text provider dispatch, upload helpers, and parsed provider results."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal
import urllib.error
import urllib.request
from uuid import uuid4

from .schemas import SpeechTranscription

DEFAULT_STT_LANGUAGE = "en"
DEFAULT_STT_PROVIDER = "xai"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"
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
    """Transcribe one local audio file to plain text."""

    options = options or SpeechToTextOptions()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")

    if options.provider == "xai":
        from . import xai

        api_key = options.api_key or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is required for xAI speech-to-text")
        response = xai.request_speech_to_text(
            audio_path,
            api_key=api_key,
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
        return SpeechTranscription.from_xai_stt(response).text
    if options.provider == "elevenlabs":
        from . import elevenlabs

        api_key = options.api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs speech-to-text")
        response = elevenlabs.request_speech_to_text(
            audio_path,
            api_key=api_key,
            model=options.model or DEFAULT_ELEVENLABS_STT_MODEL,
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
        return SpeechTranscription.from_elevenlabs_stt(response).text
    raise ValueError(f"Unsupported speech-to-text provider: {options.provider}")


def transcribe_audio_details(
    audio_path: Path,
    *,
    options: SpeechToTextOptions | None = None,
) -> SpeechTranscription:
    """Transcribe one local audio file and preserve provider metadata."""

    options = options or SpeechToTextOptions()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")

    if options.provider == "xai":
        from . import xai

        api_key = options.api_key or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is required for xAI speech-to-text")
        response = xai.request_speech_to_text(
            audio_path,
            api_key=api_key,
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
        return SpeechTranscription.from_xai_stt(response)
    if options.provider == "elevenlabs":
        from . import elevenlabs

        api_key = options.api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabs speech-to-text")
        response = elevenlabs.request_speech_to_text(
            audio_path,
            api_key=api_key,
            model=options.model or DEFAULT_ELEVENLABS_STT_MODEL,
            language=options.language,
            timeout_seconds=options.timeout_seconds,
        )
        return SpeechTranscription.from_elevenlabs_stt(response)
    raise ValueError(f"Unsupported speech-to-text provider: {options.provider}")


def run_json_request(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    provider_name: str,
) -> dict[str, Any]:
    """Run an STT HTTP request and return its JSON object response."""

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS API endpoint.
            raw_response = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{provider_name} STT request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider_name} STT request failed: {exc.reason}") from exc

    try:
        response_data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{provider_name} STT response was not valid JSON") from exc
    if not isinstance(response_data, dict):
        raise RuntimeError(f"{provider_name} STT response was not a JSON object")
    return response_data


def encode_multipart_form(
    *,
    fields: dict[str, str | None],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    """Encode STT form fields and one audio file as multipart form data."""

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
