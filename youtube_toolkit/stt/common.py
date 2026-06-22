"""Shared speech-to-text workflow and HTTP helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from youtube_toolkit.convert_zh import convert_zh

from .schemas import SpeechTranscription

DEFAULT_STT_LANGUAGE = "en"
DEFAULT_STT_PROVIDER = "xai"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2"

SpeechToTextProvider = Literal["xai", "elevenlabs"]


@dataclass(frozen=True, slots=True)
class SpeechToTextOptions:
    provider: SpeechToTextProvider = DEFAULT_STT_PROVIDER
    language: str | None = DEFAULT_STT_LANGUAGE
    model: str | None = None
    opencc_config: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300


def transcribe_audio_file(
    audio_path: Path,
    *,
    options: SpeechToTextOptions | None = None,
) -> SpeechTranscription:
    options = options or SpeechToTextOptions()
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")

    if options.provider == "xai":
        transcription = transcribe_audio_file_with_xai(audio_path, options=options)
    elif options.provider == "elevenlabs":
        transcription = transcribe_audio_file_with_elevenlabs(audio_path, options=options)
    else:
        raise ValueError(f"Unsupported speech-to-text provider: {options.provider}")
    return convert_transcription_text(transcription, options.opencc_config)


def transcribe_audio_file_with_xai(
    audio_path: Path,
    *,
    options: SpeechToTextOptions,
) -> SpeechTranscription:
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


def transcribe_audio_file_with_elevenlabs(
    audio_path: Path,
    *,
    options: SpeechToTextOptions,
) -> SpeechTranscription:
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


def convert_transcription_text(
    transcription: SpeechTranscription,
    opencc_config: str | None,
) -> SpeechTranscription:
    if opencc_config is None:
        return transcription
    return replace(
        transcription,
        text=convert_zh(
            transcription.text,
            opencc_config,
        ),
    )


def run_json_request(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    provider_name: str,
) -> dict[str, Any]:
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
            f"Content-Type: {detect_audio_content_type(file_path)}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def detect_audio_content_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.casefold()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"
