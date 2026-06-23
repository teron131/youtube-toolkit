"""Shared speech-to-text workflow and HTTP helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal
import urllib.error
import urllib.request
from uuid import uuid4

from ..convert_zh import convert_zh
from ..storage import get_connection, get_stored_video, upsert_video
from ..url import extract_video_id
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
    return SpeechTranscription(
        text=convert_zh(transcription.text, "s2hk"),
        language=transcription.language,
        duration_seconds=transcription.duration_seconds,
        word_count=transcription.word_count,
    )


def transcribe_audio_to_db(
    audio_path: Path,
    *,
    options: SpeechToTextOptions | None = None,
) -> dict[str, Any]:
    video_id = extract_video_id(audio_path.stem)
    if video_id is None:
        raise ValueError("audio filename must include a YouTube video ID so the transcript can be stored")

    with get_connection() as connection:
        stored_video = get_stored_video(
            connection,
            video_id,
            require_transcript=True,
        )
    if stored_video is not None:
        audio_path.unlink(missing_ok=True)
        return transcription_result_payload(
            stored_video,
            language=None,
            duration_seconds=None,
            word_count=None,
        )

    transcription = transcribe_audio_file(
        audio_path,
        options=options,
    )
    with get_connection() as connection:
        stored_video = upsert_video(
            connection,
            {"id": video_id},
            transcript=transcription.text,
        )
    if stored_video is None or not stored_video["transcript"]:
        raise RuntimeError(f"Stored transcript could not be found after writing video {video_id}")
    audio_path.unlink()
    return transcription_result_payload(
        stored_video,
        language=transcription.language,
        duration_seconds=transcription.duration_seconds,
        word_count=transcription.word_count,
    )


def transcription_result_payload(
    stored_video: dict[str, Any],
    *,
    language: str | None,
    duration_seconds: float | None,
    word_count: int | None,
) -> dict[str, Any]:
    return {
        **stored_video,
        "language": language,
        "duration_seconds": duration_seconds,
        "word_count": word_count,
    }


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
