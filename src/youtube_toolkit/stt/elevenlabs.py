"""ElevenLabs speech-to-text provider request construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import urllib.request

from .transcription import encode_multipart_form, run_json_request

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def elevenlabs_stt(
    audio_path: Path,
    *,
    api_key: str,
    model: str,
    language: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Send one audio file to the ElevenLabs STT API and return the decoded JSON response."""
    body, content_type = encode_multipart_form(
        fields={
            "model_id": model,
            "language_code": language,
            "tag_audio_events": "false",
            "timestamps_granularity": "word",
        },
        file_field="file",
        file_path=audio_path,
    )
    request = urllib.request.Request(  # noqa: S310 - fixed HTTPS API endpoint.
        ELEVENLABS_STT_URL,
        data=body,
        headers={
            "xi-api-key": api_key,
            "Content-Type": content_type,
        },
        method="POST",
    )
    return run_json_request(
        request,
        timeout_seconds=timeout_seconds,
        provider_name="ElevenLabs",
    )
