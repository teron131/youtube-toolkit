"""STT adapters share multipart encoding and bounded HTTP retries without depending on orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request
from uuid import uuid4

from ..execution import retry_transient

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
        )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{provider_name} STT request failed ({error.code}): {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"{provider_name} STT request failed: {error.reason}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{provider_name} STT response was not valid JSON"
        ) from error


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
