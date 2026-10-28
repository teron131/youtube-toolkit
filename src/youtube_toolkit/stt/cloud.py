"""Cloud speech-to-text for audio files, with each provider described by a row of settings and sent in one of three request styles.

ElevenLabs, OpenAI, and xAI take a multipart file upload.
Gemini reads audio from its own file store, and Vercel AI Gateway takes base64 audio inside JSON, so each has its own sender.

Provider documentation:

- ElevenLabs: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Gemini: https://ai.google.dev/gemini-api/docs/generate-content/transcribe
- OpenAI: https://developers.openai.com/api/docs/guides/speech-to-text
- Vercel AI Gateway: https://vercel.com/docs/ai-gateway/modalities/speech-to-text
- xAI: https://docs.x.ai/developers/model-capabilities/audio/speech-to-text
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from functools import cache
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Literal
import urllib.error
import urllib.request
from uuid import uuid4

from ..config import (
    DEFAULT_ELEVENLABS_STT_MODEL,
    DEFAULT_GEMINI_STT_MODEL,
    DEFAULT_OPENAI_STT_MODEL,
    DEFAULT_STT_PROVIDER,
    DEFAULT_VERCEL_STT_MODEL,
    DEFAULT_XAI_STT_MODEL,
    config_value,
    stt_provider_name,
)
from ..execution import retry_transient
from ..schemas import SpeechTranscription, TranscriptChunk, normalize_text
from .synthetic_audio import tone_recording

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CloudProvider:
    """No language is sent, so the provider detects it; a fixed language would bias the transcription toward it."""

    label: str
    url: str
    api_key_env: str
    auth_header: str
    model: str
    model_field: str = "model"
    auth_prefix: str = ""
    extra_fields: dict[str, str] = field(default_factory=dict)
    language_key: str = "language"
    request_style: Literal["gemini", "multipart", "vercel"] = "multipart"


PROVIDERS = {
    "elevenlabs": CloudProvider(
        label="ElevenLabs",
        url="https://api.elevenlabs.io/v1/speech-to-text",
        api_key_env="ELEVENLABS_API_KEY",
        auth_header="xi-api-key",
        model=DEFAULT_ELEVENLABS_STT_MODEL,
        model_field="model_id",
        extra_fields={"tag_audio_events": "false", "timestamps_granularity": "word"},
        language_key="language_code",
    ),
    "gemini": CloudProvider(
        label="Gemini",
        url="https://generativelanguage.googleapis.com",
        api_key_env="GEMINI_API_KEY",
        auth_header="x-goog-api-key",
        model=DEFAULT_GEMINI_STT_MODEL,
        request_style="gemini",
    ),
    "openai": CloudProvider(
        label="OpenAI",
        url="https://api.openai.com/v1/audio/transcriptions",
        api_key_env="OPENAI_API_KEY",
        auth_header="Authorization",
        model=DEFAULT_OPENAI_STT_MODEL,
        auth_prefix="Bearer ",
        language_key="languages",
    ),
    "vercel": CloudProvider(
        label="Vercel AI Gateway",
        url="https://ai-gateway.vercel.sh/v4/ai/transcription-model",
        api_key_env="AI_GATEWAY_API_KEY",
        auth_header="Authorization",
        model=DEFAULT_VERCEL_STT_MODEL,
        auth_prefix="Bearer ",
        request_style="vercel",
    ),
    "xai": CloudProvider(
        label="xAI",
        url="https://api.x.ai/v1/stt",
        api_key_env="XAI_API_KEY",
        auth_header="Authorization",
        model=DEFAULT_XAI_STT_MODEL,
        auth_prefix="Bearer ",
    ),
}
# The gateway's transcription route is versioned through these two headers.
VERCEL_PROTOCOL_HEADERS = {
    "ai-gateway-protocol-version": "0.0.1",
    "ai-transcription-model-specification-version": "4",
}
REQUEST_TIMEOUT_SECONDS = 300
REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
# Word timings become caption-like lines, which end at sentence punctuation, before a pause, or once they span LINE_MAX_SECONDS.
LINE_ENDINGS = (".", "?", "!", "\u3002", "\uff1f", "\uff01")
LINE_PAUSE_SECONDS = 1.0
LINE_MAX_SECONDS = 10.0
# Words that meet at a CJK character join without a space, as the language is written.
CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
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


def cloud_provider() -> CloudProvider:
    """An unknown STT_PROVIDER falls back to the default provider, with one warning per process."""
    return PROVIDERS[_known_provider(stt_provider_name())]


@cache
def _known_provider(name: str) -> str:
    if name in PROVIDERS:
        return name
    logger.warning(
        "STT_PROVIDER=%s is not one of %s; falling back to the default %s",
        name,
        ", ".join(PROVIDERS),
        DEFAULT_STT_PROVIDER,
    )
    return DEFAULT_STT_PROVIDER


def cloud_stt_configured() -> bool:
    return config_value(cloud_provider().api_key_env) is not None


def transcribe_in_cloud(audio_path: Path) -> SpeechTranscription:
    if not audio_path.is_file():
        raise ValueError(f"audio file does not exist: {audio_path}")
    provider = cloud_provider()
    transcription = _send(audio_path, provider)
    if transcription is None:
        raise ValueError(
            f"{provider.label} STT response did not include transcript text"
        )
    return transcription


def cloud_stt_status() -> dict[str, Any]:
    provider = cloud_provider()
    return {
        "provider": provider.label,
        "model": provider.model,
        "key_set": config_value(provider.api_key_env) is not None,
    }


def check_cloud_stt() -> dict[str, Any]:
    """This is a real request that the provider bills, so it runs only when the user asks for it.

    A rejected key, endpoint, or model raises with the provider's reason.
    """
    provider = cloud_provider()
    with tone_recording() as tone_path:
        _send(tone_path, provider)
    return {**cloud_stt_status(), "working": True}


def _send(audio_path: Path, provider: CloudProvider) -> SpeechTranscription | None:
    """None means the provider heard no speech."""
    api_key = config_value(provider.api_key_env)
    if api_key is None:
        raise ValueError(
            f"{provider.api_key_env} is required for {provider.label} speech-to-text"
        )
    auth = {provider.auth_header: f"{provider.auth_prefix}{api_key}"}
    if provider.request_style == "gemini":
        return _send_to_gemini(audio_path, provider, auth)
    if provider.request_style == "vercel":
        return _send_to_vercel(audio_path, provider, auth)
    return _send_multipart(audio_path, provider, auth)


def _send_to_gemini(
    audio_path: Path, provider: CloudProvider, auth: dict[str, str]
) -> SpeechTranscription | None:
    """The file store takes recordings of any size; sending audio inside the request is capped at about 20 MB.

    Gemini reports no detected language, so the result carries none.
    Its word timestamps would cap audio at 30 minutes and can lower accuracy, so the result stays untimed.
    """
    uploaded = _request(
        provider,
        f"{provider.url}/upload/v1beta/files?uploadType=media",
        audio_path.read_bytes(),
        {**auth, "Content-Type": audio_content_type(audio_path)},
    )["file"]
    try:
        reference = {"fileUri": uploaded["uri"], "mimeType": uploaded["mimeType"]}
        response = _request(
            provider,
            f"{provider.url}/v1beta/models/{provider.model}:generateContent",
            json.dumps({"contents": [{"parts": [{"fileData": reference}]}]}).encode(),
            {**auth, "Content-Type": "application/json"},
        )
    finally:
        _delete_gemini_file(uploaded["name"], provider, auth)
    candidates = response.get("candidates") or [{}]
    parts = candidates[0].get("content", {}).get("parts", [])
    text = " ".join(
        part_text
        for part in parts
        if (part_text := normalize_text(part.get("audioTranscription", {}).get("text")))
    )
    return SpeechTranscription(text=text, segments=[]) if text else None


def _delete_gemini_file(
    name: str, provider: CloudProvider, auth: dict[str, str]
) -> None:
    """Stored files expire by themselves after two days, so a failed delete must not cost the transcript."""
    try:
        _request(provider, f"{provider.url}/v1beta/{name}", None, auth, method="DELETE")
    except RuntimeError as error:
        logger.warning("Could not delete the uploaded Gemini file: %s", error)


def _send_to_vercel(
    audio_path: Path, provider: CloudProvider, auth: dict[str, str]
) -> SpeechTranscription | None:
    body = json.dumps(
        {
            "audio": base64.b64encode(audio_path.read_bytes()).decode(),
            "mediaType": audio_content_type(audio_path),
        }
    ).encode()
    headers = {
        **auth,
        **VERCEL_PROTOCOL_HEADERS,
        "ai-model-id": provider.model,
        "Content-Type": "application/json",
    }
    return parse_transcription(
        _request(provider, provider.url, body, headers), provider
    )


def _send_multipart(
    audio_path: Path, provider: CloudProvider, auth: dict[str, str]
) -> SpeechTranscription | None:
    body, content_type = encode_multipart_form(
        fields={provider.model_field: provider.model, **provider.extra_fields},
        file_path=audio_path,
    )
    response = _request(
        provider, provider.url, body, {**auth, "Content-Type": content_type}
    )
    return parse_transcription(response, provider)


def parse_transcription(
    response: dict[str, Any], provider: CloudProvider
) -> SpeechTranscription | None:
    """Per-channel transcripts are joined when there is no single text, and stay untimed.

    xAI and ElevenLabs time each word, Vercel AI Gateway times segments when its model does, and OpenAI's current models report no timing.
    The language is a plain code for most providers; OpenAI sends a list of entries, and the first entry's code is used.
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
    language = response.get(provider.language_key)
    if isinstance(language, list) and language and isinstance(language[0], dict):
        language = language[0].get("code")
    return SpeechTranscription(
        text=text,
        segments=_timed_words(response.get("words"))
        or _gateway_segments(response.get("segments")),
        language=normalize_text(language),
    )


def _timed_words(words: Any) -> list[TranscriptChunk]:
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


def _gateway_segments(segments: Any) -> list[TranscriptChunk]:
    if not isinstance(segments, list):
        return []
    return [
        TranscriptChunk(start_seconds=start, end_seconds=end, text=segment_text)
        for segment in segments
        if isinstance(segment, dict)
        and (start := _seconds(segment.get("startSecond"))) is not None
        and (end := _seconds(segment.get("endSecond"))) is not None
        and (segment_text := normalize_text(segment.get("text")))
    ]


def _lines(words: list[tuple[float, float, str]]) -> list[TranscriptChunk]:
    lines: list[TranscriptChunk] = []
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
                TranscriptChunk(
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


def audio_content_type(audio_path: Path) -> str:
    return AUDIO_CONTENT_TYPES.get(
        audio_path.suffix.casefold(), "application/octet-stream"
    )


def _request(
    provider: CloudProvider,
    url: str,
    data: bytes | None,
    headers: dict[str, str],
    *,
    method: str = "POST",
) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - fixed HTTPS API endpoints from the provider table.
        url, data=data, headers=headers, method=method
    )
    return run_json_request(request, provider_name=provider.label)


def run_json_request(
    request: urllib.request.Request, *, provider_name: str
) -> dict[str, Any]:
    def request_payload() -> dict[str, Any]:
        with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS API endpoint.
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw_response = response.read()
        response_data = json.loads(raw_response)
        if not isinstance(response_data, dict):
            raise json.JSONDecodeError("STT response was not a JSON object", "", 0)
        return response_data

    try:
        return retry_transient(
            request_payload,
            attempts=REQUEST_ATTEMPTS,
            backoff_seconds=RETRY_BACKOFF_SECONDS,
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
    fields: dict[str, str],
    file_path: Path,
) -> tuple[bytes, str]:
    """Every multipart provider expects the audio under the `file` field."""
    boundary = f"----youtube-toolkit-{uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
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
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(),
            f"Content-Type: {audio_content_type(file_path)}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
