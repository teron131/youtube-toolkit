"""JSON output policy and human-readable command renderers."""

from __future__ import annotations

import json
import sys

MAX_STDOUT_TEXT_FIELD_CHARS = 400_000
LONG_STDOUT_TEXT_FIELDS = frozenset(
    {"description", "overview", "summary", "text", "transcript"}
)


def emit_json(payload: object) -> None:
    """Agent-readable command payloads go through stdout text caps before JSON emission."""
    root_key = "text" if isinstance(payload, str) else None
    json.dump(
        stdout_payload(payload, key=root_key), sys.stdout, indent=2, ensure_ascii=False
    )
    print()


def render_doctor(payload: object) -> str:
    """Render the health-check payload for humans while preserving JSON output for other commands."""
    if not isinstance(payload, dict):
        return str(payload)

    embedding = payload.get("embedding", {})
    detail = (
        embedding.get("detail", "unknown")
        if isinstance(embedding, dict)
        else str(embedding)
    )
    return "\n".join(
        [
            "YouTube Toolkit",
            f"  yt-dlp     {payload.get('yt_dlp_version', 'unknown')}",
            f"  cache      {payload.get('cache', 'unknown')}",
            f"  vision     {'available' if payload.get('vision') else 'not installed'}",
            f"  ffmpeg     {'available' if payload.get('ffmpeg') else 'not installed'}",
            f"  embedding  {detail}",
        ]
    )


def stdout_payload(payload: object, *, key: str | None = None) -> object:
    """Cap only known long text fields while preserving valid JSON payloads."""
    if isinstance(payload, dict):
        return {
            item_key: stdout_payload(item_value, key=str(item_key))
            for item_key, item_value in payload.items()
        }
    if isinstance(payload, list):
        return [stdout_payload(item, key=key) for item in payload]
    if isinstance(payload, str) and key in LONG_STDOUT_TEXT_FIELDS:
        return payload[:MAX_STDOUT_TEXT_FIELD_CHARS]
    return payload
