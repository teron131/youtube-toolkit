"""Stdout JSON policy for command results."""

from __future__ import annotations

import json
import sys

MAX_STDOUT_TEXT_FIELD_CHARS = 400_000
LONG_STDOUT_TEXT_FIELDS = frozenset({"description", "text"})


def emit_json(payload: object) -> None:
    """Strings under LONG_STDOUT_TEXT_FIELDS keys, and a bare string payload, are cut to MAX_STDOUT_TEXT_FIELD_CHARS without a marker."""
    root_key = "text" if isinstance(payload, str) else None
    json.dump(
        stdout_payload(payload, key=root_key), sys.stdout, indent=2, ensure_ascii=False
    )
    print()


def stdout_payload(payload: object, *, key: str | None = None) -> object:
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
