"""Command-line package surface."""

from __future__ import annotations

import json
import sys

from .parser import build_parser

__all__ = ["main"]

MAX_STDOUT_TEXT_FIELD_CHARS = 400_000
LONG_STDOUT_TEXT_FIELDS = frozenset({"description", "overview", "summary", "text", "transcript"})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        emit_json(args.handler(args))
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def emit_json(payload: object) -> None:
    json.dump(stdout_payload(payload), sys.stdout, indent=2, ensure_ascii=False)
    print()


def stdout_payload(payload: object, *, key: str | None = None) -> object:
    """Cap only known long text fields while preserving valid JSON payloads."""
    if isinstance(payload, dict):
        return {item_key: stdout_payload(item_value, key=str(item_key)) for item_key, item_value in payload.items()}
    if isinstance(payload, list):
        return [stdout_payload(item, key=key) for item in payload]
    if isinstance(payload, str) and key in LONG_STDOUT_TEXT_FIELDS:
        return payload[:MAX_STDOUT_TEXT_FIELD_CHARS]
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
