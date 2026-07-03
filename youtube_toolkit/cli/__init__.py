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
        payload = args.handler(args)
        if args.command == "doctor":
            print(render_doctor(payload))
        else:
            emit_json(payload)
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def emit_json(payload: object) -> None:
    json.dump(stdout_payload(payload), sys.stdout, indent=2, ensure_ascii=False)
    print()


def render_doctor(payload: object) -> str:
    if not isinstance(payload, dict):
        return str(payload)

    database_schema = payload.get("database_schema", {})
    watchlist_schedule = payload.get("watchlist_schedule", {})
    model_apis = payload.get("model_apis", {})
    embedding = model_apis.get("embedding", {}) if isinstance(model_apis, dict) else {}
    llm = model_apis.get("llm", {}) if isinstance(model_apis, dict) else {}
    return "\n".join(
        [
            "YouTube Toolkit",
            f"  yt-dlp     {payload.get('yt_dlp_version', 'unknown')}",
            f"  database   {payload.get('database', 'unknown')}",
            f"  schema     {health_text(database_schema)}",
            "",
            "Watchlist",
            f"  schedule   {schedule_health_text(watchlist_schedule)}",
            "",
            "Models",
            f"  embedding  {model_health_text(embedding, setup_hint='set EMBEDDING_API_KEY')}",
            f"  llm        {model_health_text(llm, setup_hint='set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL')}",
        ]
    )


def health_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    if payload.get("ok"):
        return status
    detail = payload.get("detail")
    return f"{status} - {detail}" if detail else status


def schedule_health_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    detail = str(payload.get("detail", ""))
    if payload.get("ok"):
        interval = schedule_interval_text(detail)
        return f"{status} ({interval})" if interval else status
    return f"{status} - {detail}" if detail else status


def schedule_interval_text(detail: str) -> str | None:
    if "runs every" not in detail:
        return None
    seconds_text = detail.rsplit("runs every ", 1)[-1].removesuffix(" seconds")
    if not seconds_text.isdigit():
        return None
    seconds = int(seconds_text)
    if seconds % 3600 == 0:
        return f"every {seconds // 3600}h"
    if seconds % 60 == 0:
        return f"every {seconds // 60}m"
    return f"every {seconds}s"


def model_health_text(
    payload: object,
    *,
    setup_hint: str,
) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    model_name = payload.get("model")
    model_suffix = f" ({model_name})" if model_name and model_name != "not configured" else ""
    if payload.get("ok"):
        return f"{status}{model_suffix}"
    hint = setup_hint if status == "missing" else str(payload.get("detail", ""))
    return f"{status}{model_suffix} - {hint}" if hint else f"{status}{model_suffix}"


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
