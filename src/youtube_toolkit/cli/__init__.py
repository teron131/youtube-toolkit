"""CLI entrypoint, JSON output policy, and human-readable command renderers."""

from __future__ import annotations

import json
import sys
from typing import cast

from ..storage.connection import DatabaseWritePermissionError
from .parser import build_parser

__all__ = ["main"]

MAX_STDOUT_TEXT_FIELD_CHARS = 400_000
LONG_STDOUT_TEXT_FIELDS = frozenset({"description", "overview", "summary", "text", "transcript"})
DATABASE_PERMISSION_ERROR = (
    "Database write access was denied. Grant this process write access to the active database and retry; run `youtube doctor` to see its path. "
    "Set YOUTUBE_TOOLKIT_STORAGE_HOME only when you intentionally want a different database."
)
PERMISSION_ERROR_MARKERS = (
    "operation not permitted",
    "permission denied",
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch parsed CLI commands and keep user-facing failures on stderr."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
        if args.command == "db" and args.database_command == "view":
            return 0
        if args.command == "doctor":
            print(render_doctor(payload))
        elif args.command == "setup" and getattr(args, "setup_command", None) == "codex":
            print(render_codex_setup(payload))
        elif args.command == "setup":
            print(render_model_setup(payload))
        elif args.command == "index":
            print(render_index(payload))
        else:
            emit_json(payload)
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {user_facing_error(exc)}", file=sys.stderr)
        return 1


def user_facing_error(error: RuntimeError | ValueError) -> str:
    """Translate opaque LanceDB permission failures without hiding other command errors."""
    detail = str(error)
    normalized_detail = detail.lower()
    is_lance_permission_error = "lance error" in normalized_detail and any(marker in normalized_detail for marker in PERMISSION_ERROR_MARKERS)
    if isinstance(error, DatabaseWritePermissionError) or is_lance_permission_error:
        return DATABASE_PERMISSION_ERROR
    return detail


def emit_json(payload: object) -> None:
    """Agent-readable command payloads go through stdout text caps before JSON emission."""
    json.dump(stdout_payload(payload), sys.stdout, indent=2, ensure_ascii=False)
    print()


def render_doctor(payload: object) -> str:
    """Render the health-check payload for humans while preserving JSON output for other commands."""
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


def render_model_setup(payload: object) -> str:
    """Render model setup state without exposing configured secret values."""
    if not isinstance(payload, dict):
        return str(payload)

    model_apis = payload.get("model_apis", {})
    settings = payload.get("settings", {})
    verification = payload.get("verification", {})
    embedding = model_apis.get("embedding", {}) if isinstance(model_apis, dict) else {}
    llm = model_apis.get("llm", {}) if isinstance(model_apis, dict) else {}
    auto_embeddings = settings.get("auto_embeddings") if isinstance(settings, dict) else None
    embedding_verification = verification.get("embedding", {}) if isinstance(verification, dict) else {}
    llm_verification = verification.get("llm", {}) if isinstance(verification, dict) else {}
    embedding_verification_text = verification_text(
        embedding_verification,
        ("Embedding API returned ", "Embedding verification failed: "),
    )
    llm_verification_text = verification_text(
        llm_verification,
        ("LLM API returned ", "LLM verification failed: "),
    )
    return "\n".join(
        [
            "Model Setup",
            f"  config     {payload.get('config', 'unknown')}",
            "",
            "Models",
            f"  embedding  {model_health_text(embedding, setup_hint='set EMBEDDING_API_KEY')}",
            f"  llm        {model_health_text(llm, setup_hint='set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL')}",
            "",
            "Settings",
            f"  auto index {setting_text(auto_embeddings)}",
            "",
            "Verification",
            f"  embedding  {embedding_verification_text}",
            f"  llm        {llm_verification_text}",
        ]
    )


def render_codex_setup(payload: object) -> str:
    """Render the Codex sandbox integration without hiding its selected paths."""
    if not isinstance(payload, dict):
        return str(payload)
    lines = [
        "Codex Setup",
        f"  config      {payload.get('config', 'unknown')}",
        f"  write root  {payload.get('writable_root', 'unknown')}",
        f"  status      {payload.get('status', 'unknown')}",
    ]
    if payload.get("status") in {"configured", "already_configured"}:
        lines.extend(("", "Start a new Codex task if this task does not already have write access."))
    return "\n".join(lines)


def render_index(payload: object) -> str:
    """Render embedding index results with per-video failure detail only when needed."""
    if not isinstance(payload, dict):
        return str(payload)

    indexed = payload_list(payload.get("indexed"))
    skipped = payload_list(payload.get("skipped"))
    failed = payload_list(payload.get("failed"))
    lines = [
        "Embedding Index",
        f"  videos   {payload.get('videos', 0)}",
        f"  indexed  {len(indexed)}{index_dimensions_text(indexed)}",
        f"  skipped  {len(skipped)}",
        f"  failed   {len(failed)}",
    ]
    append_index_details(lines, "Skipped", skipped)
    append_index_details(lines, "Failed", failed)
    return "\n".join(lines)


def payload_list(value: object) -> list[dict[str, object]]:
    """Narrow untrusted command payload fields before rendering their item details."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]


def index_dimensions_text(indexed: list[dict[str, object]]) -> str:
    """Show embedding dimensions only when every indexed result reports the same size."""
    dimensions = {item.get("dimensions") for item in indexed if item.get("dimensions") is not None}
    if len(dimensions) != 1:
        return ""
    dimension = dimensions.pop()
    unit = "dimension" if dimension == 1 else "dimensions"
    return f" ({dimension} {unit})"


def append_index_details(
    lines: list[str],
    title: str,
    items: list[dict[str, object]],
) -> None:
    """Append compact per-video detail blocks to the human index renderer."""
    if not items:
        return
    lines.extend(("", title))
    for item in items:
        video_id = item.get("video_id", "unknown")
        detail = item.get("detail") or item.get("error") or item.get("status", "")
        lines.append(f"  {video_id}  {detail}")


def health_text(payload: object) -> str:
    """Collapse structured health payloads into one status line."""
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    if payload.get("ok"):
        return status
    detail = payload.get("detail")
    return f"{status} - {detail}" if detail else status


def schedule_health_text(payload: object) -> str:
    """Collapse LaunchAgent health into one status line with interval detail when available."""
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    detail = str(payload.get("detail", ""))
    if payload.get("ok"):
        interval = schedule_interval_text(detail)
        return f"{status} ({interval})" if interval else status
    return f"{status} - {detail}" if detail else status


def schedule_interval_text(detail: str) -> str | None:
    """Extract the scheduler interval phrase from LaunchAgent status detail text."""
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
    """Format one model API health result with a setup hint only for missing config."""
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    model_name = payload.get("model")
    model_suffix = f" ({model_name})" if model_name and model_name != "not configured" else ""
    if payload.get("ok"):
        return f"{status}{model_suffix}"
    hint = setup_hint if status == "missing" else str(payload.get("detail", ""))
    return f"{status}{model_suffix} - {hint}" if hint else f"{status}{model_suffix}"


def setting_text(value: object) -> str:
    """Render boolean setup settings in the CLI's compact on/off vocabulary."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "unknown"


def verification_text(payload: object, detail_prefixes: tuple[str, ...]) -> str:
    """Strip provider-specific verification prefixes before showing setup results."""
    if not isinstance(payload, dict):
        return str(payload)
    status = str(payload.get("status", "unknown"))
    detail = str(payload.get("detail", ""))
    for prefix in detail_prefixes:
        detail = detail.removeprefix(prefix)
    if not detail or detail == "Verification skipped":
        return status
    return f"{status} - {detail}"


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
