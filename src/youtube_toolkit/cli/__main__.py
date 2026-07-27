"""CLI entrypoint and command dispatch."""

from __future__ import annotations

import sys

from ..capabilities import capability_available, require_capability
from . import output
from .parser import build_parser

DATABASE_PERMISSION_ERROR = (
    "Database write access was denied. Grant this process write access to the active database and retry; run `youtube doctor` to see its path. "
    "Set YOUTUBE_TOOLKIT_STORAGE_HOME only when you intentionally want a different database."
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch parsed CLI commands and keep user-facing failures on stderr."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        required_capability = getattr(args, "required_capability", None)
        if required_capability is not None:
            require_capability(
                required_capability,
                args.capability_feature,
            )
        payload = args.handler(args)
        if args.command == "db" and args.database_command == "view":
            return 0
        if args.command == "doctor":
            print(output.render_doctor(payload))
        elif args.command == "setup" and getattr(args, "setup_command", None) == "codex":
            print(output.render_codex_setup(payload))
        elif args.command == "setup":
            print(output.render_model_setup(payload))
        elif args.command == "index":
            print(output.render_index(payload))
        else:
            output.emit_json(payload)
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {user_facing_error(exc)}", file=sys.stderr)
        return 1


def user_facing_error(error: RuntimeError | ValueError) -> str:
    """Translate opaque LanceDB permission failures without hiding other command errors."""
    if not capability_available("storage"):
        return str(error)
    try:
        from ..storage.connection import is_database_write_permission_error
    except ImportError:
        return str(error)
    if is_database_write_permission_error(error):
        return DATABASE_PERMISSION_ERROR
    return str(error)


if __name__ == "__main__":
    raise SystemExit(main())
