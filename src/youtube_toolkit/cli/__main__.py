"""CLI entrypoint and command dispatch."""

from __future__ import annotations

import sys

from ..capabilities import require_capability
from . import output
from .parser import build_parser


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
        if args.command == "doctor":
            print(output.render_doctor(payload))
        else:
            output.emit_json(payload)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
