"""CLI entrypoint: command results go to stdout, and expected failures become one stderr line with exit status 1."""

from __future__ import annotations

import sys

from . import output
from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
        if args.command == "doctor":
            print(payload)
        else:
            output.emit_json(payload)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
