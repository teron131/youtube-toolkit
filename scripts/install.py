"""Build and install the system YouTube CLI, then configure Codex when available."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_LEVEL_EXTRAS = {
    "lite": (),
    "standard": ("storage",),
    "full": ("storage", "vision", "llm"),
}


def run(*arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one required installation command from the repository root."""
    return subprocess.run(  # noqa: S603
        arguments,
        cwd=REPO_ROOT,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def built_wheel_path() -> Path:
    """Resolve the wheel name from project metadata after the build completes."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    distribution = str(project["name"]).replace("-", "_")
    version = str(project["version"])
    return REPO_ROOT / "dist" / f"{distribution}-{version}-py3-none-any.whl"


def wheel_requirement(wheel_path: Path, level: str) -> str:
    """Select the wheel capability extras for one cumulative installation level."""
    extras = INSTALL_LEVEL_EXTRAS[level]
    if not extras:
        return str(wheel_path)
    return f"{wheel_path}[{','.join(extras)}]"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the cumulative dependency level installed into the system tool environment."""
    parser = argparse.ArgumentParser(description="Install the YouTube Toolkit system command.")
    parser.add_argument(
        "--level",
        choices=tuple(INSTALL_LEVEL_EXTRAS),
        default="standard",
        help="lite is stateless, standard adds LanceDB storage, and full adds vision and LLM dependencies",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Replace the system tool at one level and configure storage permissions when applicable."""
    args = parse_args(argv)
    run("uv", "build", "--wheel", "--python", sys.executable)
    wheel_path = built_wheel_path()
    if not wheel_path.exists():
        raise RuntimeError(f"Built wheel is missing: {wheel_path}")
    run("uv", "tool", "install", "--force", "--python", sys.executable, wheel_requirement(wheel_path, args.level))
    tool_bin_dir = Path(run("uv", "tool", "dir", "--bin", capture_output=True).stdout.strip())
    youtube_command = tool_bin_dir / "youtube"
    if not youtube_command.exists():
        raise RuntimeError(f"Installed youtube command is missing: {youtube_command}")
    if args.level != "lite" and shutil.which("codex") is not None:
        run(str(youtube_command), "setup", "codex", "--yes")
    run(str(youtube_command), "doctor")


if __name__ == "__main__":
    main()
