"""Build and install the system YouTube CLI, then configure Codex when available."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    """Replace any existing tool install and idempotently grant its storage root to Codex."""
    run("uv", "build", "--wheel", "--python", sys.executable)
    wheel_path = built_wheel_path()
    if not wheel_path.exists():
        raise RuntimeError(f"Built wheel is missing: {wheel_path}")
    run("uv", "tool", "install", "--force", "--python", sys.executable, str(wheel_path))
    tool_bin_dir = Path(run("uv", "tool", "dir", "--bin", capture_output=True).stdout.strip())
    youtube_command = tool_bin_dir / "youtube"
    if not youtube_command.exists():
        raise RuntimeError(f"Installed youtube command is missing: {youtube_command}")
    if shutil.which("codex") is not None:
        run(str(youtube_command), "setup", "codex", "--yes")
    run(str(youtube_command), "doctor")


if __name__ == "__main__":
    main()
