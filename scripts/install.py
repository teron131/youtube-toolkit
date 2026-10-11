"""Build and install the YouTube CLI with optional visual extraction and local speech-to-text dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(
    *arguments: str, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
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
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    distribution = str(project["name"]).replace("-", "_")
    version = str(project["version"])
    return REPO_ROOT / "dist" / f"{distribution}-{version}-py3-none-any.whl"


def wheel_requirement(
    wheel_path: Path, *, vision: bool = False, stt: bool = False
) -> str:
    """Select the optional capabilities to install with the wheel."""
    extras = [name for name, wanted in (("vision", vision), ("stt", stt)) if wanted]
    return f"{wheel_path}[{','.join(extras)}]" if extras else str(wheel_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the optional capability installation flags."""
    parser = argparse.ArgumentParser(
        description="Install the YouTube Toolkit system command."
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Include visual extraction dependencies; ffmpeg must be installed separately",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Include local speech-to-text dependencies; ffmpeg must be installed separately; STT_LOCAL_MODEL can name any Hugging Face ASR model that runs in the transformers automatic-speech-recognition pipeline",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Replace the installed system command and report retrieval readiness."""
    args = parse_args(argv)
    run("uv", "build", "--wheel", "--python", sys.executable)
    wheel_path = built_wheel_path()
    if not wheel_path.exists():
        raise RuntimeError(f"Built wheel is missing: {wheel_path}")
    run(
        "uv",
        "tool",
        "install",
        "--force",
        "--python",
        sys.executable,
        wheel_requirement(wheel_path, vision=args.vision, stt=args.stt),
    )
    tool_bin_dir = Path(
        run("uv", "tool", "dir", "--bin", capture_output=True).stdout.strip()
    )
    youtube_command = shutil.which(str(tool_bin_dir / "youtube"))
    if youtube_command is None:
        raise RuntimeError(
            f"Installed youtube command is missing or not executable in {tool_bin_dir}"
        )
    run(youtube_command, "doctor")


if __name__ == "__main__":
    main()
