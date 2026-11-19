#!/bin/sh
# Installs the youtube command on macOS: Homebrew when missing, then whichever of uv, ffmpeg, and Deno (the JavaScript runtime yt-dlp needs for YouTube) is missing through it, then the toolkit from GitHub as a uv tool.
# Run: curl -LsSf https://raw.githubusercontent.com/teron131/youtube-toolkit/main/install.sh | sh
# Everything sits in main, which runs only once the whole script has arrived, and each installer reads /dev/null, since under `curl | sh` standard input is the rest of this script.
set -eu

PACKAGE="youtube-toolkit[vision] @ https://github.com/teron131/youtube-toolkit/archive/refs/heads/main.zip"
# Apple silicon keeps Homebrew outside the default PATH, so new shells load it from .zprofile, as Homebrew's installer advises; Intel keeps it in /usr/local/bin, already on PATH.
SILICON_BREW=/opt/homebrew/bin/brew

main() {
  if ! command -v brew >/dev/null 2>&1 && [ ! -x "$SILICON_BREW" ]; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/null
  fi
  if ! command -v brew >/dev/null 2>&1; then
    eval "$("$SILICON_BREW" shellenv)"
    grep -qs "brew shellenv" "$HOME/.zprofile" || echo "eval \"\$($SILICON_BREW shellenv)\"" >>"$HOME/.zprofile"
  fi
  for formula in uv ffmpeg deno; do
    command -v "$formula" >/dev/null 2>&1 || brew install "$formula" </dev/null
  done
  uv tool install --force --reinstall "$PACKAGE" </dev/null
  uv tool update-shell </dev/null
  "$(uv tool dir --bin)/youtube" doctor
}

main
