# Installs the youtube command on Windows: whichever of uv, FFmpeg, and Deno (the JavaScript runtime yt-dlp needs for YouTube) is missing through winget, then the toolkit from GitHub as a uv tool.
# Run in PowerShell: irm https://raw.githubusercontent.com/teron131/youtube-toolkit/main/install.ps1 | iex
# The script block keeps its settings and variables out of the session that runs it; Windows PowerShell does not stop on a failing program's exit code, so each one is checked.
& {
    $ErrorActionPreference = "Stop"
    $Package = "youtube-toolkit[vision] @ https://github.com/teron131/youtube-toolkit/archive/refs/heads/main.zip"
    $Tools = @(
        @{ Command = "uv"; Id = "astral-sh.uv" },
        @{ Command = "ffmpeg"; Id = "Gyan.FFmpeg" },
        @{ Command = "deno"; Id = "DenoLand.Deno" }
    )

    function Update-SessionPath {
        # winget and uv add to the saved PATH, which this session sees only once it reads it again.
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    }

    foreach ($Tool in $Tools) {
        if (-not (Get-Command $Tool.Command -ErrorAction SilentlyContinue)) {
            winget install --id $Tool.Id --exact --silent --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) { throw "winget could not install $($Tool.Id)" }
            Update-SessionPath
        }
    }
    uv tool install --force --reinstall $Package
    if ($LASTEXITCODE -ne 0) { throw "uv could not install the toolkit" }
    uv tool update-shell
    Update-SessionPath
    & (Join-Path (uv tool dir --bin) "youtube.exe") doctor
}
