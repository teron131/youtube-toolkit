from pathlib import Path
from types import SimpleNamespace

from youtube_toolkit.ytdlp.client import STT_AUDIO_FORMAT_SELECTOR, YtDlpClient


def test_stt_audio_format_prefers_xai_and_elevenlabs_compatible_containers() -> None:
    """Audio downloads prefer documented STT upload containers without conversion."""
    assert STT_AUDIO_FORMAT_SELECTOR == "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"


def test_download_transcription_audio_uses_default_cache_stem(
    monkeypatch,
    tmp_path,
) -> None:
    """Audio download paths use the video ID and let yt-dlp choose the extension."""
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run(
        command,
        capture_output,
        text,
        check,
    ):
        """Capture yt-dlp args and create the expected output file."""
        calls["command"] = command
        audio_path = Path(".cache/audio/JpSHyEIZ_bo.m4a")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("youtube_toolkit.ytdlp.client.subprocess.run", fake_run)

    assert YtDlpClient().download_transcription_audio("JpSHyEIZ_bo") == Path(
        ".cache/audio/JpSHyEIZ_bo.m4a",
    )
    assert calls["command"][calls["command"].index("--format") + 1] == STT_AUDIO_FORMAT_SELECTOR
    assert calls["command"][calls["command"].index("--output") + 1] == (".cache/audio/JpSHyEIZ_bo.%(ext)s")
