"""Media download commands and directory-local request caching."""

from pathlib import Path
from types import SimpleNamespace

from youtube_toolkit.youtube.client import STT_AUDIO_FORMAT_SELECTOR, YtDlpClient


def test_stt_audio_format_prefers_xai_and_elevenlabs_compatible_containers() -> None:
    """Audio downloads prefer documented STT upload containers without conversion."""
    assert (
        STT_AUDIO_FORMAT_SELECTOR
        == "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"
    )


def test_download_transcription_audio_uses_unique_cache_stem(
    monkeypatch,
    tmp_path,
) -> None:
    """Audio download paths include the video ID without sharing a mutable output file."""
    monkeypatch.chdir(tmp_path)
    commands = []

    def fake_run(
        command,
        capture_output,
        text,
        check,
    ):
        """Capture yt-dlp args and create the expected output file."""
        commands.append(command)
        output_pattern = Path(command[command.index("--output") + 1])
        audio_path = output_pattern.with_name(
            output_pattern.name.replace("%(ext)s", "m4a")
        )
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)

    first_path = YtDlpClient().download_transcription_audio("JpSHyEIZ_bo")
    second_path = YtDlpClient().download_transcription_audio("JpSHyEIZ_bo")

    assert first_path != second_path
    assert first_path.parent == tmp_path / ".cache/youtube-toolkit/audio"
    assert first_path.name.startswith("JpSHyEIZ_bo-")
    assert first_path.suffix == ".m4a"
    assert commands[0][commands[0].index("--format") + 1] == STT_AUDIO_FORMAT_SELECTOR
    assert commands[0][commands[0].index("--output") + 1].startswith(
        str(tmp_path / ".cache/youtube-toolkit/audio/JpSHyEIZ_bo-")
    )
    assert commands[0][commands[0].index("--output") + 1].endswith(".%(ext)s")


def test_channel_videos_omits_playlist_end_when_unlimited(monkeypatch) -> None:
    """Unlimited channel fetches should stay flat and avoid playlist caps."""
    calls = {}

    def fake_json(
        self,
        command,
        *,
        error_message,
    ):
        calls["command"] = command
        calls["error_message"] = error_message
        return {"entries": []}

    monkeypatch.setattr(YtDlpClient, "_json", fake_json)

    assert YtDlpClient().channel_videos(
        "https://www.youtube.com/@OpenAI/videos",
        last=None,
    ) == {"entries": []}
    assert "--flat-playlist" in calls["command"]
    assert "--playlist-end" not in calls["command"]
    assert calls["command"][-1] == "https://www.youtube.com/@OpenAI/videos"


def test_request_cache_reuses_expires_and_recovers(monkeypatch, tmp_path) -> None:
    """Search requests share only identical task-local results and refetch expired or deleted entries."""
    from youtube_toolkit.cache import REQUEST_MAX_AGE_SECONDS, cache_root

    monkeypatch.chdir(tmp_path)
    now = [1000.0]
    monkeypatch.setattr("youtube_toolkit.cache.time.time", lambda: now[0])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--no-cache-dir" in command
        return SimpleNamespace(
            returncode=0, stdout='{"entries": [{"id": "JpSHyEIZ_bo"}]}'
        )

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", run)
    client = YtDlpClient()
    expected = {"entries": [{"id": "JpSHyEIZ_bo"}]}
    assert client.search("agents", fetch=5, sort="relevance") == expected
    assert client.search("agents", fetch=5, sort="relevance") == expected
    assert len(calls) == 1
    client.search("agents", fetch=10, sort="relevance")
    assert len(calls) == 2
    now[0] += REQUEST_MAX_AGE_SECONDS
    client.search("agents", fetch=5, sort="relevance")
    assert len(calls) == 3
    for entry in (cache_root() / "requests").glob("*.json"):
        entry.unlink()
    client.search("agents", fetch=5, sort="relevance")
    assert len(calls) == 4
