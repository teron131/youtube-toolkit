"""Media download commands and directory-local retrieval archives."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from youtube_toolkit.youtube.client import STT_AUDIO_FORMAT_SELECTOR, YtDlpClient


def test_stt_audio_format_prefers_xai_and_elevenlabs_compatible_containers() -> None:
    """Audio downloads prefer documented STT upload containers without conversion."""
    assert (
        STT_AUDIO_FORMAT_SELECTOR
        == "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"
    )


def test_download_transcription_audio_uses_unique_artifact_stem(
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
        timeout,
    ):
        """Capture yt-dlp args and create the expected output file."""
        assert timeout == 30
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
        namespace,
        identifier,
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


@pytest.mark.parametrize(
    ("namespace", "identifier", "retrieve"),
    [
        ("metadata", "JpSHyEIZ_bo", lambda client: client.metadata("JpSHyEIZ_bo")),
        (
            "searches",
            "agents",
            lambda client: client.search("agents", fetch=5, sort="relevance"),
        ),
        (
            "channels",
            "OpenAI",
            lambda client: client.channel_videos(
                "https://www.youtube.com/@OpenAI/videos", last=5
            ),
        ),
    ],
)
def test_identical_requests_refetch_and_preserve_history(
    monkeypatch, tmp_path, namespace, identifier, retrieve
) -> None:
    """Every retrieval contacts yt-dlp and keeps earlier responses searchable by identifier."""
    import json

    from youtube_toolkit.artifacts import artifact_root

    monkeypatch.chdir(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--no-cache-dir" in command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"id": "JpSHyEIZ_bo", "generation": len(calls)}),
        )

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", run)
    client = YtDlpClient()
    assert retrieve(client)["generation"] == 1
    first_path = next((artifact_root() / namespace).glob("*.json"))
    first_copy = first_path.read_bytes()
    assert retrieve(client)["generation"] == 2
    assert len(calls) == 2
    paths = list((artifact_root() / namespace).glob("*.json"))
    assert len(paths) == 2
    assert all(identifier in path.name for path in paths)
    assert first_path.read_bytes() == first_copy
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert sorted(entry["payload"]["generation"] for entry in entries) == [1, 2]
    assert all(
        entry["request"] and entry["fetched_at"].endswith("+00:00") for entry in entries
    )


def test_artifact_names_sanitize_identifiers_and_preserve_same_time_copies(
    monkeypatch, tmp_path
) -> None:
    """Unsafe identifiers stay within their namespace and same-time writes keep every payload."""
    from datetime import UTC, datetime
    import json

    from youtube_toolkit import artifacts

    class FrozenDatetime:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return datetime(2026, 10, 8, 14, 30, 12, 123456, tzinfo=UTC)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(artifacts, "datetime", FrozenDatetime)
    paths = [
        artifacts.write_artifact(
            "searches",
            "../agents / tools",
            {"query": "agents / tools"},
            {"generation": generation},
        )
        for generation in (1, 2)
    ]
    assert paths[0] != paths[1]
    assert all(path.parent == artifacts.artifact_root() / "searches" for path in paths)
    assert all(path.name.startswith("2026-10-08T143012.123456Z-") for path in paths)
    assert all("agents" in path.name and "tools" in path.name for path in paths)
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [entry["payload"]["generation"] for entry in entries] == [1, 2]
    assert all(entry["request"] == {"query": "agents / tools"} for entry in entries)
    assert all(
        entry["fetched_at"] == "2026-10-08T14:30:12.123456+00:00" for entry in entries
    )


@pytest.mark.parametrize("timed_out", [False, True])
def test_failed_audio_download_cleans_only_its_own_partial_files(
    monkeypatch, tmp_path, timed_out
) -> None:
    """Failures and timeouts release this download's partial files without deleting other audio."""
    from youtube_toolkit.artifacts import artifact_root

    monkeypatch.chdir(tmp_path)
    audio_dir = artifact_root() / "audio"
    audio_dir.mkdir(parents=True)
    unrelated = audio_dir / "keep.m4a"
    unrelated.write_bytes(b"existing audio")

    def fail(command, **kwargs):
        assert kwargs["timeout"] == 30
        pattern = Path(command[command.index("--output") + 1])
        pattern.with_suffix(".m4a.part").write_bytes(b"partial")
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout=30)
        return SimpleNamespace(returncode=1, stderr="download failed")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fail)
    with pytest.raises(
        RuntimeError, match="timed out" if timed_out else "download failed"
    ):
        YtDlpClient().download_transcription_audio("JpSHyEIZ_bo")
    assert list(audio_dir.iterdir()) == [unrelated]
    assert unrelated.read_bytes() == b"existing audio"
