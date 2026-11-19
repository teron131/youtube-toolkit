"""Media download commands and directory-local retrieval records."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from youtube_toolkit.youtube.client import STT_AUDIO_FORMAT_SELECTOR, YtDlpClient


def test_stt_audio_format_prefers_xai_and_elevenlabs_compatible_containers() -> None:
    assert (
        STT_AUDIO_FORMAT_SELECTOR
        == "bestaudio[ext=m4a]/bestaudio[ext=mp4]/bestaudio[ext=aac]"
    )


def test_download_transcription_audio_saves_into_the_given_directory(
    monkeypatch,
    tmp_path,
) -> None:
    commands = []

    def fake_run(
        command,
        input,
        capture_output,
        text,
        check,
        timeout,
    ):
        assert timeout == 600
        commands.append(command)
        output_pattern = Path(command[command.index("--output") + 1])
        audio_path = output_pattern.with_name(
            output_pattern.name.replace("%(ext)s", "m4a")
        )
        audio_path.write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout=f"{audio_path}\n", stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)

    path = YtDlpClient().download_transcription_audio({"id": "JpSHyEIZ_bo"}, tmp_path)

    assert path == tmp_path / "audio.m4a"
    assert path.read_bytes() == b"audio"
    assert commands[0][commands[0].index("--format") + 1] == STT_AUDIO_FORMAT_SELECTOR


def test_download_transcription_audio_fails_when_nothing_was_saved(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "youtube_toolkit.youtube.client.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="saved no audio"):
        YtDlpClient().download_transcription_audio({"id": "JpSHyEIZ_bo"}, tmp_path)


def test_channel_videos_requests_a_flat_listing_capped_at_last(monkeypatch) -> None:
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
        last=5,
    ) == {"entries": []}
    assert "--flat-playlist" in calls["command"]
    assert calls["command"][calls["command"].index("--playlist-end") + 1] == "5"
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
    first_path = next((artifact_root() / "records" / namespace).glob("*.json"))
    first_copy = first_path.read_bytes()
    assert retrieve(client)["generation"] == 2
    assert len(calls) == 2
    paths = list((artifact_root() / "records" / namespace).glob("*.json"))
    assert len(paths) == 2
    assert all(identifier in path.name for path in paths)
    assert first_path.read_bytes() == first_copy
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert sorted(entry["payload"]["generation"] for entry in entries) == [1, 2]
    assert all(
        entry["request"] and entry["fetched_at"].endswith("+00:00") for entry in entries
    )


def test_record_names_sanitize_identifiers_and_preserve_same_time_copies(
    monkeypatch, tmp_path
) -> None:
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
        artifacts.write_record(
            "searches",
            "../agents / tools",
            {"query": "agents / tools"},
            {"generation": generation},
        )
        for generation in (1, 2)
    ]
    assert paths[0] != paths[1]
    assert all(
        path.parent == tmp_path / ".youtube-toolkit/records/searches" for path in paths
    )
    assert (tmp_path / ".youtube-toolkit/.gitignore").read_text() == "*\n"
    assert all(path.name.startswith("2026-10-08T143012.123456Z-") for path in paths)
    assert all("agents" in path.name and "tools" in path.name for path in paths)
    entries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [entry["payload"]["generation"] for entry in entries] == [1, 2]
    assert all(entry["request"] == {"query": "agents / tools"} for entry in entries)
    assert all(
        entry["fetched_at"] == "2026-10-08T14:30:12.123456+00:00" for entry in entries
    )


@pytest.mark.parametrize("timed_out", [False, True])
def test_failed_audio_download_reports_why(monkeypatch, tmp_path, timed_out) -> None:
    def fail(command, **kwargs):
        assert kwargs["timeout"] == 600
        if timed_out:
            raise subprocess.TimeoutExpired(command, timeout=600)
        return SimpleNamespace(returncode=1, stderr="download failed")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fail)
    with pytest.raises(
        RuntimeError, match="timed out" if timed_out else "download failed"
    ):
        YtDlpClient().download_transcription_audio({"id": "JpSHyEIZ_bo"}, tmp_path)


@pytest.mark.parametrize(
    ("refusals", "expected_calls", "expected_delays"),
    [(3, 4, [5.0, 10.0, 15.0]), (4, 4, [5.0, 10.0, 15.0])],
)
def test_refused_downloads_are_rerun_after_a_pause(
    monkeypatch, tmp_path, refusals, expected_calls, expected_delays
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []
    delays: list[float] = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) <= refusals:
            return SimpleNamespace(
                returncode=1,
                stderr="ERROR: unable to download video data: HTTP Error 403: Forbidden",
            )
        audio_path = tmp_path / "audio.m4a"
        audio_path.write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout=f"{audio_path}\n", stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", run)
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)

    if refusals < expected_calls:
        assert (
            YtDlpClient()
            .download_transcription_audio({"id": "JpSHyEIZ_bo"}, tmp_path)
            .is_file()
        )
    else:
        with pytest.raises(RuntimeError, match="HTTP Error 403"):
            YtDlpClient().download_transcription_audio({"id": "JpSHyEIZ_bo"}, tmp_path)
    assert len(calls) == expected_calls
    assert delays == expected_delays


def test_comment_search_returns_matches_and_records_the_whole_sample(
    monkeypatch, tmp_path
) -> None:
    import json

    from youtube_toolkit.youtube.comments import search_comments

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        YtDlpClient,
        "comments",
        lambda self, _ref, *, limit: {
            "id": "JpSHyEIZ_bo",
            "title": "TITLE",
            "comments": [
                {"id": "a", "text": "Great agents demo", "like_count": 1},
                {"id": "b", "text": "First", "like_count": 9},
                {"id": "c", "text": "AGENTS everywhere", "like_count": 5},
            ],
        },
    )

    result = search_comments("JpSHyEIZ_bo", " Agents ")

    assert result == {
        "title": "TITLE",
        "url": "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        "query": " Agents ",
        "sampled": 3,
        "comments": [
            {"text": "Great agents demo", "like_count": 1},
            {"text": "AGENTS everywhere", "like_count": 5},
        ],
    }
    (record,) = (tmp_path / ".youtube-toolkit/records/comments").glob("*.json")
    sample = json.loads(record.read_text(encoding="utf-8"))["comments"]
    assert [comment["text"] for comment in sample] == [
        "Great agents demo",
        "First",
        "AGENTS everywhere",
    ]
