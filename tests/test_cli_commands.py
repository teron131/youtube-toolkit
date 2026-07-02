"""CLI command handler tests."""

import argparse

from youtube_toolkit.cli.commands import run_doctor


def test_doctor_prints_runtime_and_database_path(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    """Doctor reports the effective DB path without opening storage."""
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    assert run_doctor(argparse.Namespace()) == 0

    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("yt-dlp import ok: ")
    assert output[1] == f"database: {storage_home / 'youtube.sqlite3'}"
    assert not storage_home.exists()
