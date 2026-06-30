"""Watchlist scheduled sync LaunchAgent management tests."""

import fcntl
import plistlib
import subprocess
import sys

from youtube_toolkit.youtube.channels import ChannelOptions
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.watchlist_schedule import (
    WatchlistScheduleOptions,
    build_watchlist_schedule_plist,
    install_watchlist_schedule,
    scheduled_sync_watchlist,
    uninstall_watchlist_schedule,
    watchlist_schedule_status,
)


def test_parser_accepts_watchlist_schedule_defaults() -> None:
    args = build_parser().parse_args(["watchlist", "schedule", "install"])

    assert args.watchlist_command == "schedule"
    assert args.schedule_command == "install"
    assert args.minutes == 30
    assert args.last == 50
    assert args.days == 2
    assert args.label == "com.teron.youtube-toolkit.watchlist"
    assert args.no_load is False


def test_build_watchlist_schedule_plist_targets_scheduled_sync(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    plist = build_watchlist_schedule_plist(
        WatchlistScheduleOptions(
            minutes=15,
            last=25,
            days=3,
            load=False,
        )
    )

    assert plist["Label"] == "com.teron.youtube-toolkit.watchlist"
    assert plist["ProgramArguments"] == [
        sys.executable,
        "-m",
        "youtube_toolkit.cli",
        "watchlist",
        "scheduled-sync",
        "--last",
        "25",
        "--days",
        "3",
    ]
    assert plist["StartInterval"] == 900
    assert plist["RunAtLoad"] is True
    assert plist["EnvironmentVariables"] == {
        "YOUTUBE_TOOLKIT_STORAGE_HOME": str(tmp_path / "state"),
    }
    assert plist["StandardOutPath"] == str(tmp_path / "state" / "watchlist-scheduled-sync.out.log")
    assert plist["StandardErrorPath"] == str(tmp_path / "state" / "watchlist-scheduled-sync.err.log")


def test_install_watchlist_schedule_writes_plist_without_loading(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("youtube_toolkit.watchlist_schedule.platform.system", lambda: "Darwin")
    home = tmp_path / "home"
    storage_home = tmp_path / "state"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    result = install_watchlist_schedule(
        options=WatchlistScheduleOptions(
            minutes=10,
            last=20,
            days=None,
            label="com.example.youtube-watch",
            load=False,
        )
    )

    plist_path = home / "Library" / "LaunchAgents" / "com.example.youtube-watch.plist"
    assert result["plist_path"] == str(plist_path)
    assert result["loaded"] is False
    assert result["load_error"] is None
    with plist_path.open("rb") as plist_file:
        plist = plistlib.load(plist_file)
    assert plist["ProgramArguments"] == [
        sys.executable,
        "-m",
        "youtube_toolkit.cli",
        "watchlist",
        "scheduled-sync",
        "--last",
        "20",
    ]


def test_status_reports_launchctl_state(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("youtube_toolkit.watchlist_schedule.platform.system", lambda: "Darwin")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    plist_path = home / "Library" / "LaunchAgents" / "com.example.youtube-watch.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("plist", encoding="utf-8")

    def fake_print_launch_agent(
        label: str,
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert label == "com.example.youtube-watch"
        assert check is False
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("youtube_toolkit.watchlist_schedule.print_launch_agent", fake_print_launch_agent)

    assert watchlist_schedule_status(label="com.example.youtube-watch") == {
        "label": "com.example.youtube-watch",
        "plist_path": str(plist_path),
        "installed": True,
        "loaded": True,
    }


def test_uninstall_removes_plist_and_unloads(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("youtube_toolkit.watchlist_schedule.platform.system", lambda: "Darwin")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    plist_path = home / "Library" / "LaunchAgents" / "com.example.youtube-watch.plist"
    plist_path.parent.mkdir(parents=True)
    plist_path.write_text("plist", encoding="utf-8")

    def fake_unload_launch_agent(
        path,
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert path == plist_path
        assert check is False
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("youtube_toolkit.watchlist_schedule.unload_launch_agent", fake_unload_launch_agent)

    assert uninstall_watchlist_schedule(label="com.example.youtube-watch") == {
        "label": "com.example.youtube-watch",
        "plist_path": str(plist_path),
        "removed": True,
        "unloaded": True,
    }
    assert not plist_path.exists()


def test_scheduled_sync_skips_when_lock_is_held(
    monkeypatch,
    tmp_path,
) -> None:
    """Scheduler wrapper should not stack a second sync run."""
    storage_home = tmp_path / "state"
    storage_home.mkdir()
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    lock_path = storage_home / "watchlist-sync.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def fail_sync(*_args, **_kwargs):
            raise AssertionError("sync should not run while the lock is held")

        monkeypatch.setattr("youtube_toolkit.watchlist_schedule.sync_watchlist", fail_sync)

        assert scheduled_sync_watchlist(options=ChannelOptions()) == {
            "status": "skipped",
            "reason": "watchlist sync already running",
        }
