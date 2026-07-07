"""Opinionated macOS launchd management for watchlist scheduled sync."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import plistlib
import platform
import subprocess
import sys
from typing import Any, Literal, NotRequired, TypedDict

from .schemas import WatchlistSyncResult
from .storage import resolve_storage_dir
from .youtube.channels import ChannelOptions, sync_watchlist

DEFAULT_WATCHLIST_SCHEDULE_LABEL = "com.teron.youtube-toolkit.watchlist"
DEFAULT_WATCHLIST_SCHEDULE_MINUTES = 720
DEFAULT_WATCHLIST_SCHEDULE_LAST = None
DEFAULT_WATCHLIST_SCHEDULE_DAYS = 3
WATCHLIST_SYNC_LOCK_FILENAME = "watchlist-sync.lock"
LAUNCHCTL_PATH = "/bin/launchctl"


@dataclass(frozen=True, slots=True)
class WatchlistScheduleOptions:
    """Configuration for the launchd job that runs watchlist scheduled-sync."""

    minutes: int = DEFAULT_WATCHLIST_SCHEDULE_MINUTES
    last: int | None = DEFAULT_WATCHLIST_SCHEDULE_LAST
    days: int | None = DEFAULT_WATCHLIST_SCHEDULE_DAYS
    label: str = DEFAULT_WATCHLIST_SCHEDULE_LABEL
    load: bool = True


class WatchlistScheduleInstallResult(TypedDict):
    """Result contract for installing or loading the macOS watchlist LaunchAgent."""

    label: str
    plist_path: str
    loaded: bool
    load_error: NotRequired[str]


class WatchlistScheduleUninstallResult(TypedDict):
    """Result contract for unloading and removing the macOS watchlist LaunchAgent."""

    label: str
    plist_path: str
    removed: bool
    unloaded: bool


class WatchlistScheduleStatusResult(TypedDict):
    """Status contract combining LaunchAgent file presence and launchd runtime state."""

    label: str
    plist_path: str
    ok: bool
    status: Literal["not_installed", "not_loaded", "running", "idle", "failed"]
    detail: str


class ScheduledSyncResult(TypedDict):
    """Result contract for one lock-protected scheduled watchlist sync attempt."""

    status: Literal["skipped", "synced"]
    reason: NotRequired[str]
    sync: NotRequired[WatchlistSyncResult]


def install_watchlist_schedule(
    *,
    options: WatchlistScheduleOptions | None = None,
) -> WatchlistScheduleInstallResult:
    """Write the LaunchAgent plist and optionally load it into the current GUI session."""
    ensure_launchd_supported()
    options = options or WatchlistScheduleOptions()
    validate_schedule_options(options)

    plist_path = launch_agent_path(options.label)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_payload = build_watchlist_schedule_plist(options)
    with plist_path.open("wb") as plist_file:
        plistlib.dump(plist_payload, plist_file, sort_keys=True)

    loaded = False
    load_error = None
    if options.load:
        unload_launch_agent(plist_path, check=False)
        load_result = load_launch_agent(plist_path, check=False)
        loaded = load_result.returncode == 0
        if not loaded:
            load_error = load_result.stderr.strip() or load_result.stdout.strip() or "launchctl bootstrap failed"

    result: WatchlistScheduleInstallResult = {
        "label": options.label,
        "plist_path": str(plist_path),
        "loaded": loaded,
    }
    if load_error is not None:
        result["load_error"] = load_error
    return result


def uninstall_watchlist_schedule(
    *,
    label: str = DEFAULT_WATCHLIST_SCHEDULE_LABEL,
) -> WatchlistScheduleUninstallResult:
    """Unload and remove the LaunchAgent plist if it exists."""
    ensure_launchd_supported()
    plist_path = launch_agent_path(label)
    existed = plist_path.exists()
    unload_result = unload_launch_agent(plist_path, check=False) if existed else None
    if existed:
        plist_path.unlink()
    return {
        "label": label,
        "plist_path": str(plist_path),
        "removed": existed,
        "unloaded": unload_result is not None and unload_result.returncode == 0,
    }


def watchlist_schedule_status(
    *,
    label: str = DEFAULT_WATCHLIST_SCHEDULE_LABEL,
) -> WatchlistScheduleStatusResult:
    """Return file and launchd runtime status for the watchlist LaunchAgent."""
    ensure_launchd_supported()
    plist_path = launch_agent_path(label)
    if not plist_path.exists():
        return {
            "label": label,
            "plist_path": str(plist_path),
            "ok": False,
            "status": "not_installed",
            "detail": "LaunchAgent plist is missing",
        }

    print_result = print_launch_agent(label, check=False)
    if print_result.returncode != 0:
        detail = print_result.stderr.strip() or print_result.stdout.strip() or "LaunchAgent is not loaded"
        return {
            "label": label,
            "plist_path": str(plist_path),
            "ok": False,
            "status": "not_loaded",
            "detail": detail,
        }

    launchd_status = parse_launchctl_print(print_result.stdout)
    launchd_state = launchd_status.get("launchd_state")
    last_exit_code = launchd_status.get("last_exit_code")
    interval_seconds = launchd_status.get("interval_seconds")
    if launchd_state == "running":
        return {
            "label": label,
            "plist_path": str(plist_path),
            "ok": True,
            "status": "running",
            "detail": "Scheduled sync is running now",
        }
    if isinstance(last_exit_code, int) and last_exit_code != 0:
        return {
            "label": label,
            "plist_path": str(plist_path),
            "ok": False,
            "status": "failed",
            "detail": f"Last scheduled sync exited with code {last_exit_code}",
        }
    detail = "LaunchAgent is loaded and idle"
    if isinstance(interval_seconds, int):
        detail = f"{detail}; runs every {interval_seconds} seconds"
    return {
        "label": label,
        "plist_path": str(plist_path),
        "ok": True,
        "status": "idle",
        "detail": detail,
    }


def scheduled_sync_watchlist(
    *,
    options: ChannelOptions,
) -> ScheduledSyncResult:
    """Run the scheduler entrypoint once, skipping cleanly when another run holds the lock."""
    storage_dir = resolve_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    lock_path = storage_dir / WATCHLIST_SYNC_LOCK_FILENAME
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            return {
                "status": "skipped",
                "reason": "watchlist sync already running",
            }

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "started_at": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            )
        )
        lock_file.flush()
        return {
            "status": "synced",
            "sync": sync_watchlist(options=options),
        }


def build_watchlist_schedule_plist(
    options: WatchlistScheduleOptions,
) -> dict[str, Any]:
    """Build the LaunchAgent plist payload owned by the scheduler installer."""
    validate_schedule_options(options)
    storage_home = resolve_storage_dir()
    program_arguments = [
        sys.executable,
        "-m",
        "youtube_toolkit.cli",
        "watchlist",
        "scheduled-sync",
    ]
    if options.last is not None:
        program_arguments.extend(["--last", str(options.last)])
    if options.days is not None:
        program_arguments.extend(["--days", str(options.days)])

    return {
        "EnvironmentVariables": {
            "YOUTUBE_TOOLKIT_STORAGE_HOME": str(storage_home),
        },
        "Label": options.label,
        "ProgramArguments": program_arguments,
        "RunAtLoad": True,
        "StandardErrorPath": str(storage_home / "watchlist-scheduled-sync.err.log"),
        "StandardOutPath": str(storage_home / "watchlist-scheduled-sync.out.log"),
        "StartInterval": options.minutes * 60,
        "WorkingDirectory": str(project_root()),
    }


def validate_schedule_options(options: WatchlistScheduleOptions) -> None:
    """Validate scheduler interval, fetch limits, and launchd label before file writes."""
    if options.minutes <= 0:
        raise ValueError("--minutes must be greater than 0")
    if options.last is not None and options.last <= 0:
        raise ValueError("--last must be greater than 0")
    if options.days is not None and options.days <= 0:
        raise ValueError("--days must be greater than 0")
    if not options.label.strip():
        raise ValueError("--label must be non-empty")


def ensure_launchd_supported() -> None:
    """Raise when launchd management is invoked outside macOS."""
    if platform.system() != "Darwin":
        raise RuntimeError("watchlist schedule management uses macOS launchd")


def launch_agent_path(label: str) -> Path:
    """Return the user LaunchAgents plist path for a scheduler label."""
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def project_root() -> Path:
    """Return the repository root used as the scheduled command working directory."""
    return Path(__file__).resolve().parents[1]


def launchctl_domain() -> str:
    """Return the current user's GUI launchctl domain."""
    return f"gui/{os.getuid()}"


def load_launch_agent(
    plist_path: Path,
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Bootstrap a LaunchAgent plist into the current user GUI domain."""
    return run_launchctl(
        ["bootstrap", launchctl_domain(), str(plist_path)],
        check=check,
    )


def unload_launch_agent(
    plist_path: Path,
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Boot out a LaunchAgent plist from the current user GUI domain."""
    return run_launchctl(
        ["bootout", launchctl_domain(), str(plist_path)],
        check=check,
    )


def print_launch_agent(
    label: str,
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Print launchd state for the scheduler label in the current user GUI domain."""
    return run_launchctl(
        ["print", f"{launchctl_domain()}/{label}"],
        check=check,
    )


def parse_launchctl_print(output: str) -> dict[str, int | str]:
    """Extract the small scheduler status subset from launchctl print output."""
    status: dict[str, int | str] = {}
    for line in output.splitlines():
        key, separator, value = line.strip().partition(" = ")
        if not separator:
            continue
        if key == "state" and "launchd_state" not in status:
            status["launchd_state"] = value
        elif key == "last exit code" and value.lstrip("-").isdigit():
            status["last_exit_code"] = int(value)
        elif key == "run interval":
            interval_value = value.removesuffix(" seconds")
            if interval_value.isdigit():
                status["interval_seconds"] = int(interval_value)
    return status


def run_launchctl(
    arguments: list[str],
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Run launchctl with fixed binary path and caller-provided subcommand arguments."""
    return subprocess.run(  # noqa: S603 - launchctl subcommands are assembled by this module from fixed operations.
        [LAUNCHCTL_PATH, *arguments],
        capture_output=True,
        text=True,
        check=check,
    )
