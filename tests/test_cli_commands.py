"""CLI command handler tests."""

import argparse
import json

import pytest

from youtube_toolkit import cli
from youtube_toolkit.cli import commands, main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.storage import get_connection, upsert_watchlist_channel


def test_doctor_returns_runtime_and_database_path(
    monkeypatch,
    tmp_path,
) -> None:
    """Doctor reports the effective DB path without opening storage."""
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    payload = commands.run_doctor(argparse.Namespace())

    assert isinstance(payload, dict)
    assert str(payload["yt_dlp_version"])
    assert payload["database"] == str(storage_home / "youtube.sqlite3")
    assert not storage_home.exists()


def test_main_emits_command_payload_as_json(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    assert main(["doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert str(payload["yt_dlp_version"])
    assert payload["database"] == str(storage_home / "youtube.sqlite3")


@pytest.mark.parametrize("command", ["audio", "transcribe-audio", "inspect", "transcript", "channel-videos"])
def test_removed_or_internal_commands_are_not_public(command: str) -> None:
    """Internal or renamed commands are not public CLI commands."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, "JpSHyEIZ_bo"])


def test_video_and_channel_command_groups_parse_public_actions() -> None:
    metadata_args = build_parser().parse_args(["video", "metadata", "JpSHyEIZ_bo"])
    transcript_args = build_parser().parse_args(["video", "transcript", "JpSHyEIZ_bo"])
    channel_args = build_parser().parse_args(["channel", "videos", "@OpenAI", "--last", "5"])
    search_args = build_parser().parse_args(["search", "langgraph tutorial"])

    assert metadata_args.video_command == "metadata"
    assert transcript_args.video_command == "transcript"
    assert channel_args.channel_command == "videos"
    assert channel_args.channel_ref == "@OpenAI"
    assert channel_args.last == 5
    assert search_args.command == "search"


def test_scheduler_job_command_is_hidden_from_watchlist_help(capsys) -> None:
    """Launchd can call scheduled-sync, but users see schedule management instead."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["watchlist", "--help"])

    output = capsys.readouterr().out
    assert "scheduled-sync" not in output
    assert "schedule" in output


def test_scheduler_job_command_stays_parseable() -> None:
    """The generated LaunchAgent still needs the internal job entrypoint."""
    args = build_parser().parse_args(["watchlist", "scheduled-sync", "--last", "5"])

    assert args.watchlist_command == "scheduled-sync"
    assert args.last == 5
    assert args.days == 3


@pytest.mark.parametrize(
    ("argv", "command_attr"),
    [
        (["watchlist", "add", "@OpenAI", "UC123"], "watchlist_command"),
        (["watchlist", "remove", "@OpenAI", "UC123"], "watchlist_command"),
        (["blacklist", "add", "@OpenAI", "UC123"], "blacklist_command"),
        (["blacklist", "remove", "@OpenAI", "UC123"], "blacklist_command"),
    ],
)
def test_channel_membership_commands_accept_multiple_refs(
    argv: list[str],
    command_attr: str,
) -> None:
    """Channel membership commands accept one or more URL, handle, or ID refs."""
    args = build_parser().parse_args(argv)

    assert getattr(args, command_attr) == argv[1]
    assert args.channel_refs == ["@OpenAI", "UC123"]


def test_watchlist_list_returns_stored_channel_refs(
    monkeypatch,
    tmp_path,
) -> None:
    """List reads local watchlist membership without fetching YouTube."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )

    payload = commands.WatchlistCommands.list(argparse.Namespace())

    assert payload == ["@OpenAI"]


def test_emit_json_truncates_long_text_fields(capsys, monkeypatch) -> None:
    """Large text fields are capped without cutting the whole JSON payload."""
    monkeypatch.setattr(cli, "MAX_STDOUT_TEXT_FIELD_CHARS", 20)

    cli.emit_json({"title": "y" * 100, "transcript": "x" * 100})

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "title": "y" * 100,
        "transcript": "x" * 20,
    }


def test_watchlist_new_accepts_repeated_channel_filters() -> None:
    """New-video consumption can be scoped to multiple watched channels."""
    args = build_parser().parse_args(
        [
            "watchlist",
            "new",
            "--channel",
            "@OpenAI",
            "--channel",
            "UC123",
        ]
    )

    assert args.channel_refs == ["@OpenAI", "UC123"]
