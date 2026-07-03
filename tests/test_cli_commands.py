"""CLI command handler tests."""

import argparse
import json
import stat

import pytest

from youtube_toolkit import cli
from youtube_toolkit.cli import commands, main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.storage import get_connection, upsert_watchlist_channel

MODEL_CONFIG_ENV_KEYS = (
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
)


def test_doctor_returns_runtime_and_database_path(
    monkeypatch,
    tmp_path,
) -> None:
    """Doctor reports the effective DB path without opening storage."""
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.setattr(
        commands,
        "watchlist_schedule_health",
        lambda: {
            "label": "com.teron.youtube-toolkit.watchlist",
            "ok": True,
            "status": "idle",
            "detail": "LaunchAgent is loaded and idle",
            "plist_path": str(storage_home / "com.teron.youtube-toolkit.watchlist.plist"),
        },
    )
    monkeypatch.setattr(
        commands,
        "model_api_health",
        lambda: {
            "embedding": {
                "ok": True,
                "model": "gemini-embedding-2",
                "status": "configured",
                "detail": "Embedding API key is configured",
            },
            "llm": {
                "ok": False,
                "model": "not configured",
                "status": "missing",
                "detail": "LLM API key is missing",
            },
        },
    )

    payload = commands.run_doctor(argparse.Namespace())

    assert isinstance(payload, dict)
    assert str(payload["yt_dlp_version"])
    assert payload["database"] == str(storage_home / "youtube.sqlite3")
    assert payload["database_schema"] == {
        "path": str(storage_home / "youtube.sqlite3"),
        "ok": False,
        "status": "missing",
        "detail": "Database file is missing",
        "repair": "Run youtube setup database to create a fresh schema",
    }
    assert payload["watchlist_schedule"] == {
        "label": "com.teron.youtube-toolkit.watchlist",
        "ok": True,
        "status": "idle",
        "detail": "LaunchAgent is loaded and idle",
        "plist_path": str(storage_home / "com.teron.youtube-toolkit.watchlist.plist"),
    }
    assert payload["model_apis"]["embedding"]["ok"] is True
    assert payload["model_apis"]["llm"]["ok"] is False
    assert not storage_home.exists()


def test_doctor_schedule_health_handles_unsupported_launchd(monkeypatch) -> None:
    """Doctor should report schedule health failures instead of failing itself."""

    def fail_status(*, label: str):
        raise RuntimeError(f"unsupported for {label}")

    monkeypatch.setattr(commands, "watchlist_schedule_status", fail_status)

    assert commands.watchlist_schedule_health(label="com.example.watch") == {
        "label": "com.example.watch",
        "ok": False,
        "status": "unsupported",
        "detail": "unsupported for com.example.watch",
    }


def test_main_emits_doctor_as_human_report(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.setattr(
        commands,
        "watchlist_schedule_health",
        lambda: {
            "label": "com.teron.youtube-toolkit.watchlist",
            "ok": False,
            "status": "not_installed",
            "detail": "LaunchAgent plist is missing",
            "plist_path": str(storage_home / "com.teron.youtube-toolkit.watchlist.plist"),
        },
    )
    monkeypatch.setattr(
        commands,
        "model_api_health",
        lambda: {
            "embedding": {
                "ok": False,
                "model": "gemini-embedding-2",
                "status": "missing",
                "detail": "Embedding API key is missing",
            },
            "llm": {
                "ok": False,
                "model": "not configured",
                "status": "missing",
                "detail": "LLM API key is missing",
            },
        },
    )

    assert main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "YouTube Toolkit" in output
    assert f"  database   {storage_home / 'youtube.sqlite3'}" in output
    assert "  schema     missing - Database file is missing" in output
    assert "  schedule   not_installed - LaunchAgent plist is missing" in output
    assert "  embedding  missing (gemini-embedding-2) - set EMBEDDING_API_KEY" in output
    assert "  llm        missing - set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL" in output


def test_model_api_health_reports_key_presence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "EMBEDDING_API_KEY": "gemini-key",
        "EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "EMBEDDING_MODEL": "embedding-model",
        "EMBEDDING_DIMENSIONS": "512",
        "LLM_API_KEY": "llm-key",
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_MODEL": "llm-model",
    }.items():
        monkeypatch.setenv(key, value)

    assert commands.model_api_health() == {
        "embedding": {
            "ok": True,
            "model": "embedding-model",
            "status": "configured",
            "detail": "Embedding API is configured",
        },
        "llm": {
            "ok": True,
            "model": "llm-model",
            "status": "configured",
            "detail": "LLM API is configured",
        },
    }


def test_model_api_health_reports_missing_llm_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "llm-key")

    assert commands.llm_api_health() == {
        "ok": False,
        "model": "not configured",
        "status": "missing",
        "detail": "LLM base URL, model missing",
    }


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
    setup_args = build_parser().parse_args(["setup", "models", "--llm-model", "summary-model", "--verify"])

    assert metadata_args.video_command == "metadata"
    assert transcript_args.video_command == "transcript"
    assert channel_args.channel_command == "videos"
    assert channel_args.channel_ref == "@OpenAI"
    assert channel_args.last == 5
    assert search_args.command == "search"
    assert setup_args.setup_command == "models"
    assert setup_args.llm_model == "summary-model"
    assert setup_args.verify is True


def test_setup_models_writes_app_local_config(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    args = build_parser().parse_args(
        [
            "setup",
            "models",
            "--embedding-api-key",
            "embedding-key",
            "--embedding-base-url",
            "https://embedding.example/v1/",
            "--embedding-model",
            "embedding-model",
            "--embedding-dimensions",
            "512",
            "--llm-api-key",
            "llm-key",
            "--llm-base-url",
            "https://llm.example/v1/",
            "--llm-model",
            "summary-model",
        ]
    )

    payload = args.handler(args)
    config_path = home / "Library" / "Application Support" / "youtube-toolkit" / "config.env"

    assert payload == {
        "config": str(config_path),
        "model_apis": {
            "embedding": {
                "ok": True,
                "model": "embedding-model",
                "status": "configured",
                "detail": "Embedding API is configured",
            },
            "llm": {
                "ok": True,
                "model": "summary-model",
                "status": "configured",
                "detail": "LLM API is configured",
            },
        },
        "verification": {
            "embedding": {
                "ok": False,
                "status": "skipped",
                "detail": "Verification skipped",
            },
            "llm": {
                "ok": False,
                "status": "skipped",
                "detail": "Verification skipped",
            },
        },
    }
    assert config_path.read_text(encoding="utf-8") == (
        "EMBEDDING_API_KEY=embedding-key\n"
        "EMBEDDING_BASE_URL=https://embedding.example/v1/\n"
        "EMBEDDING_MODEL=embedding-model\n"
        "EMBEDDING_DIMENSIONS=512\n"
        "LLM_API_KEY=llm-key\n"
        "LLM_BASE_URL=https://llm.example/v1/\n"
        "LLM_MODEL=summary-model\n"
    )
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_setup_models_can_verify_saved_config(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    calls = {}

    def fake_call_llm_hello(config):
        calls["config"] = config
        return "Hello back"

    monkeypatch.setattr(commands, "embed_text_with_gemini", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(commands, "call_llm_hello", fake_call_llm_hello)

    args = build_parser().parse_args(
        [
            "setup",
            "models",
            "--embedding-api-key",
            "embedding-key",
            "--embedding-base-url",
            "https://embedding.example/v1/",
            "--embedding-model",
            "embedding-model",
            "--embedding-dimensions",
            "512",
            "--llm-api-key",
            "llm-key",
            "--llm-base-url",
            "https://llm.example/v1/",
            "--llm-model",
            "summary-model",
            "--verify",
        ]
    )

    payload = args.handler(args)

    assert payload["verification"] == {
        "embedding": {
            "ok": True,
            "status": "passed",
            "detail": "Embedding API returned 3 dimensions",
        },
        "llm": {
            "ok": True,
            "status": "passed",
            "detail": "LLM API returned 10 characters",
        },
    }
    assert calls["config"].model == "summary-model"
    assert calls["config"].api_key == "llm-key"
    assert calls["config"].base_url == "https://llm.example/v1"


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


def test_watchlist_recent_accepts_optional_channel_filters() -> None:
    """Recent watchlist videos can be scoped to multiple watched channels."""
    default_args = build_parser().parse_args(["watchlist", "recent"])
    args = build_parser().parse_args(
        [
            "watchlist",
            "recent",
            "@OpenAI",
            "UC123",
        ]
    )

    assert default_args.channel_refs == []
    assert default_args.days == 3
    assert args.channel_refs == ["@OpenAI", "UC123"]
    assert args.days == 3


@pytest.mark.parametrize("command", ["new", "read"])
def test_old_watchlist_newness_commands_are_not_public(command: str) -> None:
    """Old subjective-newness command names are not kept as aliases."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["watchlist", command])
