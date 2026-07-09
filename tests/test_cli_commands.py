"""CLI command handler tests."""

import argparse
import json
import stat

import pytest

from youtube_toolkit import cli
from youtube_toolkit.cli import commands, main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.search import ChannelTarget, SearchRequest
from youtube_toolkit.storage import get_connection, upsert_video, upsert_watchlist_channel

MODEL_CONFIG_ENV_KEYS = (
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_AUTO_INDEX",
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
                "model": "text-embedding-3-small",
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
    assert payload["database"] == str(storage_home / "youtube.lancedb")
    assert payload["database_schema"] == {
        "path": str(storage_home / "youtube.lancedb"),
        "ok": False,
        "status": "missing",
        "detail": "LanceDB store is missing",
        "repair": "Run youtube setup database to create a fresh LanceDB store",
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
                "model": "text-embedding-3-small",
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
    assert f"  database   {storage_home / 'youtube.lancedb'}" in output
    assert "  schema     missing - LanceDB store is missing" in output
    assert "  schedule   not_installed - LaunchAgent plist is missing" in output
    assert "  embedding  missing (text-embedding-3-small) - set EMBEDDING_API_KEY" in output
    assert "  llm        missing - set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL" in output


def test_main_emits_model_setup_as_human_report(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    assert (
        main(
            [
                "setup",
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
                "--no-auto-embeddings",
                "--no-verify",
            ]
        )
        == 0
    )

    config_path = home / "Library" / "Application Support" / "youtube-toolkit" / "config.env"
    output = capsys.readouterr().out
    assert "Model Setup" in output
    assert f"  config     {config_path}" in output
    assert "  embedding  configured (embedding-model)" in output
    assert "  llm        configured (summary-model)" in output
    assert "Settings" in output
    assert "  auto index off" in output
    assert "Verification" in output
    assert "  embedding  skipped" in output
    assert "  llm        skipped" in output
    assert config_path.exists()


def test_interactive_setup_uses_section_menu(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    home = tmp_path / "home"
    keys = iter(
        [
            "down",
            "down",
            "enter",
            "enter",
            "down",
            "enter",
            "down",
            "down",
            "enter",
        ]
    )
    monkeypatch.setenv("HOME", str(home))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(commands, "setup_interactive_terminal", lambda: True)
    monkeypatch.setattr(commands, "read_setup_key", lambda: next(keys))

    assert main(["setup", "--no-verify"]) == 0

    config_path = home / "Library" / "Application Support" / "youtube-toolkit" / "config.env"
    output = capsys.readouterr().out
    assert "Setup" in output
    assert "Use Up/Down to move, Enter to select, q to quit." in output
    assert "Use Up/Down to move, Enter to select, q to go back." in output
    assert "> Settings" in output
    assert "> Auto-index transcript embeddings on" in output
    assert "> Auto-index transcript embeddings off" in output
    assert "  auto index off" in output
    assert "EMBEDDING_AUTO_INDEX=false\n" in config_path.read_text(encoding="utf-8")


def test_model_api_health_reports_key_presence(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "EMBEDDING_API_KEY": "embedding-key",
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
    index_args = build_parser().parse_args(["index", "JpSHyEIZ_bo"])
    setup_args = build_parser().parse_args(["setup", "--llm-model", "summary-model", "--no-auto-embeddings", "--verify"])

    assert metadata_args.video_command == "metadata"
    assert transcript_args.video_command == "transcript"
    assert channel_args.channel_command == "videos"
    assert channel_args.channel_ref == "@OpenAI"
    assert channel_args.last == 5
    assert search_args.command == "search"
    assert search_args.source is None
    assert search_args.mode == "native"
    assert search_args.return_type is None
    assert search_args.channels == []
    assert search_args.extra_queries == []
    assert search_args.limit is None
    assert search_args.min_views == commands.DEFAULT_SEARCH_MIN_VIEWS
    assert search_args.min_duration == commands.DEFAULT_SEARCH_MIN_DURATION
    assert search_args.max_duration == commands.DEFAULT_SEARCH_MAX_DURATION
    assert search_args.uploaded_after is None
    assert search_args.uploaded_within_days is None
    assert index_args.command == "index"
    assert index_args.video_refs == ["JpSHyEIZ_bo"]
    assert index_args.include_without_transcript is False
    assert setup_args.command == "setup"
    assert setup_args.llm_model == "summary-model"
    assert setup_args.auto_embeddings is False
    assert setup_args.verify is True


def test_parser_normalizes_string_arguments_before_handlers() -> None:
    """CLI string hygiene applies to every parsed argument value before command handlers run."""
    video_args = build_parser().parse_args(
        [
            "video",
            "transcript",
            "'https://www.youtube.com/watch?v=JpSHyEIZ_bo'",
        ]
    )
    search_args = build_parser().parse_args(
        [
            "search",
            "'agent memory'",
            "--extra-query",
            "'langgraph'",
            "--source",
            "'channel'",
            "--mode",
            "'semantic'",
            "--return",
            "'video'",
            "--channels",
            "'@OpenAI'",
            "--include-channel",
            "'OpenAI'",
            "--exclude-title",
            "'shorts'",
            "--sort",
            "'popularity'",
        ]
    )
    watchlist_args = build_parser().parse_args(["watchlist", "add", "'@OpenAI'", '"UC123"'])
    setup_args = build_parser().parse_args(["setup", "--llm-model", "'summary-model'"])
    unmatched_quote_args = build_parser().parse_args(["search", "'earnings"])

    assert video_args.video_ref == "https://www.youtube.com/watch?v=JpSHyEIZ_bo"
    assert search_args.query == "agent memory"
    assert search_args.extra_queries == ["langgraph"]
    assert search_args.source == ["channel"]
    assert search_args.mode == "semantic"
    assert search_args.return_type == "video"
    assert search_args.channels == ["@OpenAI"]
    assert search_args.include_channel == ["OpenAI"]
    assert search_args.exclude_title == ["shorts"]
    assert search_args.sort == "popularity"
    assert watchlist_args.channel_refs == ["@OpenAI", "UC123"]
    assert setup_args.llm_model == "summary-model"
    assert unmatched_quote_args.query == "'earnings"


@pytest.mark.parametrize(
    ("args", "usage"),
    [
        (["video", "transcript", "--help"], "usage: youtube video transcript [-h] url"),
        (["video", "metadata", "--help"], "usage: youtube video metadata [-h] url"),
        (["index", "--help"], "usage: youtube index [-h] [--include-without-transcript] [url ...]"),
    ],
)
def test_video_reference_help_names_positionals_url(
    args: list[str],
    usage: str,
    capsys,
) -> None:
    """Video reference help shows the shell-facing URL placeholder instead of internal destination names."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)

    output = capsys.readouterr().out
    assert usage in output
    assert "url" in output
    assert "YouTube video URL or ID" in output
    assert "video_ref" not in output


def test_search_command_group_parses_channel_source() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "--source",
            "channel",
            "--channels",
            "@OpenAI",
            "--channels",
            "UC123",
            "--limit",
            "2",
            "--sort",
            "popularity",
        ]
    )

    assert search_args.command == "search"
    assert search_args.source == ["channel"]
    assert search_args.channels == ["@OpenAI", "UC123"]
    assert search_args.limit == 2
    assert search_args.sort == "popularity"


def test_search_command_group_allows_channel_source_without_query() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "--source",
            "channel",
            "--channels",
            "@OpenAI",
        ]
    )

    assert search_args.query == ""
    assert search_args.source == ["channel"]
    assert search_args.channels == ["@OpenAI"]


def test_search_command_group_parses_extra_queries() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "--extra-query",
            "agent skills",
            "--extra-query",
            "tool use",
        ]
    )

    assert search_args.query == "agents"
    assert search_args.extra_queries == [
        "agent skills",
        "tool use",
    ]


def test_search_command_group_accepts_plural_filter_aliases() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "--include-channels",
            "OpenAI",
            "--include-channels",
            "Anthropic",
            "--exclude-channels",
            "shorts",
            "--exclude-channels",
            "clips",
            "--include-titles",
            "agent",
            "--include-titles",
            "tools",
            "--exclude-titles",
            "reaction",
            "--exclude-titles",
            "drama",
        ]
    )

    assert search_args.include_channel == ["OpenAI", "Anthropic"]
    assert search_args.exclude_channel == ["shorts", "clips"]
    assert search_args.include_title == ["agent", "tools"]
    assert search_args.exclude_title == ["reaction", "drama"]


def test_search_command_builds_default_youtube_request(monkeypatch) -> None:
    requests = []

    def fake_run_search_request(request: SearchRequest) -> list[dict]:
        requests.append(request)
        return [{"id": "aaaaaaaaaaa"}]

    monkeypatch.setattr(commands, "run_search_request", fake_run_search_request)
    payload = commands.SearchCommands.search(
        argparse.Namespace(
            query="agents",
            extra_queries=["agent skills", "tool use"],
            source=None,
            mode="native",
            return_type=None,
            channels=[],
            limit=None,
            sort="relevance",
            min_views=5000,
            min_duration="3:00",
            max_duration="8:00:00",
            uploaded_after="2026-07-01",
            uploaded_within_days=7,
            include_channel=["OpenAI"],
            exclude_channel=[],
            include_title=[],
            exclude_title=["shorts"],
        )
    )

    assert payload == [{"id": "aaaaaaaaaaa"}]
    assert requests == [
        SearchRequest(
            query="agents",
            extra_queries=("agent skills", "tool use"),
            sources=("youtube",),
            mode="native",
            return_type=None,
            limit=commands.DEFAULT_YOUTUBE_SEARCH_LIMIT,
            sort="relevance",
            min_views=5000,
            min_duration="3:00",
            max_duration="8:00:00",
            uploaded_after="2026-07-01",
            uploaded_within_days=7,
            include_channel=("OpenAI",),
            exclude_title=("shorts",),
        )
    ]


def test_search_command_builds_channel_request(monkeypatch) -> None:
    requests = []

    def fake_run_search_request(request: SearchRequest) -> list[dict]:
        requests.append(request)
        return [{"id": "aaaaaaaaaaa"}]

    monkeypatch.setattr(commands, "run_search_request", fake_run_search_request)
    payload = commands.SearchCommands.search(
        argparse.Namespace(
            query="agents",
            source=["channel"],
            mode="native",
            return_type=None,
            channels=["@OpenAI", "UC123"],
            limit=None,
            sort="popularity",
            min_views=5000,
            min_duration="3:00",
            max_duration="8:00:00",
            include_channel=[],
            exclude_channel=[],
            include_title=[],
            exclude_title=[],
        )
    )

    assert payload == [{"id": "aaaaaaaaaaa"}]
    assert requests == [
        SearchRequest(
            query="agents",
            sources=("channel",),
            mode="native",
            return_type=None,
            channels=(
                ChannelTarget(channel_handle="@OpenAI"),
                ChannelTarget(youtube_channel_id="UC123"),
            ),
            limit=commands.DEFAULT_CHANNEL_SEARCH_LIMIT,
            sort="popularity",
            min_views=5000,
            min_duration="3:00",
            max_duration="8:00:00",
        )
    ]


def test_search_command_requires_channels_for_channel_source() -> None:
    with pytest.raises(ValueError, match="at least one --channels"):
        commands.SearchCommands.search(
            argparse.Namespace(
                query="agents",
                source=["channel"],
                mode="native",
                return_type=None,
                channels=[],
                limit=2,
                sort="recency",
                min_views=5000,
                min_duration="3:00",
                max_duration="8:00:00",
                include_channel=[],
                exclude_channel=[],
                include_title=[],
                exclude_title=[],
            )
        )


def test_search_command_rejects_channels_option_without_channel_source() -> None:
    with pytest.raises(ValueError, match="--channels requires --source channel"):
        commands.SearchCommands.search(
            argparse.Namespace(
                query="agents",
                source=["youtube"],
                mode="native",
                return_type=None,
                channels=["@OpenAI"],
                limit=2,
                sort="relevance",
                min_views=5000,
                min_duration="3:00",
                max_duration="8:00:00",
                include_channel=[],
                exclude_channel=[],
                include_title=[],
                exclude_title=[],
            )
        )


def test_search_command_rejects_native_youtube_recency() -> None:
    with pytest.raises(ValueError, match="does not support --sort recency"):
        commands.SearchCommands.search(
            argparse.Namespace(
                query="agents",
                source=["youtube"],
                mode="native",
                return_type=None,
                channels=[],
                limit=2,
                sort="recency",
                min_views=5000,
                min_duration="3:00",
                max_duration="8:00:00",
                include_channel=[],
                exclude_channel=[],
                include_title=[],
                exclude_title=[],
            )
        )


def test_index_command_indexes_stored_transcript_videos(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    indexed_video_ids: list[str] = []

    def fake_index_video_embedding(connection, video_id: str):
        indexed_video_ids.append(video_id)
        return {
            "video_id": video_id,
            "dimensions": 2,
            "content_hash": "hash",
        }

    monkeypatch.setattr(commands, "index_video_embedding", fake_index_video_embedding)

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
            },
            transcript="晶片需求和 GPU 投资回升",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "No transcript",
            },
        )

    payload = commands.IndexCommands.videos(
        argparse.Namespace(
            video_refs=[],
            include_without_transcript=False,
        )
    )

    assert indexed_video_ids == ["JpSHyEIZ_bo"]
    assert payload == {
        "videos": 1,
        "indexed": [
            {
                "status": "indexed",
                "video_id": "JpSHyEIZ_bo",
                "dimensions": 2,
            }
        ],
        "skipped": [],
        "failed": [],
    }


def test_index_command_reports_missing_explicit_video(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    payload = commands.IndexCommands.videos(
        argparse.Namespace(
            video_refs=["JpSHyEIZ_bo"],
            include_without_transcript=False,
        )
    )

    assert payload == {
        "videos": 1,
        "indexed": [],
        "skipped": [
            {
                "status": "skipped",
                "video_id": "JpSHyEIZ_bo",
                "detail": "Video is not stored",
            }
        ],
        "failed": [],
    }


def test_main_emits_index_as_human_report(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    assert main(["index", "JpSHyEIZ_bo"]) == 0

    output = capsys.readouterr().out
    assert output == ("Embedding Index\n  videos   1\n  indexed  0\n  skipped  1\n  failed   0\n\nSkipped\n  JpSHyEIZ_bo  Video is not stored\n")


def test_render_index_summarizes_successes() -> None:
    assert cli.render_index(
        {
            "videos": 7,
            "indexed": [
                {
                    "status": "indexed",
                    "video_id": "JpSHyEIZ_bo",
                    "dimensions": 768,
                }
            ],
            "skipped": [],
            "failed": [],
        }
    ) == ("Embedding Index\n  videos   7\n  indexed  1 (768 dimensions)\n  skipped  0\n  failed   0")


def test_setup_writes_app_local_config(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    args = build_parser().parse_args(
        [
            "setup",
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
        "settings": {
            "auto_embeddings": True,
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
        "EMBEDDING_AUTO_INDEX=true\n"
        "LLM_API_KEY=llm-key\n"
        "LLM_BASE_URL=https://llm.example/v1/\n"
        "LLM_MODEL=summary-model\n"
    )
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_setup_can_verify_saved_config(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    for key in MODEL_CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    calls = {}

    def fake_call_llm_hello(config):
        calls["config"] = config
        return "Hello back"

    monkeypatch.setattr(commands, "embed_text", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(commands, "call_llm_hello", fake_call_llm_hello)

    args = build_parser().parse_args(
        [
            "setup",
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
