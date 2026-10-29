"""CLI contracts for retrieval archives, exports, and native or semantic search."""

import argparse
import json
import os
from types import SimpleNamespace
import warnings

import pytest

from youtube_toolkit import config
from youtube_toolkit.cli import commands, health, output, setup
from youtube_toolkit.cli.__main__ import main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import TranscriptChunk, TranscriptDocument, VideoMetadata
from youtube_toolkit.search.request import ChannelTarget, SearchRequest


def test_parser_normalizes_string_arguments_before_handlers() -> None:
    video_args = build_parser().parse_args(
        [
            "transcript",
            "'https://www.youtube.com/watch?v=JpSHyEIZ_bo'",
        ]
    )
    search_args = build_parser().parse_args(
        [
            "search",
            "'agent memory'",
            "'langgraph'",
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
    unmatched_quote_args = build_parser().parse_args(["search", "'earnings"])

    assert video_args.video_ref == "https://www.youtube.com/watch?v=JpSHyEIZ_bo"
    assert search_args.queries == ["agent memory", "langgraph"]
    assert search_args.mode == "semantic"
    assert search_args.return_type == "video"
    assert search_args.channels == ["@OpenAI"]
    assert search_args.include_channel == ["OpenAI"]
    assert search_args.exclude_title == ["shorts"]
    assert search_args.sort == "popularity"
    assert unmatched_quote_args.queries == ["'earnings"]


def test_video_metadata_command_returns_normalized_video_facts(monkeypatch) -> None:
    class FakeClient:
        def metadata(self, video_ref: str) -> dict[str, object]:
            assert video_ref == "JpSHyEIZ_bo"
            return {
                "id": "JpSHyEIZ_bo",
                "title": "Example",
                "uploader": "Example Channel",
                "channel_id": "UC1234567890123456789012",
                "channel_handle": "@example",
                "upload_date": "20260716",
                "duration": 125.5,
                "view_count": 42_000,
                "like_count": 1_200,
                "description": "Video description",
                "thumbnail": "https://example.com/thumbnail.jpg",
                "chapters": [
                    {"title": "Introduction", "start_time": 0, "end_time": 60},
                    {"title": "Demo", "start_time": 60},
                ],
                "formats": [{"format_id": "provider-internal"}],
            }

    monkeypatch.setattr(commands, "YtDlpClient", FakeClient)

    assert commands.metadata(argparse.Namespace(video_ref="JpSHyEIZ_bo")) == {
        "video_id": "JpSHyEIZ_bo",
        "url": "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        "title": "Example",
        "channel": "Example Channel",
        "channel_id": "UC1234567890123456789012",
        "channel_handle": "@example",
        "upload_date": "2026-07-16",
        "duration_seconds": 125.5,
        "view_count": 42_000,
        "like_count": 1_200,
        "description": "Video description",
        "thumbnail": "https://example.com/thumbnail.jpg",
        "chapters": [
            {"title": "Introduction", "start_seconds": 0.0, "end_seconds": 60.0},
            {"title": "Demo", "start_seconds": 60.0, "end_seconds": None},
        ],
    }


@pytest.fixture
def comment_response(monkeypatch, tmp_path):
    """Replace only the yt-dlp process so comment tests exercise real parsing, filtering, ordering, and archives."""
    payload = {"id": "JpSHyEIZ_bo"}
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", run)
    return payload, calls


@pytest.mark.parametrize(
    ("arguments", "expected_ids", "limit"),
    [
        (["PRICING"], ["first", "second"], 100),
        (["pricing details", "--limit", "1000"], ["first"], 1000),
        (["[pricing]"], ["second"], 100),
        (["STRASSE"], ["second"], 100),
        ([], ["first", "second", "third"], 100),
        ([""], ["first", "second", "third"], 100),
        (["   "], ["first", "second", "third"], 100),
        (["unmatched"], [], 100),
    ],
)
def test_comments_search_filters_text_and_archives_the_full_sample(
    comment_response, tmp_path, capsys, arguments, expected_ids, limit
) -> None:
    payload, calls = comment_response
    payload.update(
        title="Example",
        comments=[
            {
                "id": "first",
                "parent": "root",
                "text": "Pricing details\nCosts $20.",
                "like_count": 42,
            },
            {
                "id": "second",
                "parent": "root",
                "text": "Straße [pricing] discussion",
                "is_pinned": True,
            },
            {"id": "third", "parent": "root", "author": "pricing", "text": None},
        ],
        formats=[{"format_id": "provider-internal"}],
    )

    assert main(["comments", "JpSHyEIZ_bo", *arguments]) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result == {
        "title": "Example",
        "url": "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        "comments": [
            {"text": comment.get("text") or "", "like_count": comment.get("like_count")}
            for comment in payload["comments"]
            if comment["id"] in expected_ids
        ],
    }
    assert captured.err == ""
    assert len(calls) == 1
    command = calls[0]
    assert "--skip-download" in command
    assert "--no-playlist" in command
    assert "--write-comments" in command
    assert command[command.index("--extractor-args") + 1] == (
        f"youtube:comment_sort=top;max_comments={limit},all,0,0,1"
    )
    archive_path = next((tmp_path / ".cache/youtube-toolkit/comments").glob("*.json"))
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archive == {
        "title": "Example",
        "url": "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        "comments": [
            {"text": "Pricing details\nCosts $20.", "like_count": 42},
            {"text": "Straße [pricing] discussion", "like_count": None},
            {"text": "", "like_count": None},
        ],
    }


@pytest.mark.parametrize("comments", [None, []])
def test_comments_search_returns_an_empty_array_when_none_are_available(
    comment_response, capsys, comments
) -> None:
    payload, _calls = comment_response
    payload["comments"] = comments

    assert main(["comments", "JpSHyEIZ_bo", "pricing"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["comments"] == []


@pytest.mark.parametrize(
    ("query", "expected_texts"),
    [
        ("", ["A", "D match", "B match", "E", "C match", "F match"]),
        ("match", ["B match", "D match", "C match", "F match"]),
    ],
)
def test_comments_interleaves_next_unseen_ranked_matches(
    comment_response, capsys, query, expected_texts
) -> None:
    comments = [
        {"id": "a", "text": "A", "like_count": 100},
        {"id": "b", "text": "B match", "like_count": 40},
        {"id": "c", "text": "C match", "like_count": 30},
        {"id": "d", "text": "D match", "like_count": 90},
        {"id": "e", "text": "E", "like_count": 80},
        {"id": "f", "text": "F match", "like_count": 70},
    ]
    payload, _calls = comment_response
    payload["comments"] = [*comments, comments[0]]

    assert main(["comments", "JpSHyEIZ_bo", query]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [comment["text"] for comment in result["comments"]] == expected_texts


def test_comments_preserves_like_ties_and_unknown_counts(
    comment_response, capsys
) -> None:
    """Two comments share their text on purpose: only a repeated ID counts as a duplicate."""
    payload, _calls = comment_response
    payload["comments"] = [
        {"id": "a", "text": "Same text", "like_count": 1},
        {"id": "b", "text": "B", "like_count": None},
        {"id": "c", "text": "C", "like_count": 0},
        {"id": "d", "text": "D", "like_count": 0},
        {"id": "e", "text": "Same text", "like_count": 0},
    ]
    assert main(["comments", "JpSHyEIZ_bo"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["comments"] == [
        {"text": "Same text", "like_count": 1},
        {"text": "C", "like_count": 0},
        {"text": "B", "like_count": None},
        {"text": "D", "like_count": 0},
        {"text": "Same text", "like_count": 0},
    ]


def test_comments_refetches_and_preserves_lean_archive_history(
    comment_response, tmp_path
) -> None:
    payload, calls = comment_response
    payload["comments"] = [{"id": "a", "text": "A", "like_count": 1}]
    assert main(["comments", "JpSHyEIZ_bo"]) == 0
    directory = tmp_path / ".cache/youtube-toolkit/comments"
    first_path = next(directory.glob("*.json"))
    first_copy = first_path.read_bytes()
    payload["comments"][0]["like_count"] = 2
    assert main(["comments", "JpSHyEIZ_bo"]) == 0
    assert len(calls) == 2
    assert first_path.read_bytes() == first_copy
    assert sorted(
        json.loads(path.read_text())["comments"][0]["like_count"]
        for path in directory.glob("*.json")
    ) == [1, 2]


@pytest.mark.parametrize("limit", ["0", "-1", "1001"])
def test_comments_rejects_out_of_range_fetch_limits_before_retrieval(
    comment_response, capsys, limit
) -> None:
    _payload, calls = comment_response

    assert main(["comments", "JpSHyEIZ_bo", "pricing", "--limit", limit]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--limit must be between 1 and 1000" in captured.err
    assert calls == []


def test_video_transcript_command_selects_plain_or_timestamped_output(
    monkeypatch,
) -> None:
    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo",
            url="https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        ),
        language="en",
        source="subtitles",
        chunks=[
            TranscriptChunk(start_seconds=0.0, end_seconds=2.5, text="First sentence."),
            TranscriptChunk(
                start_seconds=2.5, end_seconds=5.0, text="Second sentence."
            ),
        ],
    )
    monkeypatch.setattr(commands, "fetch_transcript", lambda _video_ref: document)

    assert (
        commands.transcript(
            argparse.Namespace(
                video_ref="JpSHyEIZ_bo",
                timestamps=False,
                export=False,
                force=False,
            )
        )
        == "First sentence. Second sentence."
    )
    assert commands.transcript(
        argparse.Namespace(
            video_ref="JpSHyEIZ_bo",
            timestamps=True,
            export=False,
            force=False,
        )
    ) == [
        {
            "start_seconds": 0.0,
            "end_seconds": 2.5,
            "text": "First sentence.",
        },
        {
            "start_seconds": 2.5,
            "end_seconds": 5.0,
            "text": "Second sentence.",
        },
    ]


def test_video_transcript_export_writes_metadata_and_plain_transcript(
    capsys,
    monkeypatch,
    tmp_path,
) -> None:
    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo",
            url="https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            title="What is DARKMATTER ?",
            channel="Example Channel",
            channel_id="UC1234567890123456789012",
            channel_handle="@example",
            upload_date="2026-07-16",
            duration_seconds=5025,
            view_count=120_000,
            like_count=4_500,
        ),
        language="en",
        source="subtitles",
        chunks=[
            TranscriptChunk(start_seconds=0.0, end_seconds=2.5, text="First sentence."),
            TranscriptChunk(
                start_seconds=62.0, end_seconds=65.0, text="Second sentence."
            ),
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        commands, "fetch_transcript", lambda _video_ref, **_options: document
    )

    assert main(["transcript", "JpSHyEIZ_bo", "--export"]) == 0

    export_path = tmp_path / "What_is_DARKMATTER_Example_Channel.txt"
    assert json.loads(capsys.readouterr().out) == {
        "video_id": "JpSHyEIZ_bo",
        "path": str(export_path),
    }
    assert export_path.read_text(encoding="utf-8") == (
        "Title: What is DARKMATTER ?\n"
        "Author: Example Channel\n"
        "Channel: @example\n"
        "Published: 2026-07-16\n"
        "Duration: 01:23:45\n"
        "Views: 120,000\n"
        "Likes: 4,500\n"
        "URL: https://www.youtube.com/watch?v=JpSHyEIZ_bo\n"
        "Language: English\n"
        "Transcript source: manual captions\n"
        "\n"
        "First sentence. Second sentence.\n"
    )

    with pytest.raises(ValueError, match="use --force to replace it"):
        commands.transcript(
            argparse.Namespace(
                video_ref="JpSHyEIZ_bo",
                timestamps=True,
                export=True,
                force=False,
            )
        )

    commands.transcript(
        argparse.Namespace(
            video_ref="JpSHyEIZ_bo",
            timestamps=True,
            export=True,
            force=True,
        )
    )
    assert export_path.read_text(encoding="utf-8").endswith(
        "\n[00:00:00] First sentence.\n[00:01:02] Second sentence.\n"
    )


def test_video_transcript_force_requires_export(monkeypatch) -> None:
    monkeypatch.setattr(
        commands,
        "fetch_transcript",
        lambda _video_ref: pytest.fail("transcript should not be fetched"),
    )

    with pytest.raises(ValueError, match="--force requires --export"):
        commands.transcript(
            argparse.Namespace(
                video_ref="JpSHyEIZ_bo",
                timestamps=False,
                export=False,
                force=True,
            )
        )


@pytest.mark.parametrize(
    ("args", "usage"),
    [
        (
            ["transcript", "--help"],
            "usage: youtube transcript [-h] [--timestamps] [--export] [--force] url",
        ),
        (["metadata", "--help"], "usage: youtube metadata [-h] url"),
    ],
)
def test_reference_help_names_positionals_url(
    args: list[str],
    usage: str,
    capsys,
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)

    output = capsys.readouterr().out
    assert usage in " ".join(output.split())
    assert "url" in output
    assert "YouTube video URL or ID" in output
    assert "video_ref" not in output


def test_search_command_group_parses_channel_search() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "--channels",
            "@OpenAI",
            "--channels",
            "UC1234567890123456789012",
            "--limit",
            "2",
            "--sort",
            "popularity",
        ]
    )

    assert search_args.command == "search"
    assert search_args.channels == ["@OpenAI", "UC1234567890123456789012"]
    assert search_args.limit == 2
    assert search_args.sort == "popularity"


def test_search_command_group_allows_channel_search_without_query() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "--channels",
            "@OpenAI",
        ]
    )

    assert search_args.queries == []
    assert search_args.channels == ["@OpenAI"]


def test_search_command_group_parses_several_queries() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "agent skills",
            "tool use",
        ]
    )

    assert search_args.queries == ["agents", "agent skills", "tool use"]


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

    def fake_search_videos(request: SearchRequest) -> list[dict]:
        requests.append(request)
        return [{"id": "aaaaaaaaaaa"}]

    monkeypatch.setattr(commands, "search_videos", fake_search_videos)
    payload = commands.search(
        argparse.Namespace(
            queries=["agents", "agent skills", "tool use"],
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
            queries=("agents", "agent skills", "tool use"),
            mode="native",
            return_type=None,
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

    def fake_search_videos(request: SearchRequest) -> list[dict]:
        requests.append(request)
        return [{"id": "aaaaaaaaaaa"}]

    monkeypatch.setattr(commands, "search_videos", fake_search_videos)
    payload = commands.search(
        argparse.Namespace(
            queries=["agents"],
            uploaded_after=None,
            uploaded_within_days=None,
            mode="native",
            return_type=None,
            channels=["@OpenAI", "UC1234567890123456789012"],
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
            queries=("agents",),
            mode="native",
            return_type=None,
            channels=(
                ChannelTarget(channel_handle="@OpenAI"),
                ChannelTarget(youtube_channel_id="UC1234567890123456789012"),
            ),
            sort="popularity",
            min_views=5000,
            min_duration="3:00",
            max_duration="8:00:00",
        )
    ]


def test_emit_json_truncates_long_text_fields(capsys, monkeypatch) -> None:
    monkeypatch.setattr(output, "MAX_STDOUT_TEXT_FIELD_CHARS", 20)

    output.emit_json({"title": "y" * 100, "description": "x" * 100})

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "title": "y" * 100,
        "description": "x" * 20,
    }

    output.emit_json("z" * 100)

    assert json.loads(capsys.readouterr().out) == "z" * 20


@pytest.mark.parametrize("command", ["db", "watchlist", "blacklist", "index", "video"])
def test_superseded_commands_are_removed(command: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_search_source_flag_is_removed() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["search", "agents", "--source", "channel"])


def test_doctor_reports_local_artifacts_without_creating_them(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(health, "local_stt_installed", lambda: False)
    assert main(["doctor"]) == 0
    report = capsys.readouterr().out
    assert f"  artifacts  {tmp_path / '.cache' / 'youtube-toolkit'}\n" in report
    assert "  embedding  Set EMBEDDING_API_KEY" in report
    assert "Set XAI_API_KEY" in report
    assert "database" not in report
    assert not (tmp_path / ".cache").exists()


def test_stt_setup_reports_both_routes_and_bills_only_when_asked(monkeypatch) -> None:
    local_check = {"model": "example/speech-model", "device": "cpu", "working": True}
    status = {"provider": "xAI", "model": "example-model", "key_set": True}
    monkeypatch.setattr(commands, "cloud_stt_status", lambda: status)
    monkeypatch.setattr(
        commands, "check_cloud_stt", lambda: {**status, "working": True}
    )
    monkeypatch.setattr(commands, "check_local_stt", lambda: local_check)

    monkeypatch.setattr(commands, "local_stt_installed", lambda: True)
    plain = commands.stt_setup(build_parser().parse_args(["stt-setup"]))
    assert plain == {"cloud": status, "local": local_check}

    monkeypatch.setattr(commands, "local_stt_installed", lambda: False)
    tested = commands.stt_setup(
        build_parser().parse_args(["stt-setup", "--cloud-test"])
    )
    assert tested == {
        "cloud": {**status, "working": True},
        "local": {"installed": False},
    }


@pytest.mark.parametrize(
    ("environment", "installed", "downloaded", "expected"),
    [
        ({"XAI_API_KEY": "test-key"}, True, True, "Configured (xAI API)"),
        ({}, False, False, "Set XAI_API_KEY with `youtube setup --stt`"),
        ({}, True, False, "nvidia/parakeet-tdt-0.6b-v3 downloads on first use"),
        (
            {"STT_LOCAL_MODEL": "example/speech-model"},
            True,
            True,
            "Configured (local example/speech-model)",
        ),
        (
            {"STT_PROVIDER": "nobody", "XAI_API_KEY": "test-key"},
            True,
            True,
            "Configured (xAI API)",
        ),
    ],
)
def test_doctor_names_the_speech_to_text_route_or_the_next_step(
    monkeypatch, environment, installed, downloaded, expected
) -> None:
    for name in ("XAI_API_KEY", "STT_PROVIDER", "STT_LOCAL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(health, "local_stt_installed", lambda: installed)
    monkeypatch.setattr(health, "local_model_downloaded", lambda _model: downloaded)

    assert expected in health.stt_health()


def test_vision_requires_dependencies_before_fetch(monkeypatch, capsys) -> None:
    def reject() -> None:
        raise RuntimeError("Vision dependencies missing")

    monkeypatch.setattr(commands, "require_vision", reject)
    assert main(["vision", "JpSHyEIZ_bo"]) == 1
    assert "Vision dependencies missing" in capsys.readouterr().err


def test_setup_saves_keys_without_echoing_or_exporting_them(
    monkeypatch, capsys
) -> None:
    for name in (
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL",
        "STT_PROVIDER",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    answers = iter(
        ["https://embedding.example/v1", "example-model", "512", "", "", "gemini", ""]
    )
    keys = iter(["secret-embedding-key", "secret-gemini-key"])
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(setup.getpass, "getpass", lambda *_args, **_kwargs: next(keys))
    before = dict(os.environ)

    assert main(["setup"]) == 0

    captured = capsys.readouterr()
    saved = config.read_config()
    assert saved == {
        "EMBEDDING_API_KEY": "secret-embedding-key",
        "EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "EMBEDDING_MODEL": "example-model",
        "EMBEDDING_DIMENSIONS": "512",
        "STT_PROVIDER": "gemini",
        "GEMINI_API_KEY": "secret-gemini-key",
    }
    assert json.loads(captured.out) == {
        "config_path": str(config.config_path()),
        "saved_settings": list(saved),
    }
    assert "secret-embedding-key" not in captured.out + captured.err
    assert "secret-gemini-key" not in captured.out + captured.err
    assert dict(os.environ) == before
    assert health.embedding_health() == "Configured (example-model)"
    assert "Configured (Gemini API)" in health.stt_health()


def test_setup_keeps_or_removes_saved_values_without_changing_other_settings(
    monkeypatch, capsys
) -> None:
    config.save_config(
        {
            "EMBEDDING_API_KEY": "keep-key",
            "EMBEDDING_MODEL": "remove-model",
            "XAI_API_KEY": "unrelated-key",
        }
    )
    answers = iter(["", "-", "", "", ""])
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(setup.getpass, "getpass", lambda *_args, **_kwargs: "")

    assert main(["setup", "--embeddings"]) == 0
    assert config.read_config() == {
        "EMBEDDING_API_KEY": "keep-key",
        "XAI_API_KEY": "unrelated-key",
    }
    assert "keep-key" not in capsys.readouterr().out


def test_setup_stt_uses_the_selected_providers_key(monkeypatch) -> None:
    config.save_config({"EMBEDDING_API_KEY": "keep-key"})
    answers = iter(["vercel", ""])
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(
        setup.getpass, "getpass", lambda *_args, **_kwargs: "gateway-key"
    )

    assert main(["setup", "--stt"]) == 0
    assert config.read_config() == {
        "EMBEDDING_API_KEY": "keep-key",
        "STT_PROVIDER": "vercel",
        "AI_GATEWAY_API_KEY": "gateway-key",
    }


@pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt])
def test_setup_cancellation_preserves_existing_config(
    monkeypatch, capsys, failure
) -> None:
    path = config.save_config({"EMBEDDING_API_KEY": "old-key"})
    before = path.read_bytes()
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        setup.getpass, "getpass", lambda *_args, **_kwargs: "replacement-key"
    )

    def cancel() -> str:
        raise failure

    monkeypatch.setattr("builtins.input", cancel)
    assert main(["setup", "--embeddings"]) == 1
    assert path.read_bytes() == before
    assert "no settings were saved" in capsys.readouterr().err


def test_setup_rejects_invalid_settings_without_replacing_config(
    monkeypatch, capsys
) -> None:
    path = config.save_config({"EMBEDDING_API_KEY": "old-key"})
    before = path.read_bytes()
    answers = iter(["", "", "zero", "", ""])
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(setup.getpass, "getpass", lambda *_args, **_kwargs: "new-key")

    assert main(["setup", "--embeddings"]) == 1
    assert path.read_bytes() == before
    assert "EMBEDDING_DIMENSIONS must be a positive integer" in capsys.readouterr().err


def test_setup_requires_a_terminal_and_refuses_echoed_secret_input(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: False)
    assert main(["setup"]) == 1
    assert "interactive terminal" in capsys.readouterr().err
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)

    def no_hidden_input(*_args, **_kwargs) -> str:
        warnings.warn("Cannot hide input", setup.getpass.GetPassWarning, stacklevel=2)
        pytest.fail("Must not fall back to echoing the key")

    monkeypatch.setattr(setup.getpass, "getpass", no_hidden_input)
    assert main(["setup", "--embeddings"]) == 1
    assert "Hidden key entry is unavailable" in capsys.readouterr().err
    assert not config.config_path().exists()


def test_channel_command_dispatches_listing_without_subcommand(
    monkeypatch, capsys
) -> None:
    calls = []

    def fetch(channel_ref, *, options):
        calls.append((channel_ref, options.last, options.days))
        return {"entries": [{"id": "JpSHyEIZ_bo"}]}

    monkeypatch.setattr(commands, "fetch_channel_videos", fetch)
    assert main(["channel", "'@OpenAI'", "--last", "5", "--days", "7"]) == 0
    assert calls == [("@OpenAI", 5, 7)]
    assert json.loads(capsys.readouterr().out) == {"entries": [{"id": "JpSHyEIZ_bo"}]}
    assert build_parser().parse_args(["channel", "@OpenAI"]).last == 10
    with pytest.raises(SystemExit):
        build_parser().parse_args(["channel", "videos", "@OpenAI"])
