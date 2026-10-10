"""CLI contracts for retrieval archives, exports, and native or semantic search."""

import argparse
import json

import pytest

import youtube_toolkit.search.workflow as search_workflow
from youtube_toolkit.cli import commands, health, output
from youtube_toolkit.cli.__main__ import main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import TranscriptChunk, TranscriptDocument, VideoMetadata
from youtube_toolkit.search.request import ChannelTarget, SearchRequest


def test_parser_normalizes_string_arguments_before_handlers() -> None:
    """CLI string hygiene applies to every parsed argument value before command handlers run."""
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
    """The public metadata command excludes yt-dlp extraction internals."""

    class FakeClient:
        def metadata(self, video_ref: str) -> dict[str, object]:
            assert video_ref == "JpSHyEIZ_bo"
            return {
                "id": "JpSHyEIZ_bo",
                "title": "Example",
                "uploader": "Example Channel",
                "channel_id": "UC123",
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
        "channel_id": "UC123",
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


def test_video_transcript_command_selects_plain_or_timestamped_output(
    monkeypatch,
) -> None:
    """Transcript output is one text string unless timed chunks are requested."""
    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo",
            url="https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        ),
        language="en",
        source="manual",
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
    """Export follows the resolved transcript document and writes one deterministic text file."""
    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo",
            url="https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            title="What is DARKMATTER ?",
            channel="Example Channel",
            channel_id="UC123",
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
    """Overwrite permission is invalid when no file export was requested."""
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
    """Reference help shows the shell-facing URL placeholder instead of internal destination names."""
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
            "UC123",
            "--limit",
            "2",
            "--sort",
            "popularity",
        ]
    )

    assert search_args.command == "search"
    assert search_args.channels == ["@OpenAI", "UC123"]
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

    def fake_run_search_request(request: SearchRequest) -> list[dict]:
        requests.append(request)
        return [{"id": "aaaaaaaaaaa"}]

    monkeypatch.setattr(search_workflow, "search", fake_run_search_request)
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

    monkeypatch.setattr(search_workflow, "search", fake_run_search_request)
    payload = commands.search(
        argparse.Namespace(
            queries=["agents"],
            uploaded_after=None,
            uploaded_within_days=None,
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
            queries=("agents",),
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


def test_emit_json_truncates_long_text_fields(capsys, monkeypatch) -> None:
    """Large text fields are capped without cutting the whole JSON payload."""
    monkeypatch.setattr(output, "MAX_STDOUT_TEXT_FIELD_CHARS", 20)

    output.emit_json({"title": "y" * 100, "transcript": "x" * 100})

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "title": "y" * 100,
        "transcript": "x" * 20,
    }

    output.emit_json("z" * 100)

    assert json.loads(capsys.readouterr().out) == "z" * 20


@pytest.mark.parametrize(
    "command", ["db", "watchlist", "blacklist", "index", "setup", "video"]
)
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
    payload = health.run_doctor(argparse.Namespace())
    assert payload["artifacts"] == str(tmp_path / ".cache" / "youtube-toolkit")
    assert payload["embedding"]["status"] == "missing"
    assert not (tmp_path / ".cache").exists()
    assert main(["doctor"]) == 0
    report = capsys.readouterr().out
    assert str(tmp_path / ".cache" / "youtube-toolkit") in report
    assert "database" not in report


def test_vision_requires_dependencies_before_fetch(monkeypatch, capsys) -> None:
    def reject() -> None:
        raise RuntimeError("Vision dependencies missing")

    monkeypatch.setattr(commands, "require_vision", reject)
    assert main(["vision", "JpSHyEIZ_bo"]) == 1
    assert "Vision dependencies missing" in capsys.readouterr().err


def test_channel_command_dispatches_listing_without_subcommand(
    monkeypatch, capsys
) -> None:
    """The channel command keeps listing limits and JSON output after removing the videos tier."""
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
