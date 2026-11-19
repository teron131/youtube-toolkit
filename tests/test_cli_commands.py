"""CLI command tests."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import numpy as np
from PIL import Image
import pytest

from youtube_toolkit import config, download, summary
from youtube_toolkit.artifacts import Cached
from youtube_toolkit.cli import commands, health, setup
from youtube_toolkit.cli.__main__ import main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import (
    TranscriptDocument,
    TranscriptRewrite,
    TranscriptSegment,
    VideoMetadata,
)
from youtube_toolkit.search.request import ChannelTarget, SearchRequest
from youtube_toolkit.youtube import YtDlpClient

VIDEOS = Path(".youtube-toolkit/videos")


@pytest.fixture(autouse=True)
def working_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)


def saved_part(capsys, part: str) -> dict[str, Any]:
    path = Path(json.loads(capsys.readouterr().out)[part]["path"])
    return json.loads(path.read_text(encoding="utf-8"))


def test_parser_normalizes_string_arguments_before_handlers() -> None:
    video_args = build_parser().parse_args(
        [
            "video",
            "'https://www.youtube.com/watch?v=JpSHyEIZ_bo'",
            "--comments",
            "' pricing '",
        ]
    )
    search_args = build_parser().parse_args(
        [
            "search",
            "'agent memory'",
            "'langgraph'",
            "--semantic",
            "'video'",
            "--in-channels",
            "'@OpenAI'",
            "--include-channel-names",
            "'OpenAI'",
            "--exclude-titles",
            "'shorts'",
            "--sort",
            "'popularity'",
        ]
    )
    unmatched_quote_args = build_parser().parse_args(["search", "'earnings"])

    assert video_args.video_ref == "https://www.youtube.com/watch?v=JpSHyEIZ_bo"
    assert video_args.comments == "pricing"
    assert search_args.queries == ["agent memory", "langgraph"]
    assert search_args.semantic == "video"
    assert search_args.channels == ["@OpenAI"]
    assert search_args.include_channel == ["OpenAI"]
    assert search_args.exclude_title == ["shorts"]
    assert search_args.sort == "popularity"
    assert unmatched_quote_args.queries == ["'earnings"]


def test_video_command_saves_metadata_alone_without_part_flags(
    monkeypatch, tmp_path, capsys
) -> None:
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

    assert main(["video", "JpSHyEIZ_bo"]) == 0
    path = tmp_path / VIDEOS / "Example_JpSHyEIZ_bo" / "metadata.json"
    assert json.loads(capsys.readouterr().out) == {
        "metadata": {
            "path": str(path),
            "title": "Example",
            "channel": "Example Channel",
            "duration_seconds": 125.5,
        }
    }
    assert json.loads(path.read_text(encoding="utf-8")) == {
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
def comment_response(monkeypatch):
    """Fakes only the yt-dlp process, so the real parsing and recording run."""
    payload = {"id": "JpSHyEIZ_bo"}
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", run)
    return payload, calls


@pytest.mark.parametrize(
    ("arguments", "expected_ids", "limit"),
    [
        (["PRICING"], ["first", "second"], 100),
        (["pricing details", "--comment-limit", "1000"], ["first"], 1000),
        (["[pricing]"], ["second"], 100),
        (["STRASSE"], ["second"], 100),
        ([], ["first", "second", "third"], 100),
        ([""], ["first", "second", "third"], 100),
        (["   "], ["first", "second", "third"], 100),
        (["unmatched"], [], 100),
    ],
)
def test_comments_search_saves_matches_and_records_the_full_sample(
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

    assert main(["video", "JpSHyEIZ_bo", "--comments", *arguments]) == 0

    captured = capsys.readouterr()
    path = tmp_path / VIDEOS / "Example_JpSHyEIZ_bo" / "comments.json"
    assert json.loads(captured.out) == {
        "comments": {"path": str(path), "matches": len(expected_ids), "sampled": 3}
    }
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "title": "Example",
        "url": "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        "query": arguments[0].strip() if arguments else "",
        "sampled": 3,
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
    [record_path] = (tmp_path / ".youtube-toolkit/records/comments").glob("*.json")
    assert json.loads(record_path.read_text(encoding="utf-8")) == {
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

    assert main(["video", "JpSHyEIZ_bo", "--comments", "pricing"]) == 0
    assert saved_part(capsys, "comments")["comments"] == []


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

    assert main(["video", "JpSHyEIZ_bo", "--comments", query]) == 0
    result = saved_part(capsys, "comments")
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
    assert main(["video", "JpSHyEIZ_bo", "--comments"]) == 0
    assert saved_part(capsys, "comments")["comments"] == [
        {"text": "Same text", "like_count": 1},
        {"text": "C", "like_count": 0},
        {"text": "B", "like_count": None},
        {"text": "D", "like_count": 0},
        {"text": "Same text", "like_count": 0},
    ]


def test_comments_refetches_and_preserves_lean_record_history(
    comment_response, tmp_path
) -> None:
    payload, calls = comment_response
    payload["comments"] = [{"id": "a", "text": "A", "like_count": 1}]
    assert main(["video", "JpSHyEIZ_bo", "--comments"]) == 0
    directory = tmp_path / ".youtube-toolkit/records/comments"
    first_path = next(directory.glob("*.json"))
    first_copy = first_path.read_bytes()
    payload["comments"][0]["like_count"] = 2
    assert main(["video", "JpSHyEIZ_bo", "--comments"]) == 0
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

    assert (
        main(
            ["video", "JpSHyEIZ_bo", "--comments", "pricing", "--comment-limit", limit]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--comment-limit must be between 1 and 1000" in captured.err
    assert calls == []


@pytest.fixture
def metadata_calls(monkeypatch):
    calls = []

    class Client:
        def metadata(self, video_ref):
            calls.append(video_ref)
            return {"id": "JpSHyEIZ_bo", "title": "Ray tracing"}

    monkeypatch.setattr(commands, "YtDlpClient", Client)
    return calls


def test_video_transcript_saves_a_reading_copy_and_reuses_the_fetch(
    capsys, monkeypatch, tmp_path, metadata_calls
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
        segments=[
            TranscriptSegment(
                start_seconds=0.0, end_seconds=2.5, text="First sentence."
            ),
            TranscriptSegment(
                start_seconds=62.0, end_seconds=65.0, text="Second sentence."
            ),
        ],
    )
    caption_fetches = []

    def fetch_captions(metadata, *, languages):
        caption_fetches.append(metadata)
        return document

    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript", fetch_captions
    )
    header = (
        "# What is DARKMATTER ?\n\n"
        "- Author: Example Channel\n"
        "- Channel: @example\n"
        "- Published: 2026-07-16\n"
        "- Duration: 01:23:45\n"
        "- URL: https://www.youtube.com/watch?v=JpSHyEIZ_bo\n"
        "- Transcript source: manual captions\n"
    )

    assert main(["video", "JpSHyEIZ_bo", "--transcript"]) == 0

    result = json.loads(capsys.readouterr().out)["transcript"]
    path = Path(result.pop("path"))
    text = header + "\nFirst sentence.\nSecond sentence.\n"
    assert result == {
        "characters": len(text),
        "language": "en",
        "source": "subtitles",
        "reused": False,
    }
    assert path.parent == tmp_path / VIDEOS / "Ray_tracing_JpSHyEIZ_bo" / "transcripts"
    assert path.name.startswith("original_fallback-")
    assert path.suffix == ".md"
    assert path.with_suffix(".json").is_file()
    assert path.read_text(encoding="utf-8") == text

    assert main(["video", "JpSHyEIZ_bo", "--transcript", "--timestamps"]) == 0
    timed = json.loads(capsys.readouterr().out)["transcript"]
    assert timed["path"] == str(path.with_suffix(".timed.md"))
    assert timed["reused"] is True
    assert len(caption_fetches) == 1
    assert Path(timed["path"]).read_text(encoding="utf-8") == (
        header + "\n[00:00:00] First sentence.\n\n[00:01:02] Second sentence.\n"
    )

    assert main(["video", "JpSHyEIZ_bo", "--transcript", "--fresh"]) == 0
    assert json.loads(capsys.readouterr().out)["transcript"]["reused"] is False
    assert len(caption_fetches) == 2

    monkeypatch.setenv("LLM_API_KEY", "key")
    rewrites = []

    def rewrite_transcript(original, _llm, *, language, refine, article, fresh):
        rewrites.append((language, refine, article, fresh))
        name = "_".join([language or "en", *(["refined"] if refine else [])])
        return Cached(
            replace(
                original.value,
                language=language or original.value.language,
                segments=[TranscriptSegment(0.0, 65.0, "Rewritten paragraph.")],
                rewrite=TranscriptRewrite("example/model", translated=bool(language)),
            ),
            False,
            original.path.with_name(f"{name}_article-abc.json"),
        )

    monkeypatch.setattr(commands, "rewrite_transcript", rewrite_transcript)
    for flags, name, line in [
        (["--translate", "zh-TW"], "zh-TW", "- Translation: zh-TW by example/model"),
        (["--refine"], "en_refined", "- Refined by: example/model"),
    ]:
        assert main(["video", "JpSHyEIZ_bo", "--transcript", *flags]) == 0
        rewritten = Path(json.loads(capsys.readouterr().out)["transcript"]["path"])
        assert rewritten == path.with_name(f"{name}_article-abc.md")
        assert rewritten.read_text(encoding="utf-8").endswith(
            f"{line}\n\nRewritten paragraph.\n"
        )
    assert rewrites == [("zh-TW", False, True, False), (None, True, True, False)]


def test_a_speech_to_text_transcript_names_the_model_that_transcribed_it(
    capsys, monkeypatch, tmp_path, metadata_calls
) -> None:
    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo", url="https://www.youtube.com/watch?v=JpSHyEIZ_bo"
        ),
        language="yue",
        source="speech-to-text",
        segments=[TranscriptSegment(0.0, 2.5, "First sentence.")],
        stt_model="grok-voice-transcribe-2.0",
    )
    monkeypatch.setattr(
        commands,
        "fetch_transcript",
        lambda _video_ref, **_options: Cached(
            document, False, tmp_path / "original_always-abc.json"
        ),
    )

    assert main(["video", "JpSHyEIZ_bo", "--transcript"]) == 0

    path = Path(json.loads(capsys.readouterr().out)["transcript"]["path"])
    assert path == tmp_path / "original_always-abc.md"
    assert (
        "- Transcript source: speech-to-text\n- Transcribed by: grok-voice-transcribe-2.0\n"
        in path.read_text(encoding="utf-8")
    )


def test_video_parts_share_one_metadata_fetch_and_each_save_a_file(
    capsys, monkeypatch, tmp_path, metadata_calls
) -> None:
    from youtube_toolkit import vision as vision_package

    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo",
            url="https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            title="Ray tracing",
            channel="Example Channel",
        ),
        language="en",
        source="subtitles",
        segments=[
            TranscriptSegment(0.0, 10.0, "Intro."),
            TranscriptSegment(10.0, 20.0, "The ray tracer."),
        ],
    )
    transcripts = tmp_path / VIDEOS / "Ray_tracing_JpSHyEIZ_bo" / "transcripts"
    vision = {
        "path": str(tmp_path / "picks.json"),
        "collages": [str(tmp_path / "page1.jpg")],
        "frames": 1,
        "selection_mode": "visual",
        "warnings": [],
        "reused": False,
    }
    received = []
    policies = []

    def fetch_transcript(_video_ref, *, stt, client, metadata, fresh):
        policies.append(stt)
        received.append(metadata)
        return Cached(document, False, transcripts / "original_always-abc.json")

    def sample_vision(_video_ref, *, metadata, document, start, end, fresh, video_path):
        received.extend([metadata, document])
        return vision

    monkeypatch.setattr(commands, "require_vision", lambda: None)
    monkeypatch.setattr(commands, "fetch_transcript", fetch_transcript)
    monkeypatch.setattr(vision_package, "sample_vision", sample_vision)
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setattr(
        commands,
        "rewrite_transcript",
        lambda original, _llm, *, language, refine, article, fresh: Cached(
            replace(
                original.value,
                language=language,
                segments=[TranscriptSegment(0.0, 20.0, "Translated.")],
                rewrite=TranscriptRewrite("example/model", translated=True),
            ),
            False,
            transcripts / "ja_article-def.json",
        ),
    )

    summaries = []

    def summarize_transcript(transcript, _llm, *, language, fresh):
        summaries.append((transcript.value, language))
        return Cached("Summary.", False, transcripts.parent / "summaries" / "ja-1.json")

    monkeypatch.setattr(summary, "summarize_transcript", summarize_transcript)

    parts = ["--vision", "--transcript", "--stt", "--translate", "ja", "--metadata"]
    assert main(["video", "JpSHyEIZ_bo", *parts, "--summary"]) == 0
    assert policies == ["always"]
    output = json.loads(capsys.readouterr().out)
    assert list(output) == ["transcript", "summary", "vision", "metadata"]
    assert output["metadata"]["title"] == "Ray tracing"
    assert output["metadata"]["path"] == str(transcripts.parent / "metadata.json")
    assert output["transcript"]["path"] == str(transcripts / "ja_article-def.md")
    assert output["transcript"]["language"] == "ja"
    assert (
        Path(output["transcript"]["path"])
        .read_text(encoding="utf-8")
        .endswith("\n\nTranslated.\n")
    )
    summary_path = transcripts.parent / "summaries" / "ja-1.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    assert output["summary"] == {
        "path": str(summary_path),
        "characters": len(summary_text),
        "language": "ja",
        "reused": False,
    }
    assert summary_text.endswith("\n\nSummary.\n")
    assert summaries == [(document, "ja")]
    assert output["vision"] == vision
    payload = {"id": "JpSHyEIZ_bo", "title": "Ray tracing"}
    assert metadata_calls == ["JpSHyEIZ_bo"]
    assert received == [payload, payload, document]


@pytest.mark.parametrize(
    ("flags", "language", "output_language"),
    [
        (["--summary"], None, "en"),
        (["--summary", "zh-TW"], "zh-TW", "zh-TW"),
        (["--summary", "--translate", "ja"], "ja", "ja"),
        (["--summary", "--stt"], None, "en"),
    ],
)
def test_video_summary_alone_gives_only_the_summary(
    capsys, monkeypatch, tmp_path, metadata_calls, flags, language, output_language
) -> None:
    document = TranscriptDocument(
        metadata=VideoMetadata(
            video_id="JpSHyEIZ_bo", url="https://www.youtube.com/watch?v=JpSHyEIZ_bo"
        ),
        language="en",
        source="subtitles",
        segments=[TranscriptSegment(0.0, 2.5, "First sentence.")],
    )
    languages = []
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setattr(
        commands,
        "fetch_transcript",
        lambda _video_ref, **_options: Cached(
            document, False, tmp_path / "transcripts" / "original-abc.json"
        ),
    )
    monkeypatch.setattr(
        summary,
        "summarize_transcript",
        lambda _transcript, _llm, *, language, fresh: (
            languages.append(language)
            or Cached("Summary.", False, tmp_path / "summaries" / "summary-abc.json")
        ),
    )
    monkeypatch.setattr(
        commands,
        "rewrite_transcript",
        lambda *_args, **_options: pytest.fail("no transcript is given to rewrite"),
    )

    assert main(["video", "JpSHyEIZ_bo", *flags]) == 0
    path = tmp_path / "summaries" / "summary-abc.md"
    assert json.loads(capsys.readouterr().out) == {
        "summary": {
            "path": str(path),
            "characters": len(path.read_text(encoding="utf-8")),
            "language": output_language,
            "reused": False,
        }
    }
    assert languages == [language]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--timestamps"], "--timestamps requires --transcript"),
        (["--stt"], "--stt requires --transcript or --summary"),
        (["--refine"], "--refine requires --transcript"),
        (["--summary", "--refine"], "--refine requires --transcript"),
        (["--translate"], "--translate requires --transcript or --summary"),
        (["--transcript", "--refine"], "and --summary need LLM_API_KEY"),
        (["--transcript", "--translate"], "and --summary need LLM_API_KEY"),
        (["--summary"], "and --summary need LLM_API_KEY"),
        (["--metadata", "--fresh"], "--fresh requires --transcript, --summary"),
        (["--comment-limit", "5"], "--comment-limit requires --comments"),
        (["--transcript", "--max-frames", "5"], "--max-frames requires --vision"),
        (["--no-auto-clean"], "--auto-clean requires --vision"),
        (["--quality", "720p"], "--quality requires --download"),
        (["--download", "audio", "--quality", "720p"], "only to video and frame"),
        (["--start", "1:00"], "--start requires --download or --vision"),
        (["--at", "1:00"], "--at requires --download"),
        (["--download", "keyframes", "--at", "1:00"], "only to --download frames"),
        (["--download", "frames", "--at", "1:00", "--end", "2:00"], "--start or --end"),
        (["--download", "--start", "1::30"], "Invalid duration: 1::30"),
        (
            ["--fresh"],
            "--fresh requires --transcript, --summary, --vision, or --download",
        ),
    ],
)
def test_video_part_options_require_their_part_before_any_fetch(
    capsys, metadata_calls, arguments, message
) -> None:
    assert main(["video", "JpSHyEIZ_bo", *arguments]) == 1
    assert message in capsys.readouterr().err
    assert metadata_calls == []


DOWNLOAD_METADATA = {
    "id": "JpSHyEIZ_bo",
    "title": "Ray tracing",
    "channel": "Example Channel",
    "duration": 300,
}


@pytest.fixture
def downloads(monkeypatch):
    """The yt-dlp downloads asked for, each as the metadata it starts from, its path, and options; the fake writes the file it is given."""
    calls = []

    class Client:
        def metadata(self, _video_ref):
            return DOWNLOAD_METADATA

        def download_media(self, metadata, path, **options):
            calls.append((metadata, path, options))
            path.write_bytes(b"media")

        def download_silent_video(self, metadata, path, **options):
            calls.append((metadata, path, options))
            path.write_bytes(b"video")

        def download_silent_clips(self, metadata, directory, **options):
            calls.append((metadata, directory, options))
            clips = [directory / f"{start}.mp4" for start, _ in options["sections"]]
            for clip in clips:
                clip.write_bytes(b"clip")
            return clips

    monkeypatch.setattr(download.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(commands, "YtDlpClient", Client)
    return calls


def test_a_video_downloads_whole_at_1080p_under_its_title_and_channel(
    capsys, tmp_path, downloads
) -> None:
    assert main(["video", "JpSHyEIZ_bo", "--download"]) == 0

    path = tmp_path / "Ray_tracing_Example_Channel_1080p.mp4"
    assert json.loads(capsys.readouterr().out) == {
        "download": {"path": str(path), "bytes": len(b"media"), "reused": False}
    }
    assert downloads == [
        (
            DOWNLOAD_METADATA,
            path,
            {
                "height": 1080,
                "section": None,
                "overwrite": False,
                "timeout_seconds": 600,
            },
        )
    ]


def test_a_range_of_sound_downloads_alone_and_stops_at_the_videos_end(
    capsys, tmp_path, downloads
) -> None:
    arguments = ["--download", "audio", "--start", "1:15", "--end", "9:00"]

    assert main(["video", "JpSHyEIZ_bo", *arguments]) == 0

    path = tmp_path / "Ray_tracing_Example_Channel_00h01m15.000s_00h05m00.000s.m4a"
    assert json.loads(capsys.readouterr().out) == {
        "download": {"path": str(path), "bytes": len(b"media"), "reused": False}
    }
    assert downloads[0][2] == {
        "height": None,
        "section": (75.0, 300.0),
        "overwrite": False,
        "timeout_seconds": 600,
    }


def test_an_earlier_download_comes_back_reused_unless_fresh(
    capsys, tmp_path, downloads
) -> None:
    earlier = (
        tmp_path / "Ray_tracing_Example_Channel_720p_00h01m00.000s_00h02m00.000s.mp4"
    )
    earlier.write_bytes(b"earlier")
    clip = ["video", "JpSHyEIZ_bo", "--download", "--quality", "720p"]
    clip += ["--start", "1:00", "--end", "2:00"]

    assert main(clip) == 0
    assert json.loads(capsys.readouterr().out) == {
        "download": {"path": str(earlier), "bytes": len(b"earlier"), "reused": True}
    }
    assert downloads == []
    assert main([*clip[:-1], "2:01"]) == 0
    assert main(["video", "JpSHyEIZ_bo", "--download", "--quality", "360p"]) == 0
    assert [path.name for _, path, _ in downloads] == [
        "Ray_tracing_Example_Channel_720p_00h01m00.000s_00h02m01.000s.mp4",
        "Ray_tracing_Example_Channel_360p.mp4",
    ]
    capsys.readouterr()
    assert main([*clip, "--fresh"]) == 0
    assert json.loads(capsys.readouterr().out)["download"]["reused"] is False
    assert downloads[-1][1] == earlier
    assert downloads[-1][2]["overwrite"] is True


def test_a_range_downloads_in_bounded_requests_youtube_does_not_throttle(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_run(command, input, capture_output, text, check, timeout):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"media")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)

    YtDlpClient().download_media(
        {"id": "JpSHyEIZ_bo"},
        tmp_path / "clip.mp4",
        height=1080,
        section=(75.0, 150.0),
        overwrite=False,
        timeout_seconds=600,
    )

    command = calls[0]
    assert command[command.index("--download-sections") + 1] == "*75.0-150.0"
    assert command[command.index("--downloader-args") + 1].startswith(
        "ffmpeg_i:-request_size "
    )


@pytest.mark.parametrize(
    "arguments", [["--start", "4:00", "--end", "1:00"], ["--start", "5:00"]]
)
def test_a_range_that_holds_nothing_fails_before_downloading(
    capsys, downloads, arguments
) -> None:
    assert main(["video", "JpSHyEIZ_bo", "--download", *arguments]) == 1
    assert "must start before it ends" in capsys.readouterr().err
    assert downloads == []


@pytest.mark.parametrize(
    ("arguments", "reused"),
    [([], True), (["--quality", "480p"], False), (["--end", "1:00"], True)],
)
def test_vision_picks_from_a_video_downloaded_at_720p_or_more_over_its_range(
    monkeypatch, downloads, arguments, reused
) -> None:
    from youtube_toolkit import vision as vision_package

    received = []

    def sample_vision(_video_ref, *, video_path, start, end, **_options):
        received.append(video_path)
        assert (start, end) == (None, 60.0 if "--end" in arguments else None)
        return {"frame_count": 1}

    monkeypatch.setattr(commands, "require_vision", lambda: None)
    monkeypatch.setattr(vision_package, "sample_vision", sample_vision)
    args = build_parser().parse_args(
        ["video", "JpSHyEIZ_bo", "--download", "--vision", *arguments]
    )

    result = commands.video(args)

    assert received == [Path(result["download"]["path"]) if reused else None]


@pytest.fixture
def frame_extraction(monkeypatch):
    """Each keyframe extraction asked for, as its start, duration, size, and distinct switch; the fake saves frames at 62.81275 and 70 seconds."""
    from youtube_toolkit import vision as vision_package
    from youtube_toolkit.vision.candidates import StoredFrame, VisualScores

    calls = []

    def collect_candidates(
        _video_path, _ffmpeg, *, start, duration, frame_size, directory, distinct, **_
    ):
        calls.append((start, duration, frame_size, distinct))
        directory.mkdir(parents=True, exist_ok=True)
        frames = []
        for time in (62.81275, 70.0):
            path = directory / f"frame-{round(time * 1000):09d}.jpg"
            path.write_bytes(b"jpeg")
            frames.append(StoredFrame(time, path, VisualScores(0.0, 0.0, 0.0)))
        return frames

    monkeypatch.setattr(commands, "require_vision", lambda: None)
    monkeypatch.setattr(vision_package, "collect_candidates", collect_candidates)
    return calls


def test_frames_download_into_a_folder_named_by_quality_kind_and_range(
    capsys, tmp_path, downloads, frame_extraction
) -> None:
    clip = ["video", "JpSHyEIZ_bo", "--download", "frames", "--quality", "720p"]
    clip += ["--start", "1:00", "--end", "2:00"]

    assert main(clip) == 0

    folder = (
        tmp_path / "Ray_tracing_Example_Channel_720p_frames_00h01m00.000s_00h02m00.000s"
    )
    assert json.loads(capsys.readouterr().out) == {
        "download": {"path": str(folder), "frames": 2, "reused": False}
    }
    # Names round down to the millisecond and list in time order.
    assert sorted(path.name for path in folder.iterdir()) == [
        "00h01m02.812s.jpg",
        "00h01m10.000s.jpg",
    ]
    assert downloads[0][2] == {
        "height": 720,
        "section": (60.0, 120.0),
        "max_bytes": None,
        "timeout_seconds": 600,
    }
    assert frame_extraction == [(60.0, 60.0, (1280, 720), True)]
    assert not list(tmp_path.glob("*.part"))
    assert not list((tmp_path / ".youtube-toolkit/tmp").iterdir())
    assert main(clip) == 0
    assert json.loads(capsys.readouterr().out) == {
        "download": {"path": str(folder), "frames": 2, "reused": True}
    }
    assert len(downloads) == 1
    assert main(["video", "JpSHyEIZ_bo", "--download", "keyframes"]) == 0
    assert (tmp_path / "Ray_tracing_Example_Channel_1080p_keyframes").is_dir()
    assert frame_extraction[-1] == (0.0, 300, (1920, 1080), False)


def test_frames_at_given_times_download_together_and_come_back_when_present(
    monkeypatch, capsys, tmp_path, downloads
) -> None:
    earlier = tmp_path / "Ray_tracing_Example_Channel_360p_00h01m15.000s.jpg"
    earlier.write_bytes(b"earlier")
    monkeypatch.setattr(commands, "require_vision", lambda: None)
    monkeypatch.setattr(
        download,
        "decode_frame",
        lambda _clip, timestamp, _ffmpeg, *, width, height: np.full(
            (height, width, 3), timestamp, dtype=np.uint8
        ),
    )
    times = ["--at", "0:30", "1:15", "0:30", "4:59"]

    assert (
        main(
            [
                "video",
                "JpSHyEIZ_bo",
                "--download",
                "frames",
                *times,
                "--quality",
                "360p",
            ]
        )
        == 0
    )

    results = json.loads(capsys.readouterr().out)["download"]
    assert [Path(result["path"]).name for result in results] == [
        "Ray_tracing_Example_Channel_360p_00h00m30.000s.jpg",
        "Ray_tracing_Example_Channel_360p_00h01m15.000s.jpg",
        "Ray_tracing_Example_Channel_360p_00h04m59.000s.jpg",
    ]
    assert [result["reused"] for result in results] == [False, True, False]
    [(_, _, options)] = downloads
    assert options["sections"] == [(30.0, 31.0), (299.0, 300)]
    assert options["height"] == 360
    with Image.open(results[0]["path"]) as image:
        assert image.size == (640, 360)
    assert earlier.read_bytes() == b"earlier"
    assert main(["video", "JpSHyEIZ_bo", "--download", "frames", "--at", "5:00"]) == 1
    assert "Frame time 300 seconds is outside the video" in capsys.readouterr().err


def test_clips_download_in_one_run_from_fetched_metadata_and_come_back_in_order(
    monkeypatch, tmp_path
) -> None:
    calls = []
    metadata = {"id": "JpSHyEIZ_bo", "formats": []}

    def fake_run(command, input, capture_output, text, check, timeout):
        calls.append((command, input))
        stdout = f"75 {tmp_path}/75.mp4\n10 {tmp_path}/10.mp4\n"
        return SimpleNamespace(returncode=0, stderr="", stdout=stdout)

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)

    clips = YtDlpClient().download_silent_clips(
        metadata,
        tmp_path,
        height=720,
        sections=[(10.0, 11.0), (75.0, 76.0)],
        timeout_seconds=600,
    )

    assert clips == [tmp_path / "10.mp4", tmp_path / "75.mp4"]
    [(command, stdin)] = calls
    assert command.count("--download-sections") == 2
    assert command.count("--downloader-args") == 1
    assert command[-2:] == ["--load-info-json", "-"]
    assert json.loads(stdin) == metadata


@pytest.mark.parametrize(("flags", "language"), [([], "en"), (["ja"], "ja")])
def test_video_translate_defaults_to_english(flags, language) -> None:
    arguments = ["video", "JpSHyEIZ_bo", "--transcript", "--translate", *flags]
    assert build_parser().parse_args(arguments).translate == language


def test_video_help_names_the_positional_url(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["video", "--help"])

    output = capsys.readouterr().out
    assert "usage: youtube video [-h] [--transcript] [--timestamps]" in " ".join(
        output.split()
    )
    assert "YouTube video URL or ID" in output
    assert "video_ref" not in output


def test_search_command_group_parses_channel_search() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "--in-channels",
            "@OpenAI",
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
            "--in-channels",
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


def test_search_command_group_collects_listed_filters() -> None:
    search_args = build_parser().parse_args(
        [
            "search",
            "agents",
            "--include-channel-names",
            "OpenAI",
            "Anthropic",
            "--exclude-channel-names",
            "shorts",
            "clips",
            "--include-titles",
            "agent",
            "tools",
            "--exclude-titles",
            "reaction",
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
            semantic=None,
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
            semantic=None,
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
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(health, "local_stt_installed", lambda: False)
    assert main(["doctor"]) == 0
    report = capsys.readouterr().out
    assert f"  artifacts  {tmp_path / '.youtube-toolkit'}\n" in report
    assert "  embedding  Set EMBEDDING_API_KEY" in report
    assert "Set XAI_API_KEY" in report
    assert "database" not in report
    assert not (tmp_path / ".youtube-toolkit").exists()


def test_stt_check_reports_both_routes_and_bills_only_when_asked(monkeypatch) -> None:
    local_check = {"model": "example/speech-model", "device": "cpu", "working": True}
    status = {"provider": "xAI", "model": "example-model", "key_set": True}
    monkeypatch.setattr(commands, "cloud_stt_status", lambda: status)
    monkeypatch.setattr(
        commands, "check_cloud_stt", lambda: {**status, "working": True}
    )
    monkeypatch.setattr(commands, "check_local_stt", lambda: local_check)

    monkeypatch.setattr(commands, "local_stt_installed", lambda: True)
    plain = commands.stt_check(build_parser().parse_args(["stt-check"]))
    assert plain == {"cloud": status, "local": local_check}

    monkeypatch.setattr(commands, "local_stt_installed", lambda: False)
    tested = commands.stt_check(
        build_parser().parse_args(["stt-check", "--cloud-test"])
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
    monkeypatch.setattr(health, "local_stt_downloaded", lambda _model: downloaded)

    assert expected in health.stt_health()


@pytest.mark.parametrize(
    ("environment", "installed", "expected"),
    [
        (
            {
                "DECISION_API_KEY": "key",
                "DECISION_BASE_URL": "https://openrouter.ai/api",
                "EMBEDDING_API_KEY": "key",
            },
            True,
            "available; ranks with perplexity/pplx-decider-flash-v1.3 at openrouter.ai",
        ),
        ({"EMBEDDING_API_KEY": "key"}, True, "available; ranks with Gemini embeddings"),
        (
            {},
            True,
            "available; ranks with local jinaai/jina-embeddings-v5-omni-small-retrieval",
        ),
        (
            {},
            False,
            "available; picks by time slots until `youtube setup --decisions`",
        ),
    ],
)
def test_doctor_names_the_vision_ranking_route_in_preference_order(
    monkeypatch, environment, installed, expected
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(health, "local_embedding_installed", lambda: installed)
    monkeypatch.setattr(health, "vision_available", lambda: True)

    assert health.vision_health().startswith(expected)


def test_doctor_reports_a_key_named_after_an_unset_variable_on_its_own_line(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("DECISION_API_KEY", "MISSING_API_KEY")
    monkeypatch.setattr(health, "local_stt_installed", lambda: False)
    assert main(["doctor"]) == 0
    report = capsys.readouterr().out
    assert (
        "  decision   DECISION_API_KEY names MISSING_API_KEY, which is not set"
        in report
    )
    assert "  stt        Set XAI_API_KEY" in report


def test_vision_requires_dependencies_before_fetch(monkeypatch, capsys) -> None:
    def reject() -> None:
        raise RuntimeError("Vision dependencies missing")

    monkeypatch.setattr(commands, "require_vision", reject)
    assert main(["video", "JpSHyEIZ_bo", "--vision"]) == 1
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
        [
            "https://embedding.example/v1",
            "example-model",
            "512",
            "",
            "",
            "",
            "",
            "vercel",
            "",
            "",
            "",
        ]
    )
    # The decision and LLM keys name the variable that speech-to-text saves, so one key serves all three.
    keys = iter(
        [
            "secret-embedding-key",
            "AI_GATEWAY_API_KEY",
            "secret-gateway-key",
            "AI_GATEWAY_API_KEY",
        ]
    )
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
        "DECISION_API_KEY": "AI_GATEWAY_API_KEY",
        "STT_PROVIDER": "vercel",
        "AI_GATEWAY_API_KEY": "secret-gateway-key",
        "LLM_API_KEY": "AI_GATEWAY_API_KEY",
    }
    assert json.loads(captured.out) == {
        "config_path": str(config.config_path()),
        "saved_settings": list(saved),
    }
    assert "secret-embedding-key" not in captured.out + captured.err
    assert "secret-gateway-key" not in captured.out + captured.err
    assert dict(os.environ) == before
    assert health.embedding_health() == "Configured (example-model)"
    assert config.config_value("DECISION_API_KEY") == "secret-gateway-key"
    assert health.llm_health() == (
        "Configured (openai/gpt-6-luna at ai-gateway.vercel.sh)"
    )
    assert health.decision_health() == (
        "Configured (perplexity/pplx-decider-flash-v1.3 at ai-gateway.vercel.sh)"
    )
    assert "Configured (Vercel AI Gateway API)" in health.stt_health()


def test_setup_decisions_saves_the_key_base_url_and_model(monkeypatch) -> None:
    config.save_config({"DECISION_API_KEY": "keep-key"})
    answers = iter(["https://openrouter.ai/api", "example/decider"])
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(setup.getpass, "getpass", lambda *_args, **_kwargs: "")

    assert main(["setup", "--decisions"]) == 0
    assert config.read_config() == {
        "DECISION_API_KEY": "keep-key",
        "DECISION_BASE_URL": "https://openrouter.ai/api",
        "DECISION_MODEL": "example/decider",
    }


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
