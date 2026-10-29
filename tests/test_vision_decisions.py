"""Decision scoring shows each frame with the speech around it, combines added information and illustration into one score, and fails whole rather than ranking a partial set."""

from PIL import Image
import pytest

from youtube_toolkit.config import DecisionConfig
from youtube_toolkit.decisions import DecisionError, DecisionResult, image_asset
from youtube_toolkit.schemas import (
    TranscriptChunk,
    TranscriptDocument,
    VideoChapter,
    VideoMetadata,
)
from youtube_toolkit.vision import decisions
from youtube_toolkit.vision.candidates import StoredFrame, VisualScores

CONFIG = DecisionConfig("https://decisions.example/v1", "key", "model-id")
DOCUMENT = TranscriptDocument(
    VideoMetadata(
        "JpSHyEIZ_bo",
        "video",
        duration_seconds=200,
        chapters=[VideoChapter("Intro", 0, 60), VideoChapter("Optics", 60, None)],
    ),
    "en",
    "subtitles",
    [
        TranscriptChunk(5, 10, "Far before."),
        TranscriptChunk(70, 80, "The lens bends light."),
        TranscriptChunk(95, 100, "Here is the ray diagram."),
        TranscriptChunk(150, 160, "Far after."),
    ],
)


def frame(tmp_path, timestamp: float, shade: int = 0) -> StoredFrame:
    path = tmp_path / f"frame-{timestamp}.jpg"
    Image.new("RGB", (16, 9), (shade, shade, shade)).save(path)
    return StoredFrame(timestamp, path, VisualScores(1.0, 0.0, 0.0))


def result(has_information, unspoken, illustrative, *, cost=0.00001):
    answers = {
        name: {"type": "noul", "noul": value}
        for name, value in (
            ("has_information", has_information),
            ("unspoken", unspoken),
            ("illustrative", illustrative),
        )
        if value is not None
    }
    return DecisionResult(answers, 100, cost)


def test_context_marks_the_frame_among_lines_within_thirty_seconds() -> None:
    text = decisions._context(DOCUMENT, 90)
    assert text.startswith("Chapter: Optics\n")
    assert text.endswith(
        f"The lens bends light. {decisions.MARKER} Here is the ray diagram."
    )
    assert "Far" not in text


def test_score_takes_information_the_speech_leaves_out_or_weighted_illustration(
    monkeypatch, tmp_path
) -> None:
    frames = [frame(tmp_path, t, shade) for t, shade in ((10, 0), (20, 120), (30, 250))]
    # Requests run concurrently, so each result is keyed by the frame image it answers.
    results = {
        image_asset(frames[0].image_path)["data_base64"]: result(0.9, 0.8, 0.1),
        image_asset(frames[1].image_path)["data_base64"]: result(0.9, 0.1, 0.9),
        image_asset(frames[2].image_path)["data_base64"]: result(
            None, 0.9, 0.0, cost=None
        ),
    }
    calls = []

    def decide_each(config, *, questions, inputs):
        calls.append((config, questions, inputs))
        return [
            results[item.assets[decisions.FRAME_ASSET]["data_base64"]]
            for item in inputs
        ]

    monkeypatch.setattr(decisions, "decide_each", decide_each)
    scored, usage = decisions.score_frames(frames, DOCUMENT, CONFIG)

    assert [item and round(item.score, 2) for item in scored] == [0.72, 0.36, 0.0]
    assert usage == decisions.DecisionUsage(300, None)
    config, questions, inputs = calls[0]
    assert config == CONFIG
    assert inputs[0].state[1] == decisions.FRAME_ASSET
    assert {q["type"] for q in questions.values()} == {"noul"}


def test_a_few_failed_frames_go_unscored_but_more_fail_the_scoring(
    monkeypatch, tmp_path
) -> None:
    failed = DecisionError("OpenRouter decision request failed (HTTP 503)")
    frames = [frame(tmp_path, float(t)) for t in range(10)]

    monkeypatch.setattr(
        decisions,
        "decide_each",
        lambda *_args, **_kwargs: [failed, *[result(0.9, 0.9, 0.0)] * 9],
    )
    scored, usage = decisions.score_frames(frames, DOCUMENT, CONFIG)
    assert scored[0] is None
    assert all(item is not None for item in scored[1:])
    assert usage.input_tokens == 900

    monkeypatch.setattr(
        decisions,
        "decide_each",
        lambda *_args, **_kwargs: [failed, failed, *[result(0.9, 0.9, 0.0)] * 8],
    )
    with pytest.raises(DecisionError, match=r"2 of 10 frame requests failed.*HTTP 503"):
        decisions.score_frames(frames, DOCUMENT, CONFIG)
