"""Frame ranking with the decision model."""

from PIL import Image
import pytest
from typesafe_sdk import NoulAnswer

from youtube_toolkit.config import DecisionConfig
from youtube_toolkit.decisions import DecisionError, DecisionResult
from youtube_toolkit.schemas import (
    TranscriptDocument,
    TranscriptSegment,
    VideoChapter,
    VideoMetadata,
)
from youtube_toolkit.vision import decisions
from youtube_toolkit.vision.candidates import StoredFrame, VisualScores

CONFIG = DecisionConfig("https://decisions.example", "key", "model-id")
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
        TranscriptSegment(5, 10, "Far before."),
        TranscriptSegment(70, 80, "The lens bends light."),
        TranscriptSegment(95, 100, "Here is the ray diagram."),
        TranscriptSegment(150, 160, "Far after."),
        TranscriptSegment(185, 190, "[Music]"),
    ],
)


def frame(tmp_path, timestamp: float, shade: int) -> StoredFrame:
    path = tmp_path / f"frame-{timestamp}.jpg"
    Image.new("RGB", (16, 9), (shade, shade, shade)).save(path)
    return StoredFrame(timestamp, path, VisualScores(1.0, 0.0, 0.0))


def result(cost=0.00001, **answers):
    return DecisionResult(
        {name: NoulAnswer(noul=value) for name, value in answers.items()},
        100,
        2,
        cost,
        0.4,
    )


def image(item, key=decisions.FRAME_ASSET) -> str:
    return item.assets[key]["data_base64"]


def test_context_marks_the_frame_among_lines_within_thirty_seconds() -> None:
    text = decisions._context(DOCUMENT, 90, decisions._nearby(DOCUMENT.segments, 90))
    assert text.startswith("Chapter: Optics\n")
    assert text.endswith(
        f"The lens bends light. {decisions.MARKER} Here is the ray diagram."
    )
    assert "Far" not in text


def test_speech_lifts_footage_of_what_is_described_and_chooses_within_an_item(
    monkeypatch, tmp_path
) -> None:
    # 80 and 90 show one item, 100 is a talking head, 110 footage of what is described; 195 has only music near it.
    frames = [
        frame(tmp_path, t, shade)
        for t, shade in ((80, 0), (90, 60), (100, 120), (110, 180), (195, 240))
    ]
    data = [decisions._asset(item.image_path)["data_base64"] for item in frames]
    pictures = {
        data[0]: result(has_information=0.9, visual=0.2, impressive=0.3),
        data[1]: result(has_information=0.8, visual=0.2, impressive=0.3),
        data[2]: result(has_information=0.1, visual=0.2, impressive=0.0),
        data[3]: result(has_information=0.05, visual=0.2, impressive=0.1, cost=None),
        data[4]: result(visual=0.8, impressive=0.6),
    }
    illustrative = {data[0]: 0.1, data[1]: 0.8, data[2]: 0.1, data[3]: 0.6}
    same = {(data[0], data[1]): 0.9}
    asked: dict[str, list] = {}

    def decide_each(config, *, questions, inputs):
        assert config == CONFIG
        (name,) = {*questions} & {"has_information", "same_item", "illustrative"}
        asked[name] = inputs
        if name == "has_information":
            return [pictures[image(item)] for item in inputs]
        if name == "same_item":
            return [
                result(
                    same_item=same.get(
                        (image(item, "first"), image(item, "second")), 0.1
                    )
                )
                for item in inputs
            ]
        return [result(illustrative=illustrative[image(item)]) for item in inputs]

    monkeypatch.setattr(decisions, "decide_each", decide_each)
    ranking = decisions.rank_frames(frames, DOCUMENT, CONFIG)

    # The picture alone would keep 80 for the item and drop 110.
    assert [(index, round(item.score, 2)) for index, item in ranking.frames] == [
        (1, 1.0),
        (4, 0.8),
        (3, 0.35),
    ]
    assert ranking.frames[1][1].illustrative is None
    assert [image(item) for item in asked["illustrative"]] == data[:4]
    assert asked["illustrative"][0].state[0].startswith("Chapter: Optics\n")
    # The talking head drops out, so no pair includes it.
    assert [
        (image(item, "first"), image(item, "second")) for item in asked["same_item"]
    ] == [(data[0], data[1]), (data[3], data[4])]
    assert ranking.usage == decisions.DecisionUsage(1100, 22, None, 4.4)
    assert ranking.warning is None


def test_a_few_failed_requests_cost_their_frames_but_more_fail_the_ranking(
    monkeypatch, tmp_path
) -> None:
    failed = DecisionError("OpenRouter decision request failed (HTTP 503)")
    frames = [frame(tmp_path, float(t), t * 12) for t in range(20)]
    shown = result(has_information=0.9, visual=0.0, impressive=0.0)

    def decide_each(config, *, questions, inputs):
        if "same_item" in questions:
            return [failed, *[result(same_item=0.0)] * (len(inputs) - 1)]
        return [failed, *[shown] * (len(inputs) - 1)]

    monkeypatch.setattr(decisions, "decide_each", decide_each)
    ranking = decisions.rank_frames(frames, None, CONFIG)
    assert sorted(index for index, _ in ranking.frames) == list(range(1, 20))
    assert ranking.warning == (
        "1 of 20 frame requests failed after repeated decision host errors, so those frames were left out of the ranking. "
        "1 of 18 same-item requests failed after repeated decision host errors, so those neighbours were kept as separate items."
    )

    monkeypatch.setattr(
        decisions,
        "decide_each",
        lambda *_args, **_kwargs: [failed, failed, failed, *[shown] * 17],
    )
    with pytest.raises(DecisionError, match=r"3 of 20 frame requests failed.*HTTP 503"):
        decisions.rank_frames(frames, None, CONFIG)
