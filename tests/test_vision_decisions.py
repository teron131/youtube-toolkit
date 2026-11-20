"""Frame ranking with the decision model."""

import numpy as np
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
from youtube_toolkit.vision import decisions, workflow
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


def frame(tmp_path, timestamp: float) -> StoredFrame:
    path = tmp_path / f"frame-{timestamp}.jpg"
    # Noise makes every frame unlike the others, so picking skips none for looking alike.
    pixels = np.random.default_rng(round(timestamp * 1000)).integers(
        0, 256, (9, 16, 3), dtype=np.uint8
    )
    Image.fromarray(pixels).save(path)
    return StoredFrame(timestamp, path, VisualScores(1.0, 0.0, 0.0))


def rank(frames, document=None):
    # No limit or gap that would drop a ranked frame, so the picks show the whole ranking; nearness reaches about two seconds.
    return workflow._rank_by_decisions(
        frames, document, CONFIG, duration=200, max_frames=96, min_gap_seconds=0
    )


def result(cost=0.00001, **answers):
    return DecisionResult(
        {name: NoulAnswer(noul=value) for name, value in answers.items()},
        100,
        2,
        cost,
        0.4,
    )


def picture(cost=0.00001, loaded=1.0, **answers):
    return result(cost, loaded=loaded, **answers)


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
    # 80 and 90 show one item, 100 is a talking head too plain for speech to lift, 110 footage of what is described; 195 has only music near it.
    frames = [frame(tmp_path, t) for t in (80, 90, 100, 110, 195)]
    data = [decisions._asset(item.image_path)["data_base64"] for item in frames]
    pictures = {
        data[0]: picture(has_information=0.9, visual=0.2, impressive=0.3),
        data[1]: picture(has_information=0.8, visual=0.2, impressive=0.3),
        data[2]: picture(has_information=0.01, visual=0.02, impressive=0.0),
        data[3]: picture(has_information=0.05, visual=0.2, impressive=0.1, cost=None),
        data[4]: picture(visual=0.8, impressive=0.6),
    }
    illustrative = {data[0]: 0.1, data[1]: 0.8, data[3]: 0.6}
    same = {(data[0], data[1]): 0.9}
    asked: dict[str, list] = {}

    def decide_each(config, *, questions, inputs):
        assert config == CONFIG
        (name,) = {*questions} & {"has_information", "same_item", "illustrative"}
        asked.setdefault(name, []).append(inputs)
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
    picks, requests = rank(frames, DOCUMENT)

    # The picture alone would keep 80 for the item and drop 110.
    assert [
        (pick.timestamp_seconds, round(relevance["score"], 2))
        for pick, relevance in picks
    ] == [(90, 1.0), (195, 0.8), (110, 0.35)]
    assert picks[1][1] == {
        "selection_rank": 2,
        "score": 0.8,
        "has_information": 0.0,
        "visual": 0.8,
        "impressive": 0.6,
        "loaded": 1.0,
        "illustrative": None,
    }
    assert [image(item) for item in asked["illustrative"][0]] == [
        data[0],
        data[1],
        data[3],
    ]
    assert asked["illustrative"][0][0].state[0].startswith("Chapter: Optics\n")
    # The talking head drops out, so no pair includes it.
    assert [
        (image(item, "first"), image(item, "second")) for item in asked["same_item"][0]
    ] == [(data[0], data[1]), (data[3], data[4])]
    assert requests.usage == decisions.DecisionUsage(1200, 24, None, 4.8)
    assert requests.warning is None


@pytest.mark.parametrize(
    ("times", "tie", "even_gap_seconds"),
    [((10, 20, 30), 0.5, 1), ((10, 20, 300), None, 100)],
    ids=["another view of a ranked item", "close in time to a ranked item"],
)
def test_a_frame_waits_behind_a_lower_scored_one_that_adds_more(
    tmp_path, times, tie, even_gap_seconds
) -> None:
    # The second frame scores above the third, but it likely repeats the first or sits right beside it.
    frames = [frame(tmp_path, t) for t in times]
    passing = {
        index: decisions.FrameDecision(score, score, 0.0, 0.0, 1.0, None)
        for index, score in enumerate((0.9, 0.85, 0.7))
    }
    order = decisions.rank_items(
        decisions.Items([0, 1, 2], [tie, None]),
        passing,
        frames,
        even_gap_seconds=even_gap_seconds,
    )
    assert order == [0, 2, 1]


@pytest.mark.parametrize(
    ("same_item", "picked"),
    [(0.9, [10, 100]), (0.7, [10, 200, 100])],
    ids=["a sure repeat", "an answer that would only join neighbours"],
)
def test_a_scene_shown_again_later_is_picked_once(
    monkeypatch, tmp_path, same_item, picked
) -> None:
    # 10 and 200 show one scene, with a different scene between them.
    frames = [frame(tmp_path, t) for t in (10, 100, 200)]
    data = [decisions._asset(item.image_path)["data_base64"] for item in frames]
    information = dict(zip(data, (0.9, 0.8, 0.85), strict=True))
    asked: list[tuple[str, str]] = []

    def decide_each(config, *, questions, inputs):
        if "same_item" not in questions:
            return [
                picture(
                    has_information=information[image(item)], visual=0.0, impressive=0.0
                )
                for item in inputs
            ]
        pairs = [(image(item, "first"), image(item, "second")) for item in inputs]
        asked.extend(pairs)
        return [
            result(same_item=same_item if pair == (data[0], data[2]) else 0.0)
            for pair in pairs
        ]

    monkeypatch.setattr(decisions, "decide_each", decide_each)
    picks, _ = rank(frames)
    # Neighbours are compared once already, so only the far pair is a lookalike.
    assert asked == [(data[0], data[1]), (data[1], data[2]), (data[0], data[2])]
    assert [pick.timestamp_seconds for pick, _ in picks] == picked


@pytest.mark.parametrize(
    "first",
    [
        {"has_information": 0.45, "visual": 0.1, "impressive": 0.0, "loaded": 1.0},
        {"has_information": 0.9, "visual": 0.9, "impressive": 0.9, "loaded": 0.1},
    ],
    ids=["doubtful answers clear the score line", "a pane is still loading"],
)
def test_a_frame_drops_out_without_the_yes_it_needs_however_it_scores(first) -> None:
    # Both first frames score above MIN_SCORE.
    second = {"has_information": 0.9, "visual": 0.1, "impressive": 0.0, "loaded": 1.0}
    assert list(decisions.score_frames([first, second], {})) == [1]


def test_a_video_where_no_frame_stays_gets_no_picks(monkeypatch, tmp_path) -> None:
    frames = [frame(tmp_path, t) for t in (10, 20)]
    monkeypatch.setattr(
        decisions,
        "decide_each",
        lambda *_args, **_kwargs: [picture(visual=0.0), picture(visual=0.0)],
    )
    picks, _ = rank(frames)
    assert picks == []


def test_a_few_failed_requests_cost_their_frames_but_more_fail_the_ranking(
    monkeypatch, tmp_path
) -> None:
    failed = DecisionError("OpenRouter decision request failed (HTTP 503)")
    frames = [frame(tmp_path, float(t)) for t in range(20)]
    shown = picture(has_information=0.9, visual=0.0, impressive=0.0)
    same_item_steps: list[int] = []

    def decide_each(config, *, questions, inputs):
        if "same_item" in questions:
            # Only the neighbours' step, the first to ask same_item, loses a request.
            same_item_steps.append(len(inputs))
            first = failed if len(same_item_steps) == 1 else result(same_item=0.0)
            return [first, *[result(same_item=0.0)] * (len(inputs) - 1)]
        return [failed, *[shown] * (len(inputs) - 1)]

    monkeypatch.setattr(decisions, "decide_each", decide_each)
    picks, requests = rank(frames)
    assert sorted(pick.timestamp_seconds for pick, _ in picks) == list(range(1, 20))
    assert requests.warning == (
        "1 of 20 frame requests failed after repeated decision host errors, so those frames were left out of the ranking. "
        "1 of 18 same-item requests failed after repeated decision host errors, so those neighbours were kept as separate items."
    )

    monkeypatch.setattr(
        decisions,
        "decide_each",
        lambda *_args, **_kwargs: [failed, failed, failed, *[shown] * 17],
    )
    with pytest.raises(DecisionError, match=r"3 of 20 frame requests failed.*HTTP 503"):
        rank(frames)
