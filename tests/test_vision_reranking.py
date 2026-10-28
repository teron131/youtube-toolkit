"""Frame reranking matches only nearby speech, scores it against the whole transcript, fuses scales, and spaces and deduplicates selected frames."""

import math

import numpy as np
import pytest

from youtube_toolkit.vision.reranking import rerank_frames
from youtube_toolkit.vision.text import TextTarget


def target(text="Topic", *, level="paragraph", start=0, end=60):
    return TextTarget(level, start, end, text)


def rank(images, texts, targets, times, *, threshold=0.3, gap=10, limit=96):
    return rerank_frames(
        np.asarray(images, dtype=float),
        np.asarray(texts, dtype=float),
        targets,
        times,
        max_frames=limit,
        min_similarity=threshold,
        min_gap_seconds=gap,
    )


def test_frames_match_text_from_two_seconds_before_it_starts_to_five_after_it_ends():
    targets = [target("Intro", end=10), target("Lens diagram", start=600, end=620)]
    images = [[0.6, 0.8], [0, 1], [0.6, 0.8], [0, 1]]
    result = rank(images, [[1, 0], [0, 1]], targets, [15, 598, 15.1, 597.9], gap=0)
    assert {row.index: row.matches[0].target for row in result} == {
        0: targets[0],
        1: targets[1],
    }


def test_relative_relevance_prefers_specific_match_over_generic_high_scores():
    generic = [0.65, 0.65, math.sqrt(1 - 2 * 0.65**2)]
    specific = [0.60, 0.10, math.sqrt(1 - 0.60**2 - 0.10**2)]
    result = rank(
        [generic, specific],
        [[1, 0, 0], [0, 1, 0]],
        [target("Lens"), target("Laser")],
        [0, 20],
    )
    assert [row.index for row in result] == [1, 0]
    assert result[0].matches[0].cosine < result[1].matches[0].cosine
    assert result[0].matches[0].relative_score == pytest.approx(0.25)


def test_repeated_target_strings_keep_background_and_report_the_nearby_occurrence():
    targets = [target("Lens"), target("Laser")]
    later = [target("Lens", start=100, end=110), target("Lens", start=200, end=210)]
    images = [[1, 0], [math.sqrt(0.5), math.sqrt(0.5)]]
    texts = [[1, 0], [0, 1], [1, 0], [1, 0]]
    original = rank(images, texts[:2], targets, [0, 20])
    assert rank(images, texts, [*targets, *later], [0, 20]) == original
    nearby = rank([[1, 0]], texts, [*targets, *later], [105])
    assert nearby[0].matches[0].target == later[0]


def test_raw_cutoff_keeps_eligible_zero_relative_matches():
    result = rank(
        [[0.4, math.sqrt(0.84)], [0.2, math.sqrt(0.96)]],
        [[1, 0]],
        [target()],
        [0, 20],
    )
    assert [row.index for row in result] == [0]
    assert result[0].matches[0].cosine == pytest.approx(0.4)
    assert result[0].matches[0].relative_score == 0


def test_equal_relative_scores_receive_competition_ranks():
    result = rank(
        [
            [0.75, -0.25, math.sqrt(0.375)],
            [1, 0, 0],
            [math.sqrt(0.5), math.sqrt(0.5), 0],
        ],
        [[1, 0, 0], [0, 1, 0]],
        [target("Lens"), target("Laser")],
        [20, 0, 40],
    )
    assert [row.index for row in result] == [1, 0, 2]
    assert [row.matches[0].rank for row in result] == [1, 1, 3]
    assert [row.score for row in result] == pytest.approx([1 / 61, 1 / 61, 1 / 63])


def test_rrf_has_one_vote_per_surviving_scale():
    targets = [target(level=level) for level in ("keyword", "sentence", "paragraph")]
    result = rank([[1, 0]], [[1, 0], [1, 0], [0, 1]], targets, [0])
    assert result[0].score == pytest.approx(2 / 61)
    assert [match.target.level for match in result[0].matches] == [
        "keyword",
        "sentence",
    ]


@pytest.mark.parametrize(("targets", "texts"), [([], []), ([target()], [[0, 1]])])
def test_empty_eligibility_stays_empty(targets, texts):
    assert rank([[1, 0]], texts, targets, [0]) == []


def test_spacing_checks_actual_distances_across_bucket_boundaries():
    images = [[0.6, 0.8, 0], [0.6, 0, 0.8], [0.6, -0.8, 0]]
    result = rank(images, [[1, 0, 0]], [target()], [9.9, 10.1, 20])
    assert [row.index for row in result] == [0, 2]
    exact = rank(images, [[1, 0, 0]], [target()], [0, 10, 20], limit=2)
    assert [row.index for row in exact] == [0, 1]


def test_selection_drops_embedding_duplicates_of_selected_frames_minutes_apart():
    near = [0.99, math.sqrt(1 - 0.99**2)]
    result = rank(
        [[1, 0], near, [0.8, 0.6]], [[1, 0]], [target(end=300)], [0, 100, 200]
    )
    assert [row.index for row in result] == [0, 2]


def test_vector_rows_must_align_with_timestamps_and_targets():
    with pytest.raises(ValueError, match="align"):
        rank([[1, 0]], [[1, 0]], [target()], [0, 20])
