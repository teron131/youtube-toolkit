"""Collage layout and encoding."""

import numpy as np
from PIL import Image, JpegImagePlugin
import pytest

from youtube_toolkit import ffmpeg_utils
from youtube_toolkit.config import (
    DEFAULT_VISION_COLUMNS,
    DEFAULT_VISION_RESOLUTION,
    DEFAULT_VISION_ROWS,
    VISION_RESOLUTIONS,
)
from youtube_toolkit.vision import candidates as frame_selection, collages


def make_frame(timestamp: float, *, value: int = 80) -> frame_selection.DecodedFrame:
    return frame_selection.DecodedFrame(
        timestamp,
        np.full(
            (frame_selection.CANDIDATE_HEIGHT, frame_selection.CANDIDATE_WIDTH, 3),
            value,
            dtype=np.uint8,
        ),
        frame_selection.VisualScores(20.0, 0.01, 0.0),
    )


def test_timestamp_footer_preserves_frame_and_renders_text() -> None:
    frame = np.full(
        (frame_selection.CANDIDATE_HEIGHT, frame_selection.CANDIDATE_WIDTH, 3),
        127,
        dtype=np.uint8,
    )

    timestamped = collages.add_timestamp_footer(frame, 754.0)

    assert timestamped.shape == (
        frame_selection.CANDIDATE_HEIGHT + collages.TIMESTAMP_FOOTER_HEIGHT,
        frame_selection.CANDIDATE_WIDTH,
        3,
    )
    assert np.array_equal(timestamped[: frame_selection.CANDIDATE_HEIGHT], frame)
    text_color = np.array((211, 216, 224), dtype=np.uint8)
    assert np.any(
        np.all(timestamped[frame_selection.CANDIDATE_HEIGHT :] == text_color, axis=2)
    )


def test_collage_renderer_uses_six_540p_frames_per_progressive_jpeg(
    monkeypatch, tmp_path
) -> None:
    frames = [make_frame(float(index), value=index * 20) for index in range(7)]
    monkeypatch.setattr(
        ffmpeg_utils,
        "run_ffmpeg",
        lambda *_args: pytest.fail("renderer must not decode"),
    )
    paths = collages.render_collages(
        frames,
        tmp_path / "vision",
        rows=DEFAULT_VISION_ROWS,
        columns=DEFAULT_VISION_COLUMNS,
        frame_size=VISION_RESOLUTIONS[DEFAULT_VISION_RESOLUTION],
        timestamp_annotation=True,
    )
    assert len(paths) == 2
    with Image.open(paths[0]) as image:
        assert image.size == (1938, 1716)
        assert image.info["progressive"] == 1
        assert JpegImagePlugin.get_sampling(image) == 1


@pytest.mark.parametrize(
    ("rows", "columns", "frame_size", "timestamp_annotation", "pages", "page_size"),
    [
        (1, 1, (640, 360), True, 7, (652, 396)),
        (1, 1, (640, 360), False, 7, (652, 372)),
        (3, 3, (960, 540), True, 1, (2904, 1716)),
    ],
)
def test_collage_pages_hold_the_grid_at_full_frame_size(
    tmp_path, rows, columns, frame_size, timestamp_annotation, pages, page_size
) -> None:
    frames = [make_frame(float(index)) for index in range(7)]
    paths = collages.render_collages(
        frames,
        tmp_path / "vision",
        rows=rows,
        columns=columns,
        frame_size=frame_size,
        timestamp_annotation=timestamp_annotation,
    )
    assert len(paths) == pages
    with Image.open(paths[0]) as image:
        assert image.size == page_size
