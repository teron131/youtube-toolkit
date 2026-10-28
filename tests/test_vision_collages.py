"""Collages keep each frame intact above its timestamp and pack six 540p frames per progressive JPEG page."""

import numpy as np
from PIL import Image, JpegImagePlugin
import pytest

from youtube_toolkit import ffmpeg_utils
from youtube_toolkit.vision import candidates as frame_selection, collages


def make_frame(timestamp: float, *, value: int = 80) -> frame_selection.DecodedFrame:
    return frame_selection.DecodedFrame(
        timestamp,
        np.full(
            (ffmpeg_utils.FRAME_HEIGHT, ffmpeg_utils.FRAME_WIDTH, 3),
            value,
            dtype=np.uint8,
        ),
        frame_selection.VisualScores(20.0, 0.01, 0.0),
    )


def test_timestamp_footer_preserves_frame_and_renders_text() -> None:
    frame = np.full(
        (ffmpeg_utils.FRAME_HEIGHT, ffmpeg_utils.FRAME_WIDTH, 3),
        127,
        dtype=np.uint8,
    )

    timestamped = collages.add_timestamp_footer(frame, 754.0)

    assert timestamped.shape == (
        ffmpeg_utils.FRAME_HEIGHT + collages.TIMESTAMP_FOOTER_HEIGHT,
        ffmpeg_utils.FRAME_WIDTH,
        3,
    )
    assert np.array_equal(timestamped[: ffmpeg_utils.FRAME_HEIGHT], frame)
    text_color = np.array((211, 216, 224), dtype=np.uint8)
    assert np.any(
        np.all(timestamped[ffmpeg_utils.FRAME_HEIGHT :] == text_color, axis=2)
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
    paths = collages.render_timestamped_collages(frames, tmp_path / "vision")
    assert len(paths) == 2
    with Image.open(paths[0]) as image:
        assert image.size == (1938, 1716)
        assert image.info["progressive"] == 1
        assert JpegImagePlugin.get_sampling(image) == 1
