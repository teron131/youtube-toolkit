"""Behavior tests for transcript-guided, bounded, image-only frame selection and publishing."""

from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin
import pytest

from youtube_toolkit.cli import commands
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit import vision
from youtube_toolkit.vision import frames as frame_selection, workflow
from youtube_toolkit.youtube.caption import CaptionTranscriptUnavailable
from youtube_toolkit.youtube.client import VISION_VIDEO_FORMAT_SELECTOR, YtDlpClient
from youtube_toolkit.vision.transcript import TranscriptWindow

VIDEO_ID = "JpSHyEIZ_bo"


def make_frame(timestamp: float, *, value: int = 80) -> frame_selection.SelectedFrame:
    return frame_selection.SelectedFrame(
        timestamp,
        np.full((360, 640, 3), value, dtype=np.uint8),
        None,
        frame_selection.VisualScores(20.0, 0.01, 10.0, 0.0, 0.0),
    )


class NoFaces:
    def detectMultiScale(self, *_args, **_kwargs):
        return np.empty((0, 4), dtype=int)


def chart(heights=(20, 30, 40, 50)) -> np.ndarray:
    image = np.full((360, 640, 3), 240, dtype=np.uint8)
    image[:56] = 40
    for column, height in enumerate(heights):
        image[320 - height * 4 : 320, 60 + column * 140 : 140 + column * 140] = 100
    return image


@pytest.mark.parametrize("captions", [True, False])
@pytest.mark.parametrize("explicit_output", [True, False])
def test_sample_vision_returns_evidence_and_cleans_up(
    monkeypatch, tmp_path, captions, explicit_output
) -> None:
    """Both caption-guided and fallback retrieval preserve artifact cleanup and explain selection."""
    window = TranscriptWindow(5, 25, "A diagram explains the circuit", 0.5, "Circuits")
    selected = make_frame(12)
    selected = frame_selection.SelectedFrame(
        12, selected.image, window if captions else None, selected.visual_scores
    )

    class FakeClient:
        def metadata(self, _ref):
            return {"id": VIDEO_ID, "duration": 120, "heatmap": [{"value": 1}]}

        def download_video_for_vision(self, _ref, path, *, max_bytes):
            assert max_bytes == workflow.MAX_VIDEO_BYTES
            path.write_bytes(b"video")
            return path

    def transcript(_ref, *, options, client):
        assert not options.allow_stt
        if not captions:
            raise CaptionTranscriptUnavailable("No captions")
        return object()

    def select(path, windows, ffmpeg, *, duration, max_frames):
        assert path.exists()
        assert duration == 120
        assert max_frames == 6
        assert windows == ([window] if captions else [])
        return [selected]

    monkeypatch.setattr(workflow, "YtDlpClient", FakeClient)
    monkeypatch.setattr(workflow.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(workflow, "fetch_transcript", transcript)
    monkeypatch.setattr(
        workflow, "select_transcript_windows", lambda _doc, **_kw: [window]
    )
    monkeypatch.setattr(workflow, "select_frames", select)
    monkeypatch.chdir(tmp_path)
    output_options = {"output_dir": tmp_path} if explicit_output else {}
    result = workflow.sample_vision(VIDEO_ID, **output_options)
    assert result["selection_mode"] == ("transcript" if captions else "visual")
    assert bool(result["transcript_warning"]) is not captions
    assert result["frames"] == [selected.to_dict()]
    assert result["timestamps_seconds"] == [12]
    assert result["heatmap_available"] is True
    assert Path(result["collage_paths"][0]).is_file()
    assert not list(Path(result["collage_paths"][0]).parent.glob("work-*"))
    if not explicit_output:
        first_collage = Path(result["collage_paths"][0])
        first_contents = first_collage.read_bytes()
        second = workflow.sample_vision(VIDEO_ID)
        second_collage = Path(second["collage_paths"][0])
        assert first_collage.parent != second_collage.parent
        assert VIDEO_ID in first_collage.parent.name
        assert VIDEO_ID in second_collage.parent.name
        assert first_collage.read_bytes() == first_contents
        assert second_collage.is_file()


@pytest.mark.parametrize("limit", [0, 97, -1])
def test_invalid_budget_fails_before_fetch(monkeypatch, limit) -> None:
    monkeypatch.setattr(workflow, "YtDlpClient", lambda: pytest.fail("no network"))
    with pytest.raises(ValueError, match="max_frames"):
        workflow.sample_vision(VIDEO_ID, max_frames=limit)


def test_sample_vision_rejects_long_video_before_download(
    monkeypatch, tmp_path
) -> None:
    class FakeClient:
        def metadata(self, _ref):
            return {"id": VIDEO_ID, "duration": workflow.MAX_VIDEO_DURATION_SECONDS + 1}

        def download_video_for_vision(self, *_args, **_kwargs):
            pytest.fail("download should not run")

    monkeypatch.setattr(workflow, "YtDlpClient", FakeClient)
    monkeypatch.setattr(workflow.shutil, "which", lambda _: "ffmpeg")
    with pytest.raises(ValueError, match="longer than 2 hours"):
        workflow.sample_vision(VIDEO_ID, output_dir=tmp_path)


def test_transcript_windows_limit_decoding_and_frames_own_pixels(
    monkeypatch, tmp_path
) -> None:
    """Only the shortlisted interval is decoded, and retained frames do not hold whole windows alive."""
    window = TranscriptWindow(100, 120, "Protein structure", 0.7, "Proteins")
    images = np.repeat(chart()[None], 27, axis=0)
    calls = []

    def decode(_path, start, stop, _ffmpeg):
        calls.append((start, stop))
        return images

    monkeypatch.setattr(frame_selection, "decode_window", decode)
    frames = frame_selection.select_frames(
        tmp_path / "v.mp4", [window], "ffmpeg", duration=1000
    )
    assert calls == [(98, 125)]
    assert len(frames) == 1
    assert frames[0].transcript == window
    assert 100 <= frames[0].timestamp_seconds <= 120
    assert frames[0].image.base is None
    assert np.isfinite(frames[0].visual_scores.clarity)
    images[:] = 0
    assert frames[0].image.max() > 0


def test_visual_fallback_spreads_windows_and_drops_static_duplicates(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def decode(_path, start, stop, _ffmpeg):
        calls.append((start, stop))
        return np.repeat(chart()[None], int(stop - start), axis=0)

    monkeypatch.setattr(frame_selection, "decode_window", decode)
    frames = frame_selection.select_frames(
        tmp_path / "v.mp4", [], "ffmpeg", duration=90, max_frames=3
    )
    assert calls == [(0, 35), (28, 65), (58, 90)]
    assert len(frames) == 1
    assert frames[0].transcript is None


def test_visual_selection_preserves_distinct_charts() -> None:
    retained = []
    images = [chart(), chart((10, 20, 35, 60)), chart((60, 40, 25, 10))]
    for image in images:
        candidate = frame_selection.choose_frame(
            np.repeat(image[None], 10, axis=0),
            0,
            0,
            10,
            retained,
            cast(cv2.CascadeClassifier, NoFaces()),
        )
        assert candidate is not None
        retained.append(candidate.thumbnail)
    assert (
        frame_selection.choose_frame(
            np.repeat(images[0][None], 10, axis=0),
            0,
            0,
            10,
            retained,
            cast(cv2.CascadeClassifier, NoFaces()),
        )
        is None
    )


def test_visual_selection_avoids_transient_blank_and_blur() -> None:
    image = chart()
    frames = np.repeat(image[None], 12, axis=0)
    frames[4] = cv2.GaussianBlur(image, (31, 31), 0)
    frames[5] = 255
    candidate = frame_selection.choose_frame(
        frames, 0, 0, 12, [], cast(cv2.CascadeClassifier, NoFaces())
    )
    assert candidate is not None
    assert candidate.index not in (4, 5)


def test_large_portrait_yields_to_a_sustained_alternative() -> None:
    class Faces:
        def detectMultiScale(self, gray, **_kwargs):
            return (
                np.array([[0, 0, 100, 100]])
                if gray.mean() < 150
                else np.empty((0, 4), dtype=int)
            )

    dark = 255 - chart()
    images = np.stack([dark] * 10 + [chart()] * 7)
    candidate = frame_selection.choose_frame(
        images, 0, 0, 17, [], cast(cv2.CascadeClassifier, Faces())
    )
    assert candidate is not None
    assert candidate.index >= 10
    assert candidate.scores.face_area == 0


def test_short_blank_video_still_has_one_representative() -> None:
    images = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    candidate = frame_selection.choose_frame(
        images, 0, 0, 1, [], cast(cv2.CascadeClassifier, NoFaces())
    )
    assert candidate is not None
    assert candidate.index == 0


def test_decode_window_rejects_truncated_rgb(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(frame_selection, "run_ffmpeg", lambda *_args: b"bad")
    with pytest.raises(ValueError, match="incomplete RGB"):
        frame_selection.decode_window(tmp_path / "v.mp4", 0, 10, "ffmpeg")


def test_timestamp_footer_preserves_frame_and_renders_text() -> None:
    """Timestamp rendering adds visible text below rather than over the sampled frame."""
    frame = np.full(
        (frame_selection.FRAME_HEIGHT, frame_selection.FRAME_WIDTH, 3),
        127,
        dtype=np.uint8,
    )

    timestamped = workflow.add_timestamp_footer(frame, 754.0)

    assert timestamped.shape == (
        frame_selection.FRAME_HEIGHT + workflow.TIMESTAMP_FOOTER_HEIGHT,
        frame_selection.FRAME_WIDTH,
        3,
    )
    assert np.array_equal(timestamped[: frame_selection.FRAME_HEIGHT], frame)
    text_color = np.array((211, 216, 224), dtype=np.uint8)
    assert np.any(
        np.all(timestamped[frame_selection.FRAME_HEIGHT :] == text_color, axis=2)
    )


def test_collage_renderer_uses_six_360p_frames_per_progressive_jpeg(
    monkeypatch, tmp_path
) -> None:
    """Publishing consumes retained images without another full-video decode."""
    frames = [make_frame(float(index), value=index * 20) for index in range(7)]
    monkeypatch.setattr(
        frame_selection,
        "run_ffmpeg",
        lambda *_args: pytest.fail("renderer must not decode"),
    )
    paths = workflow.render_timestamped_collages(frames, tmp_path / "vision")
    assert len(paths) == 2
    with Image.open(paths[0]) as image:
        assert image.size == (1298, 1176)
        assert image.info["progressive"] == 1
        assert JpegImagePlugin.get_sampling(image) == 1


def test_download_video_for_vision_uses_bounded_720p_video_only_preset(
    monkeypatch, tmp_path
) -> None:
    """The yt-dlp adapter owns video-only format, size, fragment, and timeout policy."""
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text, check, timeout):
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)
    output_path = tmp_path / "video.mp4"

    assert (
        YtDlpClient().download_video_for_vision(
            VIDEO_ID, output_path, max_bytes=123_456
        )
        == output_path
    )
    command = calls[0]
    assert command[command.index("--format") + 1] == VISION_VIDEO_FORMAT_SELECTOR
    assert command[command.index("--max-filesize") + 1] == "123456"
    assert command[command.index("--concurrent-fragments") + 1] == "1"
    assert VISION_VIDEO_FORMAT_SELECTOR.count("bestvideo") == 2
    assert "+bestaudio" not in VISION_VIDEO_FORMAT_SELECTOR
    assert "--postprocessor-args" not in command
    assert "--merge-output-format" not in command


def test_vision_subprocess_timeouts_become_user_facing_runtime_errors(
    monkeypatch, tmp_path
) -> None:
    """Both external process boundaries translate timeout exceptions for CLI handling."""

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["command"], timeout=1)

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", time_out)
    with pytest.raises(RuntimeError, match="video download for vision timed out"):
        YtDlpClient().download_video_for_vision(
            VIDEO_ID, tmp_path / "video.mp4", max_bytes=123
        )

    monkeypatch.setattr(frame_selection.subprocess, "run", time_out)
    with pytest.raises(RuntimeError, match="FFmpeg vision processing timed out"):
        frame_selection.run_ffmpeg("ffmpeg", ["-version"])


def test_vision_subprocess_start_failures_become_user_facing_runtime_errors(
    monkeypatch, tmp_path
) -> None:
    """Missing or unlaunchable executables retain boundary-specific error context."""

    def fail_start(*_args, **_kwargs):
        raise OSError("cannot execute")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fail_start)
    with pytest.raises(
        RuntimeError, match="Could not start yt-dlp video download for vision"
    ):
        YtDlpClient().download_video_for_vision(
            VIDEO_ID, tmp_path / "video.mp4", max_bytes=123
        )

    monkeypatch.setattr(frame_selection.subprocess, "run", fail_start)
    with pytest.raises(RuntimeError, match="Could not start FFmpeg vision processing"):
        frame_selection.run_ffmpeg("ffmpeg", ["-version"])


def test_video_vision_cli_routes_to_public_extraction(monkeypatch) -> None:
    """The vision command returns the public operation's JSON-ready result."""
    args = build_parser().parse_args(["vision", VIDEO_ID, "--max-frames", "12"])
    expected = {
        "video_id": VIDEO_ID,
        "frame_count": 1,
        "timestamps_seconds": [8.0],
        "collage_paths": ["/example/page1.jpg"],
    }
    monkeypatch.setattr(
        vision,
        "sample_vision",
        lambda video_ref, *, max_frames: (
            expected if video_ref == VIDEO_ID and max_frames == 12 else {}
        ),
    )

    assert args.handler is commands.VideoCommands.vision
    assert commands.VideoCommands.vision(args) == expected
