"""Behavior tests for bounded YouTube vision extraction."""

import argparse
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, JpegImagePlugin
import pytest

from youtube_toolkit.cli import commands
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.youtube import vision
from youtube_toolkit.youtube.client import VISION_VIDEO_FORMAT_SELECTOR, YtDlpClient

VIDEO_ID = "JpSHyEIZ_bo"


def test_sample_vision_returns_collages_and_removes_temporary_media(
    monkeypatch,
    tmp_path,
) -> None:
    """The public operation keeps final collages while removing downloaded work files."""

    class FakeClient:
        def metadata(self, video_ref: str) -> dict[str, object]:
            assert video_ref == VIDEO_ID
            return {
                "id": VIDEO_ID,
                "duration": 120,
                "heatmap": [
                    {"start_time": 0, "end_time": 10, "value": 0.1},
                    {"start_time": 10, "end_time": 20, "value": 0.9},
                    {"start_time": 20, "end_time": 30, "value": 0.2},
                ],
            }

        def download_video_for_vision(self, video_ref: str, output_path: Path, *, max_bytes: int) -> Path:
            assert video_ref == VIDEO_ID
            assert max_bytes == vision.MAX_VIDEO_BYTES
            output_path.write_bytes(b"video")
            return output_path

    selected = [
        vision.ScoredFrame(index=8, timestamp_seconds=8.0, score=0.1),
        vision.ScoredFrame(index=20, timestamp_seconds=20.0, score=0.1),
    ]

    def fake_render(
        video_path: Path,
        frames: list[vision.ScoredFrame],
        output_stem: Path,
        ffmpeg_path: str,
    ) -> list[Path]:
        assert video_path.exists()
        assert frames == selected
        assert ffmpeg_path == "/usr/local/bin/ffmpeg"
        page = output_stem.with_name("vision_page1.jpg")
        page.write_bytes(b"jpeg")
        return [page]

    monkeypatch.setattr(vision, "YtDlpClient", FakeClient)
    monkeypatch.setattr(vision.shutil, "which", lambda _name: "/usr/local/bin/ffmpeg")
    monkeypatch.setattr(vision, "select_frames", lambda _path, _heatmap, _ffmpeg: selected)
    monkeypatch.setattr(vision, "render_timestamped_collages", fake_render)

    output_dir = tmp_path / "vision"
    result = vision.sample_vision(VIDEO_ID, output_dir=output_dir)

    assert result == {
        "video_id": VIDEO_ID,
        "duration_seconds": 120.0,
        "heatmap_available": True,
        "frame_count": 2,
        "timestamps_seconds": [8.0, 20.0],
        "collage_paths": [str((output_dir / f"{VIDEO_ID}_vision_page1.jpg").resolve())],
    }
    assert (output_dir / f"{VIDEO_ID}_vision_page1.jpg").read_bytes() == b"jpeg"
    assert not list(output_dir.glob("work-*"))
    assert not list(output_dir.glob("*.mp4"))


def test_sample_vision_rejects_long_video_before_download(monkeypatch, tmp_path) -> None:
    """Metadata preflight blocks an oversized runtime before media acquisition starts."""

    class FakeClient:
        def metadata(self, _video_ref: str) -> dict[str, object]:
            return {
                "id": VIDEO_ID,
                "duration": vision.MAX_VIDEO_DURATION_SECONDS + 1,
            }

        def download_video_for_vision(self, *_args, **_kwargs) -> Path:
            pytest.fail("download should not run")

    monkeypatch.setattr(vision, "YtDlpClient", FakeClient)
    monkeypatch.setattr(vision.shutil, "which", lambda _name: "/usr/local/bin/ffmpeg")

    with pytest.raises(ValueError, match="longer than 2 hours"):
        vision.sample_vision(VIDEO_ID, output_dir=tmp_path)


def test_replay_regions_use_local_peaks_above_video_percentile() -> None:
    """Replay selection retains qualifying local maxima rather than one global maximum."""
    bins = [
        vision.HeatmapBin(0, 10, 0.0),
        vision.HeatmapBin(10, 20, 0.8),
        vision.HeatmapBin(20, 30, 0.1),
        vision.HeatmapBin(30, 40, 0.7),
        vision.HeatmapBin(40, 50, 0.2),
    ]

    regions = vision.select_replay_regions(bins)

    assert regions == [(10, 20), (30, 40)]


@pytest.mark.parametrize(
    "raw_heatmap",
    [
        [{"start_time": 10, "end_time": 5, "value": 0.5}],
        [{"start_time": 0, "end_time": 5, "value": float("nan")}],
    ],
)
def test_heatmap_bins_reject_invalid_intervals_and_values(raw_heatmap) -> None:
    """Replay metadata must contain finite forward-moving intervals."""
    with pytest.raises(ValueError, match="invalid replay heatmap"):
        vision.heatmap_bins(raw_heatmap)


def test_selection_falls_back_when_no_candidate_source_produces_a_frame(monkeypatch, tmp_path) -> None:
    """A successfully decoded video always contributes at least one visual frame."""
    frame = vision.ScoredFrame(index=0, timestamp_seconds=0.0, score=0.0)
    monkeypatch.setattr(vision, "scene_scores", lambda _path, _ffmpeg: [frame])
    monkeypatch.setattr(vision, "blur_scores", lambda _path, _ffmpeg: [2.0])
    monkeypatch.setattr(
        vision,
        "extract_thumbnails",
        lambda _path, _count, _ffmpeg: np.zeros((1, vision.THUMBNAIL_HEIGHT, vision.THUMBNAIL_WIDTH), dtype=np.uint8),
    )
    monkeypatch.setattr(vision, "stationary_midpoints", lambda _path, _frames, _ffmpeg: [])

    assert vision.select_frames(tmp_path / "video.mp4", [], "ffmpeg") == [frame]


def test_ffmpeg_metadata_parser_rejects_incomplete_output() -> None:
    """All one-FPS metric readers reject malformed FFmpeg frame metadata consistently."""
    with pytest.raises(ValueError, match=r"incomplete lavfi\.blur metadata"):
        vision.parse_frame_metadata(b"frame:0 pts_time:0\n", "lavfi.blur")


def test_frame_limit_evenly_preserves_full_chronological_span() -> None:
    """The sanity cap uses no importance ranking and retains both endpoints."""
    frames = [vision.ScoredFrame(index=index, timestamp_seconds=float(index), score=0.0) for index in range(150)]

    selected = vision.limit_frames(frames)

    assert len(selected) == 96
    assert selected[0] == frames[0]
    assert selected[-1] == frames[-1]
    assert selected == sorted(selected, key=lambda frame: frame.timestamp_seconds)


def test_candidate_selection_discards_perceptual_duplicates() -> None:
    """Candidate selection retains chronology while dropping repeated visual content."""
    frames = [
        vision.ScoredFrame(index=0, timestamp_seconds=0.0, score=0.0),
        vision.ScoredFrame(index=1, timestamp_seconds=1.0, score=0.0),
        vision.ScoredFrame(index=2, timestamp_seconds=2.0, score=0.0),
    ]
    thumbnails = np.stack(
        [
            np.zeros((vision.THUMBNAIL_HEIGHT, vision.THUMBNAIL_WIDTH), dtype=np.uint8),
            np.zeros((vision.THUMBNAIL_HEIGHT, vision.THUMBNAIL_WIDTH), dtype=np.uint8),
            np.full((vision.THUMBNAIL_HEIGHT, vision.THUMBNAIL_WIDTH), 255, dtype=np.uint8),
        ]
    )

    selected = vision.add_distinct_candidates([], frames, thumbnails, similarity_threshold=0.95)

    assert selected == [frames[0], frames[2]]


def test_timestamp_footer_preserves_frame_and_renders_text() -> None:
    """Timestamp rendering adds visible text below rather than over the sampled frame."""
    frame = np.full((vision.FRAME_HEIGHT, vision.FRAME_WIDTH, 3), 127, dtype=np.uint8)

    timestamped = vision.add_timestamp_footer(frame, 754.0)

    assert timestamped.shape == (vision.FRAME_HEIGHT + vision.TIMESTAMP_FOOTER_HEIGHT, vision.FRAME_WIDTH, 3)
    assert np.array_equal(timestamped[: vision.FRAME_HEIGHT], frame)
    text_color = np.array((211, 216, 224), dtype=np.uint8)
    assert np.any(np.all(timestamped[vision.FRAME_HEIGHT :] == text_color, axis=2))


def test_collage_renderer_uses_six_360p_frames_per_progressive_jpeg(monkeypatch, tmp_path) -> None:
    """The artifact contract keeps six chronological 360p cues on each optimized page."""
    frames = [vision.ScoredFrame(index=index, timestamp_seconds=float(index), score=0.0) for index in range(7)]
    decoded = np.stack([np.full((vision.FRAME_HEIGHT, vision.FRAME_WIDTH, 3), index * 20, dtype=np.uint8) for index in range(len(frames))])
    ffmpeg_arguments: list[list[str]] = []

    def fake_run_ffmpeg(_ffmpeg_path, arguments, **_kwargs):
        ffmpeg_arguments.append(arguments)
        return decoded.tobytes()

    monkeypatch.setattr(vision, "run_ffmpeg", fake_run_ffmpeg)

    output_paths = vision.render_timestamped_collages(
        tmp_path / "video.mp4",
        frames,
        tmp_path / "vision",
        "ffmpeg",
    )

    assert len(output_paths) == 2
    video_filter = ffmpeg_arguments[0][ffmpeg_arguments[0].index("-vf") + 1]
    assert "flags=lanczos" in video_filter
    expected_width = vision.FRAME_WIDTH * vision.COLLAGE_COLUMNS + vision.COLLAGE_PADDING * (vision.COLLAGE_COLUMNS + 1)
    expected_height = (vision.FRAME_HEIGHT + vision.TIMESTAMP_FOOTER_HEIGHT) * vision.COLLAGE_ROWS + vision.COLLAGE_PADDING * (vision.COLLAGE_ROWS + 1)
    with Image.open(output_paths[0]) as first_page:
        assert first_page.size == (expected_width, expected_height)
        assert first_page.format == "JPEG"
        assert first_page.info["progressive"] == 1
        assert JpegImagePlugin.get_sampling(first_page) == 1


def test_download_video_for_vision_uses_bounded_720p_video_only_preset(monkeypatch, tmp_path) -> None:
    """The yt-dlp adapter owns video-only format, size, fragment, and timeout policy."""
    calls: list[list[str]] = []

    def fake_run(command, capture_output, text, check, timeout):
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)
    output_path = tmp_path / "video.mp4"

    assert YtDlpClient().download_video_for_vision(VIDEO_ID, output_path, max_bytes=123_456) == output_path
    command = calls[0]
    assert command[command.index("--format") + 1] == VISION_VIDEO_FORMAT_SELECTOR
    assert command[command.index("--max-filesize") + 1] == "123456"
    assert command[command.index("--concurrent-fragments") + 1] == "1"
    assert VISION_VIDEO_FORMAT_SELECTOR.count("bestvideo") == 2
    assert "+bestaudio" not in VISION_VIDEO_FORMAT_SELECTOR
    assert "--postprocessor-args" not in command
    assert "--merge-output-format" not in command


def test_vision_subprocess_timeouts_become_user_facing_runtime_errors(monkeypatch, tmp_path) -> None:
    """Both external process boundaries translate timeout exceptions for CLI handling."""

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["command"], timeout=1)

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", time_out)
    with pytest.raises(RuntimeError, match="video download for vision timed out"):
        YtDlpClient().download_video_for_vision(VIDEO_ID, tmp_path / "video.mp4", max_bytes=123)

    monkeypatch.setattr(vision.subprocess, "run", time_out)
    with pytest.raises(RuntimeError, match="FFmpeg vision processing timed out"):
        vision.run_ffmpeg("ffmpeg", ["-version"])


def test_vision_subprocess_start_failures_become_user_facing_runtime_errors(monkeypatch, tmp_path) -> None:
    """Missing or unlaunchable executables retain boundary-specific error context."""

    def fail_start(*_args, **_kwargs):
        raise OSError("cannot execute")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fail_start)
    with pytest.raises(RuntimeError, match="Could not start yt-dlp video download for vision"):
        YtDlpClient().download_video_for_vision(VIDEO_ID, tmp_path / "video.mp4", max_bytes=123)

    monkeypatch.setattr(vision.subprocess, "run", fail_start)
    with pytest.raises(RuntimeError, match="Could not start FFmpeg vision processing"):
        vision.run_ffmpeg("ffmpeg", ["-version"])


def test_video_vision_cli_routes_to_public_extraction(monkeypatch) -> None:
    """The video vision command returns the public operation's JSON-ready result."""
    args = build_parser().parse_args(["video", "vision", VIDEO_ID])
    expected = {
        "video_id": VIDEO_ID,
        "frame_count": 1,
        "timestamps_seconds": [8.0],
        "collage_paths": ["/example/page1.jpg"],
    }
    monkeypatch.setattr(vision, "sample_vision", lambda video_ref: expected if video_ref == VIDEO_ID else {})

    assert args.handler is commands.VideoCommands.vision
    assert commands.VideoCommands.vision(argparse.Namespace(video_ref=VIDEO_ID)) == expected
