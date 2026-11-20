"""Vision frame sampling, ranking routes, and artifacts."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from youtube_toolkit import vision
from youtube_toolkit.artifacts import Cached
from youtube_toolkit.cli import commands
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.config import (
    DEFAULT_DECISION_BASE_URL,
    DEFAULT_DECISION_MODEL,
    DEFAULT_VISION_MAX_FRAMES,
    LOCAL_EMBEDDING_CUTOFFS,
    DecisionConfig,
)
from youtube_toolkit.embeddings.results import (
    EmbeddingError,
    EmbeddingResult,
    EmbeddingUsage,
)
from youtube_toolkit.schemas import (
    SpeechTranscription,
    TranscriptDocument,
    TranscriptSegment,
    VideoMetadata,
)
from youtube_toolkit.transcript import fetch_transcript
from youtube_toolkit.vision import candidates as frame_selection, workflow
from youtube_toolkit.vision.decisions import DecisionUsage
from youtube_toolkit.youtube.captions import CaptionTranscriptUnavailable
from youtube_toolkit.youtube.client import YtDlpClient

VIDEO_ID = "JpSHyEIZ_bo"
VIDEOS = Path(".youtube-toolkit/videos")
DECISION = DecisionConfig(
    DEFAULT_DECISION_BASE_URL, "decision-key", DEFAULT_DECISION_MODEL
)
RELEVANCE = {
    "selection_rank": 1,
    "score": 0.95,
    "has_information": 0.9,
    "visual": 0.3,
    "impressive": 0.4,
    "loaded": 0.9,
    "illustrative": 0.2,
}


SCORES = frame_selection.VisualScores(20.0, 0.01, 0.0)


@pytest.fixture(autouse=True)
def working_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)


def picks(result: dict) -> dict:
    return json.loads(Path(result["path"]).read_text(encoding="utf-8"))["data"]


def store_frame(
    directory: Path, timestamp: float, image: np.ndarray
) -> frame_selection.StoredFrame:
    directory.mkdir(exist_ok=True)
    path = directory / f"frame-{round(timestamp * 1000):09d}.jpg"
    Image.fromarray(image).save(path)
    return frame_selection.StoredFrame(timestamp, path, SCORES)


def chart(heights=(20, 30, 40, 50)) -> np.ndarray:
    image = np.full((360, 640, 3), 240, dtype=np.uint8)
    image[:56] = 40
    for column, height in enumerate(heights):
        image[320 - height * 4 : 320, 60 + column * 140 : 140 + column * 140] = 100
    return cv2.resize(
        image,
        (frame_selection.CANDIDATE_WIDTH, frame_selection.CANDIDATE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )


@pytest.mark.parametrize("fallback", ["local_only", "no_key", "no_captions"])
def test_local_selection_reports_its_reason_and_lays_saved_picks_out_again(
    monkeypatch, tmp_path, fallback
) -> None:
    metadata_calls = []
    transcript_calls = []
    downloads = []

    class FakeClient:
        def metadata(self, _ref):
            metadata_calls.append(_ref)
            return {"id": VIDEO_ID, "duration": 120, "heatmap": [{"value": 1}]}

        def download_silent_video(
            self, _metadata, path, *, height, section, max_bytes, timeout_seconds
        ):
            assert (height, section, max_bytes) == (720, None, workflow.MAX_VIDEO_BYTES)
            downloads.append(path)
            path.write_bytes(b"video")

        def download_transcription_audio(self, _metadata, _directory):
            pytest.fail("no speech-to-text route is configured here")

    def transcript(metadata, *, languages):
        transcript_calls.append(metadata)
        raise CaptionTranscriptUnavailable("No captions")

    def candidates(
        path,
        _ffmpeg,
        *,
        start,
        duration,
        frame_size,
        directory,
        excluded_intervals,
        distinct,
    ):
        assert path.exists()
        assert (start, duration, frame_size, distinct) == (0, 120, (960, 540), True)
        return [store_frame(directory, 12, chart())]

    local_only = fallback == "local_only"
    monkeypatch.setattr(workflow, "YtDlpClient", FakeClient)
    monkeypatch.setattr(workflow.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript", transcript
    )
    monkeypatch.setattr(workflow, "collect_candidates", candidates)
    monkeypatch.setattr("youtube_toolkit.transcript._stt_transcriber", lambda: None)
    monkeypatch.setattr(
        workflow,
        "gemini_embedding_key",
        lambda: None if fallback == "no_key" else "test-key",
    )
    monkeypatch.setattr(
        workflow, "GeminiEmbeddingClient", lambda _key: pytest.fail("no embedding call")
    )
    result = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    destination = Path(result["path"]).parent
    first_collage = destination / "collages" / "3x2_540p" / "page1.jpg"
    assert result == {
        "path": str(destination / "picks.json"),
        "collages": [str(first_collage)],
        "frames": 1,
        "selection_mode": "visual",
        "warnings": result["warnings"],
        "reused": False,
    }
    assert len(result["warnings"]) == (0 if local_only else 1)
    assert destination.parent == tmp_path / VIDEOS / VIDEO_ID / "vision"
    assert metadata_calls == [VIDEO_ID]
    assert len(downloads) == 1
    assert bool(transcript_calls) is (fallback == "no_captions")
    saved = picks(result)
    assert saved["selection_mode"] == "visual"
    assert bool(saved["transcript_warning"]) is (fallback == "no_captions")
    assert bool(saved["embedding_warning"]) is (fallback == "no_key")
    assert saved["embedding_model"] is None
    assert saved["candidate_count"] == 1
    assert saved["frame_height"] == 540
    [frame] = saved["frames"]
    image_path = Path(frame.pop("image_path"))
    assert image_path.parent == destination / "frames"
    assert image_path.is_file()
    assert frame == {
        "timestamp_seconds": 12,
        "visual_scores": {
            "clarity": 20.0,
            "neighbor_change": 0.01,
            "similarity_to_previous": 0.0,
        },
        "relevance": None,
    }
    assert saved["timestamps_seconds"] == [12]
    assert saved["heatmap_available"] is True
    assert first_collage.is_file()
    assert not (destination / "work").exists()
    assert not list((tmp_path / ".youtube-toolkit" / "tmp").iterdir())
    first_contents = first_collage.read_bytes()

    second = workflow.sample_vision(
        VIDEO_ID, local_only=local_only, rows=1, columns=1, resolution="360p"
    )
    assert second["path"] == result["path"]
    assert second["reused"] is True
    second_collage = destination / "collages" / "1x1_360p" / "page1.jpg"
    assert second["collages"] == [str(second_collage)]
    assert len(downloads) == 1
    assert len(transcript_calls) == (fallback == "no_captions")
    assert first_collage.read_bytes() == first_contents
    with Image.open(second_collage) as page:
        assert page.size == (652, 396)

    third = workflow.sample_vision(VIDEO_ID, local_only=local_only, min_gap_seconds=5)
    assert third["reused"] is False
    assert Path(third["path"]).parent.parent == destination.parent
    assert Path(third["path"]).parent != destination
    assert len(downloads) == 2
    fresh = workflow.sample_vision(VIDEO_ID, local_only=local_only, fresh=True)
    assert fresh["path"] == result["path"]
    assert fresh["reused"] is False
    assert len(downloads) == 3
    decoded = []

    def decode_frame(_video, timestamp, _ffmpeg, *, width, height):
        decoded.append((timestamp, width, height))
        return np.zeros((height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(workflow, "decode_frame", decode_frame)
    higher = workflow.sample_vision(VIDEO_ID, local_only=local_only, resolution="720p")
    assert higher["path"] == result["path"]
    assert higher["reused"] is False
    assert len(downloads) == 4
    assert decoded == [(12, 1280, 720)]
    assert not second_collage.exists()
    with Image.open(picks(higher)["frames"][0]["image_path"]) as frame:
        assert frame.size == (1280, 720)
    lower = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    assert lower["reused"] is True
    assert lower["collages"] == [str(first_collage)]
    assert len(downloads) == 4
    assert metadata_calls == [VIDEO_ID] * 6


@pytest.mark.parametrize("limit", [0, 97, -1])
def test_invalid_budget_fails_before_fetch(monkeypatch, limit) -> None:
    monkeypatch.setattr(workflow, "YtDlpClient", lambda: pytest.fail("no network"))
    with pytest.raises(ValueError, match="max_frames"):
        workflow.sample_vision(VIDEO_ID, max_frames=limit)


@pytest.mark.parametrize(
    "options", [{"rows": 0}, {"rows": 4}, {"columns": 0}, {"columns": 4}]
)
def test_invalid_collage_layout_fails_before_fetch(monkeypatch, options) -> None:
    monkeypatch.setattr(workflow, "YtDlpClient", lambda: pytest.fail("no network"))
    with pytest.raises(ValueError, match="must be between 1 and 3"):
        workflow.sample_vision(VIDEO_ID, **options)


def test_sample_vision_rejects_long_video_before_download(monkeypatch) -> None:
    class FakeClient:
        def metadata(self, _ref):
            return {"id": VIDEO_ID, "duration": workflow.MAX_VIDEO_DURATION_SECONDS + 1}

        def download_silent_video(self, *_args, **_kwargs):
            pytest.fail("download should not run")

    monkeypatch.setattr(workflow, "YtDlpClient", FakeClient)
    monkeypatch.setattr(workflow.shutil, "which", lambda _: "ffmpeg")
    with pytest.raises(ValueError, match="at most 2 hours"):
        workflow.sample_vision(VIDEO_ID)


def test_a_vision_range_downloads_that_section_and_keeps_video_times(
    monkeypatch,
) -> None:
    sections, decoded = [], []

    class FakeClient:
        def metadata(self, _ref):
            return {"id": VIDEO_ID, "duration": 600}

        def download_silent_video(self, _metadata, path, *, section, **_options):
            sections.append(section)
            path.write_bytes(b"video")

    def candidates(_path, _ffmpeg, *, start, duration, directory, **_options):
        assert (start, duration) == (60, 30)
        return [
            store_frame(directory, 65, chart()),
            store_frame(directory, 80, chart((10, 20, 35, 60))),
        ]

    def decode_frame(_path, timestamp, _ffmpeg, *, width, height):
        decoded.append(timestamp)
        return np.zeros((height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(workflow, "YtDlpClient", FakeClient)
    monkeypatch.setattr(workflow.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(workflow, "local_embedding_installed", lambda: False)
    monkeypatch.setattr(workflow, "collect_candidates", candidates)
    monkeypatch.setattr(workflow, "decode_frame", decode_frame)

    result = workflow.sample_vision(
        VIDEO_ID,
        start=60,
        end=90,
        resolution="720p",
        local_only=True,
        min_gap_seconds=0,
    )

    assert sections == [(60, 90)]
    assert picks(result)["timestamps_seconds"] == [65, 80]
    assert decoded == [5, 20]


def test_slot_selection_takes_candidates_nearest_slot_middles(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(frame_selection, "REPEAT_SIMILARITY", 2.0)
    image = store_frame(tmp_path, 0, chart()).image_path
    candidates = [
        frame_selection.StoredFrame(float(timestamp), image, SCORES)
        for timestamp in np.arange(0, 120, 2.5)
    ]

    def times(max_frames: int) -> list[float]:
        frames = frame_selection.select_slot_frames(
            candidates,
            start=0,
            duration=120,
            max_frames=max_frames,
            min_gap_seconds=0,
        )
        return [frame.timestamp_seconds for frame in frames]

    assert times(96) == [5, 17.5, 30, 42.5, 55, 65, 77.5, 90, 102.5, 115]
    assert times(4) == [15, 45, 75, 105]


def test_slot_selection_skips_empty_slots_close_picks_and_repeats(tmp_path) -> None:
    first, second, third = chart(), chart((10, 20, 35, 60)), chart((60, 40, 25, 10))
    candidates = [
        store_frame(tmp_path, 6, first),
        store_frame(tmp_path, 13, second),
        store_frame(tmp_path, 30, second),
        store_frame(tmp_path, 42, first),
        store_frame(tmp_path, 54, third),
    ]
    frames = frame_selection.select_slot_frames(
        candidates, start=0, duration=60, max_frames=96, min_gap_seconds=10
    )
    assert [frame.timestamp_seconds for frame in frames] == [6, 30, 54]


def test_ranked_selection_skips_close_picks_and_repeats_of_any_pick(tmp_path) -> None:
    first, second, third = chart(), chart((10, 20, 35, 60)), chart((60, 40, 25, 10))
    candidates = [
        store_frame(tmp_path, 10, first),
        store_frame(tmp_path, 40, second),
        store_frame(tmp_path, 45, third),
        store_frame(tmp_path, 80, first),
    ]

    def picks(max_frames: int) -> list[int]:
        selected = frame_selection.select_ranked_frames(
            candidates,
            [1, 2, 3, 0],
            max_frames=max_frames,
            min_gap_seconds=10,
            repeats=frozenset(),
        )
        return [index for index, _frame in selected]

    assert picks(96) == [1, 3]
    assert picks(1) == [1]


@pytest.mark.parametrize(
    ("order", "picked"),
    [([1, 0, 2], [1]), ([2, 1, 0], [2, 0])],
    ids=["its repeat was picked", "its repeat sat too close to another pick"],
)
def test_ranked_selection_skips_a_repeat_only_once_the_frame_it_repeats_is_picked(
    tmp_path, order, picked
) -> None:
    # 0 and 100 show one item; 104 is something else beside 100.
    candidates = [
        store_frame(tmp_path, 0, chart()),
        store_frame(tmp_path, 100, chart((10, 20, 35, 60))),
        store_frame(tmp_path, 104, chart((60, 40, 25, 10))),
    ]
    selected = frame_selection.select_ranked_frames(
        candidates,
        order,
        max_frames=96,
        min_gap_seconds=10,
        repeats=frozenset({(0, 1)}),
    )
    assert [index for index, _frame in selected] == picked


@pytest.mark.parametrize("section", [None, (60.0, 90.0)])
def test_silent_video_download_takes_the_picture_alone_within_a_size_cap(
    monkeypatch, tmp_path, section
) -> None:
    """yt-dlp would refuse a short section of a long video by the whole stream's size, so a section's cap applies only once saved."""
    calls: list[list[str]] = []

    def fake_run(command, input, capture_output, text, check, timeout):
        calls.append(command)
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fake_run)

    YtDlpClient().download_silent_video(
        {"id": VIDEO_ID},
        tmp_path / "video.mp4",
        height=720,
        section=section,
        max_bytes=123_456,
        timeout_seconds=600,
    )
    command = calls[0]
    assert (
        command[command.index("--format") + 1]
        == "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]"
    )
    assert ("--max-filesize" in command) is (section is None)
    assert ("--download-sections" in command) is (section is not None)
    assert "--merge-output-format" not in command
    with pytest.raises(RuntimeError, match="exceeded 3 bytes"):
        YtDlpClient().download_silent_video(
            {"id": VIDEO_ID},
            tmp_path / "large.mp4",
            height=720,
            section=section,
            max_bytes=3,
            timeout_seconds=600,
        )
    assert not (tmp_path / "large.mp4").exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            subprocess.TimeoutExpired(["yt-dlp"], timeout=1),
            "video download timed out",
        ),
        (OSError("cannot execute"), "Could not start yt-dlp video download"),
    ],
)
def test_vision_download_failures_become_user_facing_runtime_errors(
    monkeypatch, tmp_path, error, message
) -> None:
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fail)
    with pytest.raises(RuntimeError, match=message):
        YtDlpClient().download_silent_video(
            {"id": VIDEO_ID},
            tmp_path / "video.mp4",
            height=720,
            section=None,
            max_bytes=123,
            timeout_seconds=600,
        )


@pytest.mark.parametrize(
    ("arguments", "options"),
    [
        ([], {}),
        (
            [
                "--max-frames",
                "12",
                "--rows",
                "2",
                "--columns",
                "3",
                "--resolution",
                "720p",
                "--no-timestamp-annotation",
                "--start",
                "1:00",
                "--end",
                "1:30",
                "--min-similarity",
                "0.4",
                "--min-gap",
                "5",
                "--local-only",
                "--no-auto-clean",
            ],
            {
                "max_frames": 12,
                "rows": 2,
                "columns": 3,
                "resolution": "720p",
                "timestamp_annotation": False,
                "start": 60.0,
                "end": 90.0,
                "min_similarity": 0.4,
                "min_gap_seconds": 5,
                "local_only": True,
                "auto_clean": False,
            },
        ),
    ],
)
def test_video_vision_passes_only_given_options_to_extraction(
    monkeypatch, arguments, options
) -> None:
    """Options left out keep sample_vision's own defaults."""
    payload = {"id": VIDEO_ID, "duration": 120}
    received = []

    class Client:
        def metadata(self, _ref):
            return payload

    def sample_vision(video_ref, **kwargs):
        received.append((video_ref, kwargs))
        return {"frame_count": 1}

    monkeypatch.setattr(commands, "YtDlpClient", Client)
    monkeypatch.setattr(commands, "require_vision", lambda: None)
    monkeypatch.setattr(vision, "sample_vision", sample_vision)
    args = build_parser().parse_args(["video", VIDEO_ID, "--vision", *arguments])

    assert commands.video(args) == {"vision": {"frame_count": 1}}
    assert received == [
        (
            VIDEO_ID,
            {
                "metadata": payload,
                "document": None,
                "start": None,
                "end": None,
                "fresh": False,
                "video_path": None,
                **options,
            },
        )
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"min_similarity": float("nan")},
        {"min_similarity": 1.1},
        {"min_gap_seconds": -1},
        {"min_gap_seconds": float("inf")},
    ],
)
def test_invalid_ranking_options_fail_before_fetch(monkeypatch, options) -> None:
    monkeypatch.setattr(workflow, "YtDlpClient", lambda: pytest.fail("no network"))
    with pytest.raises(ValueError, match="must be finite"):
        workflow.sample_vision(VIDEO_ID, **options)


def test_candidate_grid_reaches_video_end_and_respects_exclusions(
    monkeypatch, tmp_path
) -> None:
    image = chart()
    calls = []

    def decode(_path, start, stop, _ffmpeg, *, width, height):
        calls.append((start, stop))
        times = [float(time) for time in np.arange(start + 2.5, stop, 5)]
        return times, np.repeat(image[None], len(times), axis=0)

    monkeypatch.setattr(frame_selection, "decode_keyframes", decode)
    candidates = frame_selection.collect_candidates(
        tmp_path / "video.mp4",
        "ffmpeg",
        start=0,
        duration=125,
        frame_size=(frame_selection.CANDIDATE_WIDTH, frame_selection.CANDIDATE_HEIGHT),
        directory=tmp_path / "candidates",
        excluded_intervals=[(24, 75)],
        distinct=True,
    )
    times = [frame.timestamp_seconds for frame in candidates]
    assert sorted(calls) == [(0, 60), (60, 120), (120, 125)]
    assert 2.5 in times
    assert 122.5 in times
    assert not any(24 <= time < 75 for time in times)
    assert len(times) == len(set(times))
    assert len({frame.image_path for frame in candidates}) == len(candidates)
    assert all(frame.image_path.is_file() for frame in candidates)
    assert candidates[-1].load().image.shape == (
        frame_selection.CANDIDATE_HEIGHT,
        frame_selection.CANDIDATE_WIDTH,
        3,
    )


@pytest.fixture
def semantic_video(monkeypatch, tmp_path):
    class Client:
        def metadata(self, _ref):
            return {"id": VIDEO_ID, "duration": 120}

        def download_silent_video(self, _metadata, path, **_options):
            path.write_bytes(b"video")

    document = TranscriptDocument(
        VideoMetadata(VIDEO_ID, "video", duration_seconds=120),
        "en",
        "subtitles",
        [
            TranscriptSegment(
                5,
                15,
                "The circuit diagram explains how the current flows from the battery through each resistor.",
            ),
            TranscriptSegment(
                108,
                116,
                "The meter shows the current dropping as we add a second resistor in series.",
            ),
        ],
    )
    monkeypatch.setattr(workflow, "YtDlpClient", Client)
    monkeypatch.setattr(workflow.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(
        workflow,
        "fetch_transcript",
        lambda *_args, **_kwargs: Cached(
            document, False, tmp_path / "original_fallback-abc.json"
        ),
    )
    monkeypatch.setattr(workflow, "gemini_embedding_key", lambda: "test-key")

    def candidates(_path, _ffmpeg, *, start, directory, **_options):
        return [
            store_frame(directory, timestamp, chart())
            for timestamp in [12, 112]
            if timestamp >= start
        ]

    monkeypatch.setattr(workflow, "collect_candidates", candidates)
    return tmp_path


def test_artifact_directory_errors_keep_vision_context(semantic_video) -> None:
    folder = semantic_video / VIDEOS / VIDEO_ID
    folder.mkdir(parents=True)
    (folder / "vision").write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Vision artifact filesystem error") as error:
        workflow.sample_vision(VIDEO_ID, local_only=True)

    assert isinstance(error.value.__cause__, NotADirectoryError)


@pytest.mark.parametrize("empty", [False, True])
def test_semantic_selection_searches_beyond_output_budget_and_does_not_fill(
    monkeypatch, semantic_video, empty
) -> None:
    class Embedder:
        model = "gemini-embedding-2"

        def __init__(self, key):
            assert key == "test-key"

        def embed(self, images, texts):
            if not images:
                return EmbeddingResult(
                    np.empty((0, 2)),
                    np.tile([1.0, 0.0], (len(texts), 1)),
                    EmbeddingUsage(0, 20, 20),
                )
            assert not texts
            assert len(images) == 2
            assert all(path.exists() for path in images)
            image_vectors = np.array([[0.0, 1.0], [0.0, 1.0] if empty else [1.0, 0.0]])
            return EmbeddingResult(
                image_vectors, np.empty((0, 2)), EmbeddingUsage(516, 0, 516)
            )

    monkeypatch.setattr(workflow, "GeminiEmbeddingClient", Embedder)
    result = workflow.sample_vision(VIDEO_ID, max_frames=1)
    assert result["selection_mode"] == "semantic"
    assert result["frames"] == (0 if empty else 1)
    saved = picks(result)
    assert saved["candidate_count"] == 2
    assert saved["timestamps_seconds"] == ([] if empty else [112])
    assert saved["embedding_usage"] == {
        "image_tokens": 516,
        "text_tokens": 20,
        "total_tokens": 536,
    }
    assert len(result["collages"]) == (0 if empty else 1)
    if not empty:
        assert saved["frames"][0]["relevance"]["selection_rank"] == 1
        assert saved["frames"][0]["relevance"]["matches"][0]["cosine"] == 1
    assert not (Path(result["path"]).parent / "work").exists()


def test_embedding_failure_stops_before_download_and_writes_no_artifacts(
    monkeypatch, semantic_video
) -> None:
    class Embedder:
        model = "gemini-embedding-2"

        def __init__(self, _key):
            pass

        def embed(self, _images, _texts):
            raise EmbeddingError("Gemini embedding request failed (HTTP 503)")

    monkeypatch.setattr(workflow, "GeminiEmbeddingClient", Embedder)
    monkeypatch.setattr(
        workflow.YtDlpClient,
        "download_silent_video",
        lambda *_args, **_kwargs: pytest.fail("download must wait for text embeddings"),
    )
    with pytest.raises(EmbeddingError, match="503"):
        workflow.sample_vision(VIDEO_ID)
    assert not (semantic_video / VIDEOS / VIDEO_ID / "vision").exists()


def test_explicit_local_only_skips_configured_embedding_client(
    monkeypatch, semantic_video
) -> None:
    monkeypatch.setattr(
        workflow, "GeminiEmbeddingClient", lambda _key: pytest.fail("no API call")
    )
    monkeypatch.setattr(
        workflow,
        "fetch_transcript",
        lambda *_args, **_kwargs: pytest.fail("no captions"),
    )
    result = workflow.sample_vision(VIDEO_ID, local_only=True)
    assert result["selection_mode"] == "visual"
    assert picks(result)["candidate_count"] == 2
    assert picks(result)["embedding_usage"] is None


@pytest.mark.parametrize(
    ("distinct", "times"),
    [(True, [105.0, 125.0, 145.0]), (False, [105.0, 115.0, 125.0, 135.0, 145.0])],
)
def test_distinct_candidates_drop_a_cross_fade_and_a_blank_frame(
    monkeypatch, tmp_path, distinct, times
) -> None:
    """Times in the file count from where it starts in the video, and without distinct every keyframe stays."""
    before, after, later = chart(), chart((50, 40, 30, 20)), chart((10, 50, 10, 50))
    fade = ((before.astype(np.uint16) + after) // 2).astype(np.uint8)
    blank = np.zeros_like(before)

    def decode(_path, _start, _stop, _ffmpeg, *, width, height):
        return [5.0, 15.0, 25.0, 35.0, 45.0], np.stack(
            [before, fade, after, blank, later]
        )

    monkeypatch.setattr(frame_selection, "decode_keyframes", decode)
    directory = tmp_path / "candidates"
    candidates = frame_selection.collect_candidates(
        tmp_path / "video.mp4",
        "ffmpeg",
        start=100,
        duration=50,
        frame_size=(frame_selection.CANDIDATE_WIDTH, frame_selection.CANDIDATE_HEIGHT),
        directory=directory,
        excluded_intervals=[],
        distinct=distinct,
    )
    assert [frame.timestamp_seconds for frame in candidates] == times
    assert sorted(directory.iterdir()) == [frame.image_path for frame in candidates]


@pytest.mark.parametrize("local_only", [False, True])
def test_decision_model_ranks_first_with_a_decision_key(
    monkeypatch, semantic_video, local_only
) -> None:
    warning = "1 of 2 frame requests failed after repeated decision host errors, so those frames were left out of the ranking."

    def rank(candidates, document, config, *, duration, max_frames, min_gap_seconds):
        assert config == DECISION
        assert (duration, max_frames) == (120, DEFAULT_VISION_MAX_FRAMES)
        assert document.segments
        assert len(candidates) == 2
        requests = SimpleNamespace(
            usage=DecisionUsage(200, 6, 0.00002, 3.1), warning=warning
        )
        return [(candidates[0].load(), RELEVANCE)], requests

    monkeypatch.setattr(workflow, "decision_config", lambda: DECISION)
    monkeypatch.setattr(workflow, "_rank_by_decisions", rank)
    monkeypatch.setattr(
        workflow, "GeminiEmbeddingClient", lambda _key: pytest.fail("no embeddings")
    )
    result = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    saved = picks(result)
    if local_only:
        assert result["selection_mode"] == "visual"
        assert saved["decision_model"] is None
        return
    assert result["selection_mode"] == "decision"
    assert result["warnings"] == [warning]
    assert saved["decision_model"] == DEFAULT_DECISION_MODEL
    assert saved["decision_usage"] == {
        "input_tokens": 200,
        "output_tokens": 6,
        "cost_usd": 0.00002,
        "provider_seconds": 3.1,
    }
    assert saved["min_similarity"] is None
    assert saved["decision_warning"] == warning
    assert saved["timestamps_seconds"] == [12]
    assert saved["frames"][0]["relevance"] == RELEVANCE


def test_a_given_transcript_is_ranked_against_without_fetching(
    monkeypatch, semantic_video
) -> None:
    given = workflow.fetch_transcript(VIDEO_ID, fresh=False).value
    ranked_against = []

    def rank(candidates, document, config, **_picking):
        ranked_against.append(document)
        requests = SimpleNamespace(
            usage=DecisionUsage(None, None, None, None), warning=None
        )
        return [(candidates[0].load(), RELEVANCE)], requests

    monkeypatch.setattr(
        workflow, "fetch_transcript", lambda *_args, **_kwargs: pytest.fail("fetch")
    )
    monkeypatch.setattr(workflow, "decision_config", lambda: DECISION)
    monkeypatch.setattr(workflow, "_rank_by_decisions", rank)
    result = workflow.sample_vision(VIDEO_ID, document=given)
    assert ranked_against == [given]
    assert result["collages"] == [
        str(Path(result["path"]).parent / "collages" / "3x2_540p" / "page1.jpg")
    ]


@pytest.mark.parametrize("local_only", [False, True])
def test_local_model_ranks_without_a_key_on_its_own_cutoffs(
    monkeypatch, semantic_video, local_only
) -> None:
    class Embedder:
        model = "example/local-model"

        def embed(self, images, texts):
            return EmbeddingResult(
                np.tile([1.0, 0.0], (len(images), 1)),
                np.tile([1.0, 0.0], (len(texts), 1)),
                EmbeddingUsage(None, None, None),
            )

    transcript = workflow.fetch_transcript
    policies: list[str] = []

    def fetch(*args, stt: str, **kwargs):
        policies.append(stt)
        return transcript(*args, **kwargs)

    monkeypatch.setattr(workflow, "fetch_transcript", fetch)
    monkeypatch.setattr(workflow, "gemini_embedding_key", lambda: None)
    monkeypatch.setattr(workflow, "local_embedding_installed", lambda: True)
    monkeypatch.setattr(workflow, "LocalImageEmbeddingClient", Embedder)
    result = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    saved = picks(result)
    assert result["selection_mode"] == "semantic"
    assert saved["embedding_model"] == "example/local-model"
    assert saved["min_similarity"] == LOCAL_EMBEDDING_CUTOFFS.vision_match
    assert "not example/local-model" in saved["embedding_warning"]
    assert result["warnings"] == [saved["embedding_warning"]]
    assert policies == ["never" if local_only else "fallback"]


@pytest.mark.parametrize("timed", [True, False])
def test_missing_captions_fall_back_to_speech_to_text_for_ranking(
    monkeypatch, semantic_video, timed
) -> None:
    downloaded = []

    def download_audio(_self, _metadata, directory):
        audio = directory / "audio.m4a"
        audio.write_bytes(b"audio")
        downloaded.append(audio)
        return audio

    def no_captions(_metadata, *, languages):
        raise CaptionTranscriptUnavailable("No captions")

    segments = [
        TranscriptSegment(
            5,
            15,
            "The circuit diagram explains how the current flows from the battery through each resistor.",
        ),
        TranscriptSegment(
            108,
            116,
            "The meter shows the current dropping as we add a second resistor in series.",
        ),
    ]
    text = " ".join(segment.text for segment in segments)

    class Embedder:
        model = "gemini-embedding-2"

        def __init__(self, _key):
            pass

        def embed(self, images, texts):
            return EmbeddingResult(
                np.tile([0.0, 1.0], (len(images), 1)),
                np.tile([0.0, 1.0], (len(texts), 1)),
                EmbeddingUsage(0, 0, 0),
            )

    monkeypatch.setattr(workflow, "fetch_transcript", fetch_transcript)
    monkeypatch.setattr(
        workflow.YtDlpClient,
        "download_transcription_audio",
        download_audio,
        raising=False,
    )
    monkeypatch.setattr(
        "youtube_toolkit.transcript.fetch_caption_transcript", no_captions
    )
    monkeypatch.setattr(
        "youtube_toolkit.transcript._stt_transcriber",
        lambda: (
            lambda _path, _vocabulary: SpeechTranscription(
                text=text,
                segments=segments if timed else [],
                model="example-model",
                language="en",
            )
        ),
    )
    monkeypatch.setattr(workflow, "GeminiEmbeddingClient", Embedder)

    result = workflow.sample_vision(VIDEO_ID)

    [audio] = downloaded
    assert not audio.exists()
    if timed:
        assert result["selection_mode"] == "semantic"
        assert result["warnings"] == []
    else:
        assert result["selection_mode"] == "visual"
        assert result["warnings"] == [
            "Speech-to-text gave no timestamps; using visual sampling."
        ]


def test_auto_clean_off_leaves_video_and_candidates_in_a_work_folder(
    monkeypatch, semantic_video
) -> None:
    class Embedder:
        model = "gemini-embedding-2"

        def __init__(self, _key):
            pass

        def embed(self, images, texts):
            return EmbeddingResult(
                np.eye(2)[: len(images)],
                np.tile([1.0, 0.0], (len(texts), 1)),
                EmbeddingUsage(0, 0, 0),
            )

    monkeypatch.setattr(workflow, "GeminiEmbeddingClient", Embedder)
    result = workflow.sample_vision(VIDEO_ID, auto_clean=False)
    media = Path(result["path"]).parent / "work"
    assert (media / f"{VIDEO_ID}.mp4").read_bytes() == b"video"
    assert len(list((media / "candidates").glob("*.jpg"))) == 2
    assert workflow.sample_vision(VIDEO_ID, auto_clean=False)["reused"] is False
    assert workflow.sample_vision(VIDEO_ID)["reused"] is True
