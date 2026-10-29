"""Vision keeps broad candidate coverage, explicit retrieval/fallback behavior, and bounded artifact publication."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from youtube_toolkit import ffmpeg_utils, vision
from youtube_toolkit.cli import commands
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.config import (
    DEFAULT_DECISION_MODEL,
    DEFAULT_DECISION_URL,
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
    TranscriptChunk,
    TranscriptDocument,
    VideoMetadata,
)
from youtube_toolkit.transcript import fetch_transcript
from youtube_toolkit.vision import candidates as frame_selection, workflow
from youtube_toolkit.vision.decisions import (
    MIN_SCORE,
    DecisionUsage,
    FrameDecision,
)
from youtube_toolkit.youtube.caption import CaptionTranscriptUnavailable
from youtube_toolkit.youtube.client import VISION_VIDEO_FORMAT_SELECTOR, YtDlpClient

VIDEO_ID = "JpSHyEIZ_bo"
DECISION = DecisionConfig(DEFAULT_DECISION_URL, "decision-key", DEFAULT_DECISION_MODEL)


SCORES = frame_selection.VisualScores(20.0, 0.01, 0.0)


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
        (ffmpeg_utils.FRAME_WIDTH, ffmpeg_utils.FRAME_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )


@pytest.mark.parametrize("fallback", ["local_only", "no_key", "no_captions"])
def test_local_selection_reports_its_reason_and_cleans_up(
    monkeypatch, tmp_path, fallback
) -> None:
    metadata_calls = []
    transcript_calls = []

    class FakeClient:
        def metadata(self, _ref):
            metadata_calls.append(_ref)
            return {"id": VIDEO_ID, "duration": 120, "heatmap": [{"value": 1}]}

        def download_video_for_vision(self, _ref, path, *, max_bytes):
            assert max_bytes == workflow.MAX_VIDEO_BYTES
            path.write_bytes(b"video")
            return path

        def download_transcription_audio(self, _ref):
            pytest.fail("no speech-to-text route is configured here")

    def transcript(metadata, *, languages):
        transcript_calls.append(metadata)
        raise CaptionTranscriptUnavailable("No captions")

    def candidates(path, _ffmpeg, *, duration, directory, excluded_intervals):
        assert path.exists()
        assert duration == 120
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
        "vision_embedding_key",
        lambda: None if fallback == "no_key" else "test-key",
    )
    monkeypatch.setattr(
        workflow, "GeminiEmbeddingClient", lambda _key: pytest.fail("no embedding call")
    )
    monkeypatch.chdir(tmp_path)
    result = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    assert metadata_calls == [VIDEO_ID]
    assert bool(transcript_calls) is (fallback == "no_captions")
    assert result["selection_mode"] == "visual"
    assert bool(result["transcript_warning"]) is (fallback == "no_captions")
    assert bool(result["embedding_warning"]) is (fallback == "no_key")
    assert result["embedding_model"] is None
    assert result["candidate_count"] == 1
    assert result["frames"] == [
        {
            "timestamp_seconds": 12,
            "visual_scores": {
                "clarity": 20.0,
                "neighbor_change": 0.01,
                "similarity_to_previous": 0.0,
            },
            "relevance": None,
        }
    ]
    assert result["timestamps_seconds"] == [12]
    assert result["heatmap_available"] is True
    assert Path(result["collage_paths"][0]).is_file()
    assert not list(Path(result["collage_paths"][0]).parent.glob("work-*"))
    assert result["media_dir"] is None
    first_collage = Path(result["collage_paths"][0])
    first_contents = first_collage.read_bytes()
    second = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    assert metadata_calls == [VIDEO_ID, VIDEO_ID]
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
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="longer than 2 hours"):
        workflow.sample_vision(VIDEO_ID)


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
            candidates, duration=120, max_frames=max_frames, min_gap_seconds=0
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
        candidates, duration=60, max_frames=96, min_gap_seconds=10
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
            candidates, [1, 2, 3, 0], max_frames=max_frames, min_gap_seconds=10
        )
        return [index for index, _frame in selected]

    assert picks(96) == [1, 3]
    assert picks(1) == [1]


def test_download_video_for_vision_uses_bounded_720p_video_only_preset(
    monkeypatch, tmp_path
) -> None:
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


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            subprocess.TimeoutExpired(["yt-dlp"], timeout=1),
            "video download for vision timed out",
        ),
        (OSError("cannot execute"), "Could not start yt-dlp video download for vision"),
    ],
)
def test_vision_download_failures_become_user_facing_runtime_errors(
    monkeypatch, tmp_path, error, message
) -> None:
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("youtube_toolkit.youtube.client.subprocess.run", fail)
    with pytest.raises(RuntimeError, match=message):
        YtDlpClient().download_video_for_vision(
            VIDEO_ID, tmp_path / "video.mp4", max_bytes=123
        )


def test_video_vision_cli_routes_to_public_extraction(monkeypatch) -> None:
    args = build_parser().parse_args(
        [
            "vision",
            VIDEO_ID,
            "--max-frames",
            "12",
            "--min-similarity",
            "0.4",
            "--min-gap",
            "5",
            "--local-only",
            "--no-auto-clean",
        ]
    )
    expected = {
        "video_id": VIDEO_ID,
        "frame_count": 1,
        "timestamps_seconds": [8.0],
        "collage_paths": ["/example/page1.jpg"],
    }
    monkeypatch.setattr(
        vision,
        "sample_vision",
        lambda video_ref, *, max_frames, min_similarity, min_gap_seconds, local_only, auto_clean: (
            expected
            if video_ref == VIDEO_ID
            and max_frames == 12
            and min_similarity == 0.4
            and min_gap_seconds == 5
            and local_only
            and not auto_clean
            else {}
        ),
    )

    assert args.handler is commands.vision
    assert commands.vision(args) == expected


def test_vision_defaults_allow_multiple_six_frame_pages() -> None:
    args = build_parser().parse_args(["vision", VIDEO_ID])
    assert args.max_frames == 96
    assert args.min_similarity is None
    assert args.min_gap == 10
    assert args.local_only is False
    assert args.auto_clean is True


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

    def decode(_path, start, stop, _ffmpeg):
        calls.append((start, stop))
        times = [float(time) for time in np.arange(start + 2.5, stop, 5)]
        return times, np.repeat(image[None], len(times), axis=0)

    monkeypatch.setattr(frame_selection, "decode_keyframes", decode)
    candidates = frame_selection.collect_candidates(
        tmp_path / "video.mp4",
        "ffmpeg",
        duration=125,
        directory=tmp_path / "candidates",
        excluded_intervals=[(24, 75)],
    )
    times = [frame.timestamp_seconds for frame in candidates]
    assert calls == [(0, 60), (60, 120), (120, 125)]
    assert 2.5 in times
    assert 122.5 in times
    assert not any(24 <= time < 75 for time in times)
    assert len(times) == len(set(times))
    assert len({frame.image_path for frame in candidates}) == len(candidates)
    assert all(frame.image_path.is_file() for frame in candidates)
    assert candidates[-1].load().image.shape == (
        ffmpeg_utils.FRAME_HEIGHT,
        ffmpeg_utils.FRAME_WIDTH,
        3,
    )


@pytest.fixture
def semantic_video(monkeypatch, tmp_path):
    class Client:
        def metadata(self, _ref):
            return {"id": VIDEO_ID, "duration": 120}

        def download_video_for_vision(self, _ref, path, *, max_bytes):
            path.write_bytes(b"video")

    document = TranscriptDocument(
        VideoMetadata(VIDEO_ID, "video", duration_seconds=120),
        "en",
        "subtitles",
        [
            TranscriptChunk(5, 15, "The circuit diagram explains the current."),
            TranscriptChunk(108, 116, "The meter shows the current."),
        ],
    )
    monkeypatch.setattr(workflow, "YtDlpClient", Client)
    monkeypatch.setattr(workflow.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(
        workflow, "fetch_transcript", lambda *_args, **_kwargs: document
    )
    monkeypatch.setattr(workflow, "vision_embedding_key", lambda: "test-key")

    def candidates(_path, _ffmpeg, *, duration, directory, excluded_intervals):
        return [store_frame(directory, timestamp, chart()) for timestamp in [12, 112]]

    monkeypatch.setattr(workflow, "collect_candidates", candidates)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_artifact_directory_errors_keep_vision_context(semantic_video) -> None:
    archive = semantic_video / ".cache" / "youtube-toolkit"
    archive.mkdir(parents=True)
    (archive / "vision").write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Vision artifact filesystem error") as error:
        workflow.sample_vision(VIDEO_ID, local_only=True)

    assert isinstance(error.value.__cause__, FileExistsError)


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
    assert result["candidate_count"] == 2
    assert result["timestamps_seconds"] == ([] if empty else [112])
    assert result["embedding_usage"] == {
        "image_tokens": 516,
        "text_tokens": 20,
        "total_tokens": 536,
    }
    assert len(result["collage_paths"]) == (0 if empty else 1)
    if not empty:
        assert result["frames"][0]["relevance"]["selection_rank"] == 1
        assert result["frames"][0]["relevance"]["matches"][0]["cosine"] == 1
    assert not list(semantic_video.glob(".cache/youtube-toolkit/vision/*/work-*"))


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
        "download_video_for_vision",
        lambda *_args, **_kwargs: pytest.fail("download must wait for text embeddings"),
    )
    with pytest.raises(EmbeddingError, match="503"):
        workflow.sample_vision(VIDEO_ID)
    assert not (semantic_video / ".cache/youtube-toolkit/vision").exists()


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
    assert result["candidate_count"] == 2
    assert result["embedding_usage"] is None


@pytest.mark.parametrize("local_only", [False, True])
def test_decision_model_ranks_first_with_a_decision_key(
    monkeypatch, semantic_video, local_only
) -> None:
    def score(frames, document, config):
        assert config == DECISION
        assert document.chunks
        scored = [
            FrameDecision(0.9, 0.9, 1.0, 0.2),
            FrameDecision(MIN_SCORE / 2, 0.1, 0.1, 0.1),
        ][: len(frames)]
        return scored, DecisionUsage(200, 0.00002)

    monkeypatch.setattr(workflow, "decision_config", lambda: DECISION)
    monkeypatch.setattr(workflow, "score_frames", score)
    monkeypatch.setattr(
        workflow, "GeminiEmbeddingClient", lambda _key: pytest.fail("no embeddings")
    )
    result = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    if local_only:
        assert result["selection_mode"] == "visual"
        assert result["decision_model"] is None
        return
    assert result["selection_mode"] == "decision"
    assert result["decision_model"] == DEFAULT_DECISION_MODEL
    assert result["decision_usage"] == {"input_tokens": 200, "cost_usd": 0.00002}
    assert result["min_similarity"] is None
    assert result["decision_warning"] is None
    assert result["timestamps_seconds"] == [12]
    assert result["frames"][0]["relevance"] == {
        "selection_rank": 1,
        "score": 0.9,
        "has_information": 0.9,
        "unspoken": 1.0,
        "illustrative": 0.2,
    }


def test_unscored_frames_are_left_out_with_a_warning(
    monkeypatch, semantic_video
) -> None:
    def score(frames, document, config):
        return [FrameDecision(0.9, 0.9, 1.0, 0.2), None], DecisionUsage(100, None)

    monkeypatch.setattr(workflow, "decision_config", lambda: DECISION)
    monkeypatch.setattr(workflow, "score_frames", score)
    result = workflow.sample_vision(VIDEO_ID)
    assert result["timestamps_seconds"] == [12]
    assert result["decision_warning"].startswith("1 of 2 frames went unscored")


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
    allowed: list[bool] = []

    def fetch(*args, allow_stt: bool, **kwargs):
        allowed.append(allow_stt)
        return transcript(*args, **kwargs)

    monkeypatch.setattr(workflow, "fetch_transcript", fetch)
    monkeypatch.setattr(workflow, "vision_embedding_key", lambda: None)
    monkeypatch.setattr(workflow, "local_embedding_installed", lambda: True)
    monkeypatch.setattr(workflow, "LocalImageEmbeddingClient", Embedder)
    result = workflow.sample_vision(VIDEO_ID, local_only=local_only)
    assert result["selection_mode"] == "semantic"
    assert result["embedding_model"] == "example/local-model"
    assert result["min_similarity"] == LOCAL_EMBEDDING_CUTOFFS.vision_match
    assert "not example/local-model" in result["embedding_warning"]
    assert allowed == [not local_only]


@pytest.mark.parametrize("timed", [True, False])
def test_missing_captions_fall_back_to_speech_to_text_for_ranking(
    monkeypatch, semantic_video, timed
) -> None:
    audio = semantic_video / "audio.m4a"

    def download_audio(_self, _ref):
        audio.write_bytes(b"audio")
        return audio

    def no_captions(_metadata, *, languages):
        raise CaptionTranscriptUnavailable("No captions")

    segments = [
        TranscriptChunk(5, 15, "The circuit diagram explains the current."),
        TranscriptChunk(108, 116, "The meter shows the current."),
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
            lambda _path: SpeechTranscription(
                text=text, segments=segments if timed else [], language="en"
            )
        ),
    )
    monkeypatch.setattr(workflow, "GeminiEmbeddingClient", Embedder)

    result = workflow.sample_vision(VIDEO_ID)

    assert not audio.exists()
    if timed:
        assert result["selection_mode"] == "semantic"
        assert result["transcript_warning"] is None
    else:
        assert result["selection_mode"] == "visual"
        assert (
            result["transcript_warning"]
            == "Speech-to-text gave no timestamps; using visual sampling."
        )


def test_auto_clean_off_leaves_video_and_candidates_beside_collages(
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
    media = Path(result["media_dir"])
    assert media.parent == Path(result["collage_paths"][0]).parent
    assert (media / f"{VIDEO_ID}.mp4").read_bytes() == b"video"
    assert len(list((media / "candidates").glob("*.jpg"))) == 2
