"""FFmpeg helpers keep bounded subprocess options, shape decoded output, and report failures with the operation they belong to."""

import subprocess

import numpy as np
import pytest

from youtube_toolkit import ffmpeg_utils


def test_decode_audio_keeps_mono_pcm_options_and_float32_samples(
    monkeypatch, tmp_path
) -> None:
    samples = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    calls = []

    def run(command, *, capture_output, check, timeout):
        assert capture_output is True
        assert check is True
        assert timeout == ffmpeg_utils.FFMPEG_TIMEOUT_SECONDS == 600
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, samples.tobytes(), b"")

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", run)
    path = tmp_path / "audio.webm"
    decoded = ffmpeg_utils.decode_audio(path)
    (command,) = calls
    assert command[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert "-nostdin" in command
    assert command[command.index("-i") + 1] == str(path)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-f") + 1] == "f32le"
    assert command[-1] == "-"
    assert "-vf" not in command
    assert "-threads" not in command
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, samples)


def test_decode_audio_rejects_empty_pcm(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ffmpeg_utils,
        "run_ffmpeg",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    with pytest.raises(ValueError, match="Audio for speech-to-text is empty"):
        ffmpeg_utils.decode_audio(tmp_path / "audio.webm")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"invalid audio"),
            "ffmpeg could not decode audio for speech-to-text: invalid audio",
        ),
        (
            subprocess.TimeoutExpired(["ffmpeg"], timeout=600),
            "ffmpeg audio decoding timed out after 600 seconds",
        ),
    ],
)
def test_decode_audio_preserves_context_on_subprocess_failure(
    monkeypatch, tmp_path, error, message
) -> None:
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match=message):
        ffmpeg_utils.decode_audio(tmp_path / "audio.webm")


def showinfo_log(*offsets: float) -> bytes:
    return b"".join(
        f"[Parsed_showinfo_0 @ 0x1] n:{n} pts:{n} pts_time:{offset} duration:1\n".encode()
        for n, offset in enumerate(offsets)
    )


def test_decode_keyframes_skips_other_frames_and_keeps_window_timestamps(
    monkeypatch, tmp_path
) -> None:
    pixels = np.zeros(
        (2, ffmpeg_utils.FRAME_HEIGHT, ffmpeg_utils.FRAME_WIDTH, 3), dtype=np.uint8
    )
    calls = []

    def run(command, *, capture_output, check, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, pixels.tobytes(), showinfo_log(1.5, 6.5, 11.5)
        )

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", run)
    times, frames = ffmpeg_utils.decode_keyframes(
        tmp_path / "v.mp4", 60, 70, "/resolved/ffmpeg"
    )
    (command,) = calls
    assert command[command.index("-loglevel") + 1] == "info"
    assert command[command.index("-skip_frame") + 1] == "nokey"
    assert command.index("-skip_frame") < command.index("-i")
    assert command[command.index("-vf") + 1].startswith("showinfo,scale=960:540")
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-threads") + 1] == str(ffmpeg_utils.FFMPEG_THREADS)
    assert command[command.index("-ss") + 1] == "60"
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-t") + 1] == "10"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    assert times == [61.5, 66.5]
    assert frames.shape == pixels.shape


def test_decode_keyframes_rejects_timestamp_count_mismatch(
    monkeypatch, tmp_path
) -> None:
    pixels = np.zeros(
        (2, ffmpeg_utils.FRAME_HEIGHT, ffmpeg_utils.FRAME_WIDTH, 3), dtype=np.uint8
    )
    monkeypatch.setattr(
        ffmpeg_utils,
        "run_ffmpeg",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, pixels.tobytes(), showinfo_log(1.5)
        ),
    )
    with pytest.raises(ValueError, match="1 keyframe timestamps for 2 frames"):
        ffmpeg_utils.decode_keyframes(tmp_path / "v.mp4", 0, 10, "ffmpeg")


def test_decode_keyframes_rejects_truncated_rgb(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        ffmpeg_utils,
        "run_ffmpeg",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"bad", b""),
    )
    with pytest.raises(ValueError, match="incomplete RGB"):
        ffmpeg_utils.decode_keyframes(tmp_path / "v.mp4", 0, 10, "ffmpeg")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            subprocess.TimeoutExpired(["ffmpeg"], timeout=1),
            "FFmpeg vision processing timed out",
        ),
        (OSError("cannot execute"), "Could not start FFmpeg vision processing"),
    ],
)
def test_decode_keyframes_failures_become_user_facing_runtime_errors(
    monkeypatch, tmp_path, error, message
) -> None:
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match=message):
        ffmpeg_utils.decode_keyframes(tmp_path / "v.mp4", 0, 10, "ffmpeg")
