"""Decode video keyframes into RGB frames and audio into mono PCM through bounded FFmpeg subprocesses.

NumPy loads only when decoding is requested, keeping imports usable without the optional media dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Decoding a two-hour video's keyframes or audio track takes well under a minute, so ten minutes means FFmpeg is stuck.
FFMPEG_TIMEOUT_SECONDS = 10 * 60
FFMPEG_THREADS = min(4, os.process_cpu_count() or 1)
# Vision models still read on-screen numbers reliably at 960x540; in a transcription test, 640x360 frames lost about a third of them.
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FRAME_FILTER = f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
SHOWINFO_PTS_TIME = re.compile(rb"Parsed_showinfo[^\n]*?pts_time:\s*(-?[0-9.]+)")
SAMPLE_RATE = 16_000


def run_ffmpeg(
    ffmpeg_path: str, arguments: list[str], *, loglevel: str = "error"
) -> subprocess.CompletedProcess[bytes]:
    """Process failures propagate so each caller can report its audio or video operation."""
    return subprocess.run(  # noqa: S603 -- The executable and arguments are assembled by local media callers.
        [ffmpeg_path, "-hide_banner", "-loglevel", loglevel, *arguments],
        capture_output=True,
        check=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
    )


def decode_keyframes(
    video_path: Path, start: float, stop: float, ffmpeg_path: str
) -> tuple[list[float], np.ndarray]:
    """Decode only keyframes, which need no earlier frames, so cost follows how many there are rather than every frame in the window.

    Timestamps are absolute seconds read from FFmpeg's showinfo log; a count that disagrees with the decoded frames raises ValueError.
    """
    completed = _run_video_ffmpeg(
        ffmpeg_path,
        [
            "-nostats",
            "-threads",
            str(FFMPEG_THREADS),
            "-filter_threads",
            "1",
            "-y",
            "-skip_frame",
            "nokey",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(stop - start),
            "-vf",
            f"showinfo,{FRAME_FILTER}",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        loglevel="info",
    )
    frames = _rgb_frames(completed.stdout)
    # showinfo also logs the first keyframe past the window, which -t then drops from the output.
    times = [
        start + offset
        for offset in map(float, SHOWINFO_PTS_TIME.findall(completed.stderr))
        if 0 <= offset < stop - start
    ]
    if len(times) != len(frames):
        raise ValueError(
            f"FFmpeg reported {len(times)} keyframe timestamps for {len(frames)} frames"
        )
    return times, frames


def _run_video_ffmpeg(
    ffmpeg_path: str, arguments: list[str], *, loglevel: str = "error"
) -> subprocess.CompletedProcess[bytes]:
    try:
        return run_ffmpeg(ffmpeg_path, arguments, loglevel=loglevel)
    except subprocess.CalledProcessError as error:
        # Info-level keyframe runs log every frame, and FFmpeg reports the failure in the final lines.
        lines = error.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            "\n".join(lines[-5:]) or "FFmpeg vision processing failed"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"FFmpeg vision processing timed out after {FFMPEG_TIMEOUT_SECONDS} seconds"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Could not start FFmpeg vision processing: {error}"
        ) from error


def _rgb_frames(output: bytes) -> np.ndarray:
    import numpy as np

    pixels = np.frombuffer(output, dtype=np.uint8)
    frame_bytes = FRAME_HEIGHT * FRAME_WIDTH * 3
    if pixels.size % frame_bytes:
        raise ValueError(f"FFmpeg returned incomplete RGB frames ({pixels.size} bytes)")
    return pixels.reshape(-1, FRAME_HEIGHT, FRAME_WIDTH, 3)


def decode_audio(audio_path: Path) -> np.ndarray:
    """ffmpeg reads the file itself because some containers cannot be decoded from a pipe, which is how transformers would feed it."""
    import numpy as np

    try:
        decoded = run_ffmpeg(
            "ffmpeg",
            [
                "-nostdin",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "f32le",
                "-",
            ],
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg could not decode audio for speech-to-text: {detail}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"ffmpeg audio decoding timed out after {FFMPEG_TIMEOUT_SECONDS} seconds"
        ) from error
    samples = np.frombuffer(decoded.stdout, dtype=np.float32)
    if not len(samples):
        raise ValueError("Audio for speech-to-text is empty")
    return samples
