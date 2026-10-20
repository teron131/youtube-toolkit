"""The short test recording that both speech-to-text setup checks send through their route.

It uses only the standard library, so the cloud check needs none of the local route's dependencies.
"""

from __future__ import annotations

from array import array
from collections.abc import Generator
from contextlib import contextmanager
import math
from pathlib import Path
import tempfile
import wave

TONE_SECONDS = 2
TONE_SAMPLE_RATE = 16_000
TONE_FREQUENCY_HZ = 440


@contextmanager
def tone_recording() -> Generator[Path]:
    """A tone has no words, so a check that uses it proves a route accepts and processes audio, not that its transcripts are accurate."""
    samples = array(
        "h",
        (
            int(
                3_000
                * math.sin(2 * math.pi * TONE_FREQUENCY_HZ * index / TONE_SAMPLE_RATE)
            )
            for index in range(TONE_SECONDS * TONE_SAMPLE_RATE)
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test-tone.wav"
        with wave.open(str(path), "wb") as tone:
            tone.setnchannels(1)
            tone.setsampwidth(2)
            tone.setframerate(TONE_SAMPLE_RATE)
            tone.writeframes(samples.tobytes())
        yield path
