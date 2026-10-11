"""Local speech-to-text through the Hugging Face transformers automatic-speech-recognition (ASR) pipeline, for the model ID in STT_LOCAL_MODEL.

Nothing here is specific to one model: the same pipeline call runs any Hugging Face ASR model on an Nvidia GPU, then the Apple GPU, then the CPU.
The one step transformers does not handle uniformly is long audio, so recordings are cut into short pieces first.
A model that is not on this machine yet is downloaded the first time it is needed.
The heavy libraries are imported inside the functions that need them, so commands that never transcribe locally do not pay for loading them.
"""

from __future__ import annotations

from importlib.util import find_spec
import logging
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Any
import warnings

from ..config import DEFAULT_LOCAL_STT_MODEL, local_stt_model
from ..schemas import SpeechTranscription
from .synthetic_audio import tone_recording

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
# The pipeline's own chunking is right for Whisper and CTC models but repeats text for Parakeet, and without it Whisper refuses long audio and Parakeet stalls.
# Pieces under Whisper's 30-second window were correct for all three model families, and Parakeet dropped whole sentences from three-minute pieces that it kept at this length.
MAX_PIECE_SECONDS = 28
# How far back from a piece's length limit to look for the quietest moment to cut at.
CUT_SEARCH_SECONDS = 5
# Decoding even a multi-hour track to PCM takes well under a minute, so ten minutes means ffmpeg is stuck.
FFMPEG_TIMEOUT_SECONDS = 10 * 60


def local_stt_installed() -> bool:
    """Check that the model libraries and ffmpeg can be located without loading the libraries.

    Installing them is what switches the local route on; the model itself has a default.
    """
    try:
        libraries_installed = all(
            find_spec(module) is not None for module in ("transformers", "torch")
        )
    except (ImportError, ValueError):
        return False
    return libraries_installed and shutil.which("ffmpeg") is not None


def local_model_downloaded(model: str) -> bool:
    """Report whether a model is already on this machine, without loading it."""
    if not local_stt_installed():
        return False
    if Path(model).is_dir():
        return True
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


def check_local_stt() -> dict[str, Any]:
    """Load the configured model and run the test tone through the same decoding and recognition as a real recording.

    A failure anywhere, from the download to ffmpeg to the device, raises instead of returning.
    This shows the route works on this machine; it does not measure transcript quality.
    """
    recognizer = load_recognizer()
    with tone_recording() as tone_path:
        samples = decode_audio(tone_path)
    recognizer({"raw": samples, "sampling_rate": SAMPLE_RATE})
    return {
        "model": recognizer.model.name_or_path,
        "device": str(recognizer.device),
        "precision": str(recognizer.model.dtype).removeprefix("torch."),
        "working": True,
    }


def transcribe_locally(audio_path: Path) -> SpeechTranscription:
    """Transcribe one audio file on this machine with the configured model.

    The language is left for the model to work out, as with the cloud providers, so the result carries none.
    """
    pieces = split_at_quiet_points(decode_audio(audio_path))
    recognizer = load_recognizer()
    # Pieces run one at a time: batching them on an Apple GPU returned truncated text.
    results = recognizer(
        [{"raw": piece, "sampling_rate": SAMPLE_RATE} for piece in pieces]
    )
    text = " ".join(
        piece_text for result in results if (piece_text := result["text"].strip())
    )
    if not text:
        raise ValueError("Local speech-to-text found no speech")
    return SpeechTranscription(text=text)


def load_recognizer() -> Any:
    """Build the speech recognizer for the configured model, falling back to the default model with a warning when the configured one cannot be loaded.

    A mistyped or unsuitable STT_LOCAL_MODEL therefore still produces a transcript, from the default model, and the warning on stderr says so.
    """
    model = local_stt_model()
    try:
        return _build_recognizer(model)
    except (OSError, ValueError) as error:
        if model == DEFAULT_LOCAL_STT_MODEL:
            raise
        logger.warning(
            "STT_LOCAL_MODEL=%s could not be loaded (%s); falling back to the default %s",
            model,
            str(error).splitlines()[0],
            DEFAULT_LOCAL_STT_MODEL,
        )
        return _build_recognizer(DEFAULT_LOCAL_STT_MODEL)


def _build_recognizer(model: str) -> Any:
    """Build the transformers ASR pipeline for one model on the fastest device this machine has.

    transformers downloads a model that is not cached yet, showing its progress on stderr, and reuses the cache afterwards.
    """
    downloaded = local_model_downloaded(model)
    import torch
    from transformers import pipeline
    from transformers.utils import logging as transformers_logging

    # The CLI prints JSON on stdout and errors on stderr; library chatter belongs in neither, except progress for a download that can take minutes.
    transformers_logging.set_verbosity_error()
    if downloaded:
        transformers_logging.disable_progress_bar()
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # GPUs run half precision fastest and in half the memory; CPUs need full precision.
    dtype = torch.float32 if device == "cpu" else torch.float16
    return pipeline(
        "automatic-speech-recognition", model=model, device=device, dtype=dtype
    )


def decode_audio(audio_path: Path) -> np.ndarray:
    """Decode any audio container to the 16 kHz mono samples speech models read.

    ffmpeg reads the file itself because some containers cannot be decoded from a pipe, which is how transformers would feed it.
    """
    import numpy as np

    try:
        decoded = subprocess.run(  # noqa: S603 - arguments are internally assembled and the input is a file this process downloaded.
            [  # noqa: S607 - ffmpeg is resolved from the local PATH, as local_stt_installed checks.
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
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
            capture_output=True,
            check=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
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


def split_at_quiet_points(
    samples: np.ndarray, *, max_seconds: int = MAX_PIECE_SECONDS
) -> list[np.ndarray]:
    """Cut audio into consecutive pieces no longer than max_seconds without splitting words.

    Each cut lands on the quietest tenth of a second in the last few seconds before the limit, which is usually a pause between sentences.
    Pieces do not overlap, so their transcripts join without any merging.
    """
    limit = max_seconds * SAMPLE_RATE
    search_window = CUT_SEARCH_SECONDS * SAMPLE_RATE
    frame = SAMPLE_RATE // 10
    pieces: list[np.ndarray] = []
    start = 0
    while len(samples) - start > limit:
        search_start = start + limit - search_window
        loudness = abs(samples[search_start : start + limit]).reshape(-1, frame)
        quietest_frame = int(loudness.mean(axis=1).argmin())
        cut = search_start + quietest_frame * frame + frame // 2
        pieces.append(samples[start:cut])
        start = cut
    pieces.append(samples[start:])
    return pieces
