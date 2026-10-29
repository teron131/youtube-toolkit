"""Local speech-to-text through the Hugging Face transformers automatic-speech-recognition (ASR) pipeline, for the model ID in STT_LOCAL_MODEL.

Nothing here is specific to one model: the same pipeline call runs any Hugging Face ASR model on the best available device.
The one step transformers does not handle uniformly is long audio, so recordings are cut into short pieces first.
A model that is not on this machine yet is downloaded the first time it is needed.
The heavy libraries are imported inside the functions that need them, so commands that never transcribe locally do not pay for loading them.
"""

from __future__ import annotations

from importlib.util import find_spec
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any
import warnings

from ..config import DEFAULT_LOCAL_STT_MODEL, local_stt_model
from ..ffmpeg_utils import SAMPLE_RATE, decode_audio
from ..huggingface import model_downloaded, quiet_transformers, torch_device
from ..schemas import SpeechTranscription, TranscriptChunk
from .synthetic_audio import tone_recording

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
# The pipeline's own chunking is right for Whisper and CTC models but repeats text for Parakeet, and without it Whisper refuses long audio and Parakeet stalls.
# Pieces under Whisper's 30-second window were correct for all three model families, and Parakeet dropped whole sentences from three-minute pieces that it kept at this length.
MAX_PIECE_SECONDS = 28
CUT_SEARCH_SECONDS = 5


def local_stt_installed() -> bool:
    """Installing the libraries and ffmpeg is what switches the local route on; the model itself has a default."""
    try:
        libraries_installed = all(
            find_spec(module) is not None for module in ("transformers", "torch")
        )
    except (ImportError, ValueError):
        return False
    return libraries_installed and shutil.which("ffmpeg") is not None


def local_model_downloaded(model: str) -> bool:
    return local_stt_installed() and model_downloaded(model)


def check_local_stt() -> dict[str, Any]:
    """Run the test tone through the same decoding and recognition as a real recording, downloading the model first if needed.

    A failure anywhere, from the download to ffmpeg to the device, raises instead of returning.
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
    """Each piece becomes one timed segment, so timing is only as fine as the pieces, which works for any model without word timestamps.

    The language is left for the model to work out, as with the cloud providers, so the result carries none.
    """
    pieces = split_at_quiet_points(decode_audio(audio_path))
    recognizer = load_recognizer()
    # Pieces run one at a time: batching them on an Apple GPU returned truncated text.
    results = recognizer(
        [{"raw": piece, "sampling_rate": SAMPLE_RATE} for piece in pieces]
    )
    segments = []
    offset = 0
    for piece, result in zip(pieces, results, strict=True):
        if piece_text := result["text"].strip():
            segments.append(
                TranscriptChunk(
                    start_seconds=offset / SAMPLE_RATE,
                    end_seconds=(offset + len(piece)) / SAMPLE_RATE,
                    text=piece_text,
                )
            )
        offset += len(piece)
    if not segments:
        raise ValueError("Local speech-to-text found no speech")
    return SpeechTranscription(
        text=" ".join(segment.text for segment in segments), segments=segments
    )


def load_recognizer() -> Any:
    """A configured model that cannot be loaded falls back to the default with a warning, so a mistyped or unsuitable STT_LOCAL_MODEL still produces a transcript."""
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
    quiet_transformers(downloaded=local_model_downloaded(model))
    import torch
    from transformers import pipeline

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    device = torch_device()
    # GPUs run half precision fastest and in half the memory; CPUs need full precision.
    dtype = torch.float32 if device == "cpu" else torch.float16
    return pipeline(
        "automatic-speech-recognition", model=model, device=device, dtype=dtype
    )


def split_at_quiet_points(
    samples: np.ndarray, *, max_seconds: int = MAX_PIECE_SECONDS
) -> list[np.ndarray]:
    """Each cut lands on the quietest tenth of a second in the last CUT_SEARCH_SECONDS before the limit, usually a pause between sentences, so words are not split.

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
