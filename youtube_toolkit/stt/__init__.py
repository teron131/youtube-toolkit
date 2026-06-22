"""Public speech-to-text package surface."""

from .common import (
    SpeechToTextOptions,
    transcribe_audio_file,
)
from .schemas import SpeechTranscription

__all__ = [
    "SpeechToTextOptions",
    "SpeechTranscription",
    "transcribe_audio_file",
]
