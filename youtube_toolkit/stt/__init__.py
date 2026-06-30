"""Public speech-to-text package surface."""

from .common import (
    SpeechToTextOptions,
    transcribe_audio,
    transcribe_audio_details,
)
from .schemas import SpeechTranscription

__all__ = [
    "SpeechToTextOptions",
    "SpeechTranscription",
    "transcribe_audio",
    "transcribe_audio_details",
]
