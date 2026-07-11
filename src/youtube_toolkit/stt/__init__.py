"""Public speech-to-text package surface."""

from .transcription import (
    SpeechToTextOptions,
    SpeechTranscription,
    transcribe_audio,
    transcribe_audio_details,
)

__all__ = [
    "SpeechToTextOptions",
    "SpeechTranscription",
    "transcribe_audio",
    "transcribe_audio_details",
]
