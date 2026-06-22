from youtube_toolkit.stt import SpeechTranscription


def test_xai_stt_response_normalizes_to_plain_contract() -> None:
    result = SpeechTranscription.from_xai_stt(
        {
            "text": "transcript text",
            "language": "en",
            "duration": 42.5,
            "words": [{"word": "transcript"}, {"word": "text"}],
        }
    )

    assert result == SpeechTranscription(
        text="transcript text",
        language="en",
        duration_seconds=42.5,
        word_count=2,
    )


def test_elevenlabs_stt_response_normalizes_to_plain_contract() -> None:
    result = SpeechTranscription.from_elevenlabs_stt(
        {
            "text": "transcript text",
            "language_code": "en",
            "words": [
                {"text": "transcript", "start": 0, "end": 0.5, "type": "word"},
                {"text": "text", "start": 0.5, "end": 1.0, "type": "word"},
            ],
        }
    )

    assert result == SpeechTranscription(
        text="transcript text",
        language="en",
        duration_seconds=1.0,
        word_count=2,
    )
