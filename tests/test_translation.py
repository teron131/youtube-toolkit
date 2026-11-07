"""Transcript translation tests: same-language passthrough, caption lines regrouped into sentences, plain-text batches matched by position, untimed speech, and replies that break the line count."""

import re

import pytest

from youtube_toolkit import execution, translation
from youtube_toolkit.config import LLMConfig
from youtube_toolkit.schemas import TranscriptChunk, TranscriptDocument, VideoMetadata
from youtube_toolkit.translation import TranslationError, translate_transcript

CONFIG = LLMConfig(
    base_url="https://gateway.example/v1", api_key="key", model="example/model"
)
METADATA = VideoMetadata(video_id="JpSHyEIZ_bo", url="https://youtu.be/JpSHyEIZ_bo")
BLOCK = re.compile(r"<(before|translate|after)[^>]*>\n(.*?)\n</\1>", re.DOTALL)


def fake_openai(monkeypatch, answer):
    """`answer` maps one request's lines to translate onto the reply text the fake model returns."""
    requests = []

    class Completions:
        def create(self, **kwargs):
            blocks = {
                name: [line for line in text.split("\n") if line]
                for name, text in BLOCK.findall(kwargs["messages"][1]["content"])
            }
            request = {**blocks, "system": kwargs["messages"][0]["content"]}
            requests.append(request)
            message = type("Message", (), {"content": answer(request["translate"])})
            return type(
                "Response", (), {"choices": [type("Choice", (), {"message": message})]}
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            pass

    monkeypatch.setattr(translation, "OpenAI", Client)
    return requests


def prefixed(lines: list[str]) -> str:
    return "\n".join(f"T {line}" for line in lines)


def test_transcripts_already_in_the_target_language_come_back_unchanged(
    monkeypatch,
) -> None:
    requests = fake_openai(monkeypatch, prefixed)
    document = TranscriptDocument(
        METADATA, "en-orig", "automatic_captions", [TranscriptChunk(0, 1, "Hi")]
    )

    assert translate_transcript(document, "en", CONFIG) is document
    assert requests == []
    assert translate_transcript(document, "EN-us", CONFIG) is document


def test_caption_lines_are_regrouped_into_timed_sentences(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, prefixed)
    lines = [
        "I want to actually",
        "select the device. He said,",
        '"Stop. Go home." Then we',
        "call page.",
        "close and see a ton of [ __ ] Because",
        "it broke.",
    ]
    chunks = [
        TranscriptChunk(index, index + 1, text) for index, text in enumerate(lines)
    ]
    limit = translation.MAX_LINES_PER_SENTENCE
    chunks += [
        TranscriptChunk(10 + index, 11 + index, "Okay") for index in range(limit + 1)
    ]
    document = TranscriptDocument(METADATA, "es-orig", "automatic_captions", chunks)

    result = translate_transcript(document, "en", CONFIG)

    # A quotation holds its own sentences, a lowercase word continues the sentence, and a censored word ends one before a capital.
    assert requests[0]["translate"] == [
        "I want to actually select the device.",
        'He said, "Stop. Go home."',
        "Then we call page. close and see a ton of [ __ ]",
        "Because it broke.",
        " ".join(["Okay"] * limit),
        "Okay",
    ]
    assert [(chunk.start_seconds, chunk.end_seconds) for chunk in result.chunks] == [
        (0, 2),
        (1, 3),
        (2, 5),
        (4, 6),
        (10, 10 + limit),
        (10 + limit, 11 + limit),
    ]


def test_sentences_keep_their_timing_across_parallel_batches(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, prefixed)
    count = translation.SENTENCES_PER_BATCH + 2
    chunks = [TranscriptChunk(index, index + 1, f"S{index}.") for index in range(count)]
    document = TranscriptDocument(METADATA, "es", "subtitles", chunks)

    result = translate_transcript(document, "zh-TW", CONFIG)

    assert result.language == "zh-TW"
    assert requests[0]["system"].count("zh-TW") == 1
    assert result.translated_by == "example/model"
    assert result.chunks == [
        TranscriptChunk(index, index + 1, f"T S{index}.") for index in range(count)
    ]
    last = next(
        request for request in requests if f"S{count - 1}." in request["translate"]
    )
    assert last["translate"] == [f"S{count - 2}.", f"S{count - 1}."]
    assert last["before"] == [f"S{count - 4}.", f"S{count - 3}."]
    assert last["after"] == []


def test_untimed_speech_is_translated_by_sentence_and_joined_back(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, lambda lines: "\n".join(lines).upper())
    document = TranscriptDocument(
        METADATA,
        "speech-to-text",
        "speech-to-text",
        [TranscriptChunk(0, 0, "One. Two! 三。四。")],
    )

    result = translate_transcript(document, "en", CONFIG)

    assert requests[0]["translate"] == ["One.", "Two!", "三。", "四。"]
    assert result.chunks == [TranscriptChunk(0, 0, "ONE. TWO! 三。 四。")]


def test_a_reply_with_another_line_count_becomes_one_entry_for_its_batch(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_openai(monkeypatch, lambda lines: " ".join(lines))
    chunks = [TranscriptChunk(index, index + 1, f"S{index}.") for index in range(3)]
    document = TranscriptDocument(METADATA, "es", "subtitles", chunks)

    result = translate_transcript(document, "en", CONFIG)

    assert result.chunks == [TranscriptChunk(0, 3, "S0. S1. S2.")]


def test_a_batch_that_keeps_replying_empty_fails_the_whole_run(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(execution.time, "sleep", lambda _seconds: None)
    requests = fake_openai(monkeypatch, lambda _lines: "\n")
    chunks = [TranscriptChunk(index, index + 1, f"S{index}.") for index in range(2)]
    document = TranscriptDocument(METADATA, "es", "subtitles", chunks)

    with pytest.raises(TranslationError, match=r"sentences 1-2 of 2.*no translation"):
        translate_transcript(document, "en", CONFIG)
    assert len(requests) == translation.ATTEMPTS
