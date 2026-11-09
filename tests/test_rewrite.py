"""Transcript rewrite tests: same-language passthrough, caption lines rewritten in place and matched by line ID, untimed speech by sentence, replies that merge, drop, label, or drift from their lines, sentence splitting, and articles whose paragraphs start at turning points and chapter starts."""

import re

import httpx2
import openai
import pytest

from youtube_toolkit.config import LLMConfig
from youtube_toolkit.rewrite import RewriteError, article, rewrite_transcript
from youtube_toolkit.rewrite.requests import ATTEMPTS
from youtube_toolkit.rewrite.sentences import MAX_LINES_PER_SENTENCE, split_sentences
from youtube_toolkit.rewrite.timestamped import CONTEXT_LINES, LINES_PER_BATCH
from youtube_toolkit.schemas import (
    TranscriptDocument,
    TranscriptRewrite,
    TranscriptSegment,
    VideoChapter,
    VideoMetadata,
)

CONFIG = LLMConfig(
    base_url="https://gateway.example/v1", api_key="key", model="example/model"
)
METADATA = VideoMetadata(video_id="JpSHyEIZ_bo", url="https://youtu.be/JpSHyEIZ_bo")
BLOCK = re.compile(r"<(before|lines|after)[^>]*>\n(.*?)\n</\1>", re.DOTALL)
LINE_ID = re.compile(r"^\[LINE\d+\] ")


def fake_openai(monkeypatch, answer, starts=lambda _lines: "NONE"):
    """`answer` maps one rewrite request's lines, without their IDs, onto the reply text the fake model returns, and `starts` does the same for paragraph-start requests."""
    requests = []

    class Completions:
        def create(self, **kwargs):
            system = kwargs["messages"][0]["content"]
            blocks = {
                name: [LINE_ID.sub("", line) for line in text.split("\n") if line]
                for name, text in BLOCK.findall(kwargs["messages"][1]["content"])
            }
            request = {
                **blocks,
                "system": system,
                "user": kwargs["messages"][1]["content"],
            }
            requests.append(request)
            reply = (starts if "turning points" in system else answer)(request["lines"])
            message = type("Message", (), {"content": reply})
            return type(
                "Response", (), {"choices": [type("Choice", (), {"message": message})]}
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})

    monkeypatch.setattr("youtube_toolkit.llm.OpenAI", Client)
    return requests


def translate(document: TranscriptDocument, language: str) -> TranscriptDocument:
    return rewrite_transcript(
        document, CONFIG, language=language, refine=False, article=False
    )


def numbered(texts) -> str:
    return "\n".join(f"[LINE{number}] {text}" for number, text in enumerate(texts, 1))


def prefixed(lines: list[str]) -> str:
    return numbered(f"T {line}" for line in lines)


def test_transcripts_already_in_the_target_language_come_back_unchanged(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, prefixed)
    document = TranscriptDocument(
        METADATA, "en-orig", "automatic_captions", [TranscriptSegment(0, 1, "Hi")]
    )

    assert translate(document, "en") is document
    assert requests == []
    assert translate(document, "EN-us") is document
    refined = rewrite_transcript(
        document, CONFIG, language="en", refine=True, article=False
    )
    assert refined.segments == [TranscriptSegment(0, 1, "T Hi")]
    assert refined.rewrite == TranscriptRewrite("example/model", translated=False)
    assert "Correct YouTube transcript lines" in requests[0]["system"]


def test_requests_list_the_names_said_more_than_once(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, prefixed)
    segments = [
        TranscriptSegment(0, 1, "We tried Zentrix today."),
        TranscriptSegment(1, 2, "Then Zentrix beat Opus."),
    ]
    document = TranscriptDocument(METADATA, "es", "subtitles", segments)

    translate(document, "en")

    assert "\nNames said more than once: Zentrix\n" in requests[0]["user"]


def test_caption_lines_split_into_sentences_at_their_end_marks() -> None:
    lines = [
        "I want to actually",
        "select the device. He said,",
        '"Stop. Go home." Then we',
        "call page.",
        "close and see a ton of [ __ ] Because",
        "it broke.",
    ]
    segments = [
        TranscriptSegment(index, index + 1, text) for index, text in enumerate(lines)
    ]
    segments += [
        TranscriptSegment(10 + index, 11 + index, "Okay")
        for index in range(MAX_LINES_PER_SENTENCE + 1)
    ]

    sentences = split_sentences(segments)

    # A quotation holds its own sentences, a lowercase word continues the sentence, and a censored word ends one before a capital.
    assert [sentence.text for sentence in sentences] == [
        "I want to actually select the device.",
        'He said, "Stop. Go home."',
        "Then we call page. close and see a ton of [ __ ]",
        "Because it broke.",
        " ".join(["Okay"] * MAX_LINES_PER_SENTENCE),
        "Okay",
    ]
    assert [
        (sentence.start_seconds, sentence.end_seconds) for sentence in sentences
    ] == [
        (0, 2),
        (1, 3),
        (2, 5),
        (4, 6),
        (10, 10 + MAX_LINES_PER_SENTENCE),
        (10 + MAX_LINES_PER_SENTENCE, 11 + MAX_LINES_PER_SENTENCE),
    ]


def test_caption_lines_keep_their_own_timing_across_parallel_batches(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, prefixed)
    count = LINES_PER_BATCH + 2
    segments = [
        TranscriptSegment(index, index + 1, f"L{index}") for index in range(count)
    ]
    document = TranscriptDocument(METADATA, "es", "subtitles", segments)

    result = translate(document, "zh-TW")

    assert result.language == "zh-TW"
    assert requests[0]["system"].count("zh-TW") == 1
    assert result.rewrite == TranscriptRewrite("example/model", translated=True)
    assert result.segments == [
        TranscriptSegment(index, index + 1, f"T L{index}") for index in range(count)
    ]
    last = next(request for request in requests if f"L{count - 1}" in request["lines"])
    assert last["lines"] == [f"L{count - 2}", f"L{count - 1}"]
    assert last["before"] == [
        f"L{index}" for index in range(count - 2 - CONTEXT_LINES, count - 2)
    ]
    assert last["after"] == []


def test_untimed_speech_is_translated_by_sentence_and_joined_back(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(
        monkeypatch, lambda lines: numbered(line.upper() for line in lines)
    )
    document = TranscriptDocument(
        METADATA,
        "speech-to-text",
        "speech-to-text",
        [TranscriptSegment(0, 0, "One. Two! 三。四。")],
    )

    result = translate(document, "en")

    assert requests[0]["lines"] == ["One.", "Two!", "三。", "四。"]
    assert result.segments == [TranscriptSegment(0, 0, "ONE. TWO! 三。 四。")]


def test_a_reply_missing_an_id_is_asked_for_again(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    replies = iter(["[LINE1] T S0. S1.", "[LINE1] T S0.\n[LINE2] T S1."])
    requests = fake_openai(monkeypatch, lambda _lines: next(replies))
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(2)]
    document = TranscriptDocument(METADATA, "es", "subtitles", segments)

    result = translate(document, "en")

    assert result.segments == [
        TranscriptSegment(0, 1, "T S0."),
        TranscriptSegment(1, 2, "T S1."),
    ]
    assert len(requests) == 2


def test_replies_drop_labels_and_stray_ids_and_join_lines_without_one(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_openai(
        monkeypatch,
        lambda _lines: (
            "Translation:\n[LINE1] T S0\n[2] continued.\n[LINE2] T S1.\n[LINE3] T after."
        ),
    )
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(2)]
    document = TranscriptDocument(METADATA, "es", "subtitles", segments)

    result = translate(document, "en")

    assert result.segments == [
        TranscriptSegment(0, 1, "T S0 [2] continued."),
        TranscriptSegment(1, 2, "T S1."),
    ]


def test_a_translated_line_merged_into_its_neighbour_shares_its_entry(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(
        monkeypatch, lambda _lines: "[LINE2] T S0. S1.\n[LINE3] T S2."
    )
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(3)]
    document = TranscriptDocument(METADATA, "es", "subtitles", segments)

    result = translate(document, "en")

    assert result.segments == [
        TranscriptSegment(0, 2, "T S0. S1."),
        TranscriptSegment(2, 3, "T S2."),
    ]
    assert len(requests) == ATTEMPTS


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "[LINE1] The cat sat, and dogs bark.\n[LINE3] Birds fly!",
            [
                TranscriptSegment(0, 2, "The cat sat, and dogs bark."),
                TranscriptSegment(2, 3, "Birds fly!"),
            ],
        ),
        (
            "[LINE1] The cat sat down.\n[LINE3] Birds fly!",
            [
                TranscriptSegment(0, 1, "The cat sat down."),
                TranscriptSegment(1, 2, "Dogs bark."),
                TranscriptSegment(2, 3, "Birds fly!"),
            ],
        ),
    ],
    ids=["merged", "dropped"],
)
def test_a_cleaned_up_line_missing_from_the_reply_merges_or_keeps_its_original(
    monkeypatch, tmp_path, reply, expected
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_openai(monkeypatch, lambda _lines: reply)
    segments = [
        TranscriptSegment(0, 1, "The cat sat."),
        TranscriptSegment(1, 2, "Dogs bark."),
        TranscriptSegment(2, 3, "Birds fly."),
    ]
    document = TranscriptDocument(METADATA, "en", "automatic_captions", segments)

    result = rewrite_transcript(
        document, CONFIG, language=None, refine=True, article=False
    )

    assert result.segments == expected


def test_a_corrected_line_unlike_its_source_keeps_the_original(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_openai(
        monkeypatch,
        lambda _lines: "[LINE1] The cat sat down.\n[LINE2] Something else entirely.",
    )
    segments = [
        TranscriptSegment(0, 1, "The cat sat."),
        TranscriptSegment(1, 2, "Dogs bark."),
    ]
    document = TranscriptDocument(METADATA, "en", "automatic_captions", segments)

    result = rewrite_transcript(
        document, CONFIG, language=None, refine=True, article=False
    )

    assert [segment.text for segment in result.segments] == [
        "The cat sat down.",
        "Dogs bark.",
    ]


def test_an_article_rewrites_paragraphs_under_translated_chapter_titles(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(article, "SENTENCES_PER_BATCH", 2)
    requests = fake_openai(monkeypatch, prefixed, starts=lambda _lines: "[LINE4]")
    metadata = VideoMetadata(
        video_id="JpSHyEIZ_bo",
        url="https://youtu.be/JpSHyEIZ_bo",
        chapters=[VideoChapter("Main", 2, None), VideoChapter("Intro", 0, 2)],
    )
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(5)]
    document = TranscriptDocument(metadata, "es", "subtitles", segments)

    result = rewrite_transcript(
        document, CONFIG, language="ja", refine=False, article=True
    )

    # Chapter starts open paragraphs at S0 and S2, and the model's turning point opens one at S3.
    assert result.segments == [
        TranscriptSegment(
            0, 5, "## T Intro\n\nT S0. S1.\n\n## T Main\n\nT S2.\n\nT S3. S4."
        )
    ]
    rewrites = [
        request["lines"]
        for request in requests
        if "paragraph of speech" in request["system"]
    ]
    assert sorted(rewrites) == [["S0. S1."], ["S2."], ["S3. S4."]]
    assert result.language == "ja"


def test_a_chapter_heading_moves_to_the_sentence_the_model_says_opens_it(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    # Main's timestamp falls on S2 and the model moves it to its lead-in S1, while End's answer lands before Main and leaves End on its timestamp.
    requests = fake_openai(
        monkeypatch,
        prefixed,
        starts=lambda _lines: "NONE\nCHAPTER2 [LINE2]\nCHAPTER3 [LINE2]",
    )
    metadata = VideoMetadata(
        video_id="JpSHyEIZ_bo",
        url="https://youtu.be/JpSHyEIZ_bo",
        chapters=[
            VideoChapter("Intro", 0, 2),
            VideoChapter("Main", 2, 4),
            VideoChapter("End", 4, None),
        ],
    )
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(6)]
    document = TranscriptDocument(metadata, "es", "subtitles", segments)

    result = rewrite_transcript(
        document, CONFIG, language="ja", refine=False, article=True
    )

    window = next(
        request for request in requests if "turning points" in request["system"]
    )
    assert window["lines"][2:4] == ['<chapter id="CHAPTER2">Main</chapter>', "S2."]
    assert result.segments[0].text == (
        "## T Intro\n\nT S0.\n\n## T Main\n\nT S1. S2. S3.\n\n## T End\n\nT S4. S5."
    )


def test_each_paragraph_window_rules_only_on_its_own_sentences(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(article, "SENTENCES_PER_WINDOW", 3)
    monkeypatch.setattr(article, "WINDOW_CONTEXT", 1)
    # Every window names sentence 3 and sentence 5, which only one window each may rule on.
    requests = fake_openai(
        monkeypatch, prefixed, starts=lambda _lines: "[LINE3] [LINE5]"
    )
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(6)]
    document = TranscriptDocument(METADATA, "en", "automatic_captions", segments)

    result = rewrite_transcript(
        document, CONFIG, language=None, refine=True, article=True
    )

    windows = [
        request["lines"]
        for request in requests
        if "turning points" in request["system"]
    ]
    assert sorted(windows) == [
        ["S0.", "S1.", "S2.", "S3."],
        ["S2.", "S3.", "S4.", "S5."],
    ]
    assert result.segments[0].text == "T S0. S1.\n\nT S2. S3.\n\nT S4. S5."


def test_a_refined_article_batch_that_shrinks_keeps_its_source(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, lambda _lines: "[LINE1] Short.")
    segments = [
        TranscriptSegment(0, 1, "This sentence is long enough to summarize."),
        TranscriptSegment(1, 2, "So is this one, with a few more words."),
    ]
    document = TranscriptDocument(METADATA, "en", "automatic_captions", segments)

    result = rewrite_transcript(
        document, CONFIG, language=None, refine=True, article=True
    )

    assert result.segments[0].text == " ".join(segment.text for segment in segments)
    assert len(requests) == 1 + ATTEMPTS


def test_a_batch_that_keeps_replying_empty_fails_the_whole_run(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch, lambda _lines: "\n")
    segments = [TranscriptSegment(index, index + 1, f"S{index}.") for index in range(2)]
    document = TranscriptDocument(METADATA, "es", "subtitles", segments)

    with pytest.raises(RewriteError, match=r"lines 1-2 of 2.*no translation"):
        translate(document, "en")
    assert len(requests) == ATTEMPTS


def test_a_failed_request_raises_a_runtime_error_naming_the_host(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    def answer(lines: list[str]) -> str:
        if lines == ["Intro", "Main"]:
            raise openai.APIConnectionError(
                request=httpx2.Request("POST", CONFIG.base_url)
            )
        return prefixed(lines)

    fake_openai(monkeypatch, answer)
    metadata = VideoMetadata(
        video_id=METADATA.video_id,
        url=METADATA.url,
        chapters=[VideoChapter("Intro", 0, 10), VideoChapter("Main", 10, None)],
    )
    segments = [
        TranscriptSegment(0, 5, "Hello there."),
        TranscriptSegment(12, 15, "Now."),
    ]
    document = TranscriptDocument(metadata, "en", "subtitles", segments)

    with pytest.raises(RuntimeError, match=r"gateway\.example chat request failed"):
        rewrite_transcript(document, CONFIG, language="ja", refine=False, article=True)
