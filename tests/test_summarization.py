from langchain_core.messages import HumanMessage, SystemMessage

from youtube_toolkit import ModelConfig
from youtube_toolkit.summarization import CaptionSummary, Chapter, summarize_caption


def test_model_config_normalizes_values() -> None:
    config = ModelConfig(
        model=" summary-model ",
        api_key=" summary-key ",
        base_url=" https://summary.example/v1/ ",
    )

    assert config.model == "summary-model"
    assert config.api_key == "summary-key"
    assert config.base_url == "https://summary.example/v1"


def test_text_summary_formats_chapters() -> None:
    summary = CaptionSummary(
        overview="Main arc.",
        chapters=[
            Chapter(
                title="Setup",
                description="Initial context.",
                start_time="0:00",
                end_time="1:00",
            ),
        ],
    )

    assert summary.model_dump() == {
        "overview": "Main arc.",
        "chapters": [
            {
                "title": "Setup",
                "description": "Initial context.",
                "start_time": "0:00",
                "end_time": "1:00",
            }
        ],
    }
    assert "0:00-1:00 Setup" in summary.to_text()


def test_text_summary_allows_no_chapters() -> None:
    summary = CaptionSummary(
        overview="Short source with no chapter structure.",
        chapters=[],
    )

    assert summary.model_dump() == {
        "overview": "Short source with no chapter structure.",
        "chapters": [],
    }


def test_summarize_caption_uses_langchain_structured_output(monkeypatch) -> None:
    calls = {}

    class FakeStructuredLlm:
        def invoke(self, messages):
            calls["messages"] = messages
            return CaptionSummary(
                overview="A compact summary.",
                chapters=[
                    Chapter(
                        title="Chapter",
                        description="Chapter summary.",
                    )
                ],
            )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def with_structured_output(self, schema):
            calls["schema"] = schema
            return FakeStructuredLlm()

    monkeypatch.setattr("youtube_toolkit.summarization.ChatOpenAI", FakeChatOpenAI)

    summary = summarize_caption(
        "Transcript text.",
        ModelConfig(
            model="summary-model",
            api_key="summary-key",
            base_url="https://summary.example/v1/",
        ),
        title="Title",
    )

    assert summary.overview == "A compact summary."
    assert calls["kwargs"] == {
        "model": "summary-model",
        "api_key": "summary-key",
        "base_url": "https://summary.example/v1",
    }
    assert calls["schema"] is CaptionSummary
    assert isinstance(calls["messages"][0], SystemMessage)
    assert isinstance(calls["messages"][1], HumanMessage)
    assert "Write the summary in English" in calls["messages"][0].content
    assert "Title: Title" in calls["messages"][0].content
    assert calls["messages"][1].content == "Text to summarize:\nTranscript text."
