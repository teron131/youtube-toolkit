"""Summary workflow tests."""

import hashlib
import json

from youtube_toolkit.config import LLMConfig
from youtube_toolkit.schemas import TranscriptDocument, TranscriptSegment, VideoMetadata
from youtube_toolkit.summary import Outline, summarize_transcript

CONFIG = LLMConfig(
    base_url="https://gateway.example/v1", api_key="key", model="example/model"
)
DOCUMENT = TranscriptDocument(
    VideoMetadata(video_id="JpSHyEIZ_bo", url="https://youtu.be/JpSHyEIZ_bo"),
    "en",
    "subtitles",
    [
        TranscriptSegment(0, 1, "GPT Soul costs half as much."),
        TranscriptSegment(1, 2, "It is fast."),
        TranscriptSegment(2, 3, "Subscribe now."),
    ],
)
SENTENCES = [segment.text for segment in DOCUMENT.segments]
DIGESTS = [hashlib.sha256(text.encode()).hexdigest()[:6] for text in SENTENCES]


def fake_openai(monkeypatch) -> list[tuple[str, str, object]]:
    """Records each request's (system, user, response_format) in order, with None as the response_format of a plain text request."""
    requests = []

    def response(**message):
        message = type("Message", (), message)
        return type(
            "Response", (), {"choices": [type("Choice", (), {"message": message})]}
        )

    class Completions:
        async def create(self, **kwargs):
            system, user = (message["content"] for message in kwargs["messages"])
            requests.append((system, user, None))
            gist = system.startswith("Distill")
            return response(
                content="Soul undercuts its rivals." if gist else "Summary."
            )

        async def parse(self, **kwargs):
            system, user = (message["content"] for message in kwargs["messages"])
            requests.append((system, user, kwargs["response_format"]))
            # An exact reference, a miscounted number that its hash relocates, and a stale hash that is dropped.
            references = [f"1#{DIGESTS[0]}", f"9#{DIGESTS[2]}", "2#000000"]
            sound_bites = [
                {"first": reference, "last": reference} for reference in references
            ]
            return response(
                parsed=kwargs["response_format"].model_validate(
                    {"parts": [{"title": "Soul's price", "sound_bites": sound_bites}]}
                )
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})

    monkeypatch.setattr("youtube_toolkit.llm.AsyncOpenAI", Client)
    return requests


def test_outline_copies_the_sound_bites_its_references_find_after_the_gist(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = fake_openai(monkeypatch)

    assert summarize_transcript(DOCUMENT, CONFIG, language="zh-TW") == "Summary."

    (gist_system, gist_user, _), (_, outline_user, schema), (system, user, _) = requests
    assert gist_system.startswith("Distill")
    assert "GPT Soul costs half as much." in gist_user
    assert schema is Outline
    assert "<gist>\nSoul undercuts its rivals.\n</gist>" in outline_user
    assert f"[3#{DIGESTS[2]}] Subscribe now." in outline_user
    assert "zh-TW" in system
    assert "<gist>\nSoul undercuts its rivals.\n</gist>" in user
    assert (
        "<outline>\n## Soul's price\nGPT Soul costs half as much.\nSubscribe now.\n</outline>"
        in user
    )
    assert "GPT Soul costs half as much." in user
    (archive,) = (tmp_path / ".cache/youtube-toolkit/summaries").iterdir()
    payload = json.loads(archive.read_text())["payload"]
    assert list(payload) == ["gist", "outline", "summary"]
