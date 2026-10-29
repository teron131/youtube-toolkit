"""Decision requests take TypeSafe's shape at whatever DECISION_URL names, always opt in to images, and read each host's usage and error shapes."""

from email.message import Message
import io
import json
import urllib.error

import pytest

from youtube_toolkit import decisions
from youtube_toolkit.config import DecisionConfig

VERCEL = DecisionConfig(
    "https://ai-gateway.vercel.sh/typesafe/v1/systemone", "secret", "decider"
)
OPENROUTER = DecisionConfig(
    "https://openrouter.ai/api/alpha/decisions", "secret", "decider"
)


@pytest.mark.parametrize(
    ("config", "payload", "tokens", "cost"),
    [
        (
            VERCEL,
            {
                "usage": {"input_tokens": 353},
                "provider_metadata": {"gateway": {"cost": "0.00000353"}},
            },
            353,
            0.00000353,
        ),
        (
            OPENROUTER,
            {"usage": {"input_tokens": 90, "cost": 0.0000009}},
            90,
            0.0000009,
        ),
        (OPENROUTER, {}, None, None),
    ],
)
def test_request_carries_typesafe_shape_and_reads_each_hosts_usage(
    monkeypatch, tmp_path, config, payload, tokens, cost
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg bytes")
    sent = []

    def urlopen(request, timeout):
        sent.append(request)
        answers = {"shown": {"type": "noul", "noul": 0.9}}
        return io.BytesIO(json.dumps({"answers": answers, **payload}).encode())

    monkeypatch.setattr(decisions.urllib.request, "urlopen", urlopen)
    result = decisions.decide(
        config,
        state=["Transcript", "frame"],
        questions={"shown": {"type": "noul", "instructions": "Is it shown?"}},
        assets={"frame": decisions.image_asset(image)},
    )

    assert result == decisions.DecisionResult(
        {"shown": {"type": "noul", "noul": 0.9}}, tokens, cost
    )
    request = sent[0]
    assert request.full_url == config.url
    assert request.get_header("Authorization") == "Bearer secret"
    assert request.get_header("Typesafe-danger-zone-images") == "true"
    body = json.loads(request.data)
    assert body["model"] == "decider"
    assert body["assets"]["frame"] == {
        "type": "image",
        "format": "jpg",
        "data_base64": "anBlZyBieXRlcw==",
    }
    assert body["state"] == ["Transcript", "frame"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"message": "Images are not turned on.", "error_type": "invalid_request"},
            "Images are not turned on.",
        ),
        ({"error": {"message": "Invalid model", "code": 400}}, "Invalid model"),
    ],
)
def test_errors_carry_the_hosts_reason(monkeypatch, body, message) -> None:
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            Message(),
            io.BytesIO(json.dumps(body).encode()),
        )

    monkeypatch.setattr(decisions.urllib.request, "urlopen", urlopen)
    with pytest.raises(decisions.DecisionError, match=f"HTTP 400\\): {message}"):
        decisions.decide(VERCEL, state="s", questions={}, assets={})


def test_decide_each_retries_failures_once_after_the_burst_and_keeps_order(
    monkeypatch,
) -> None:
    calls = []

    def decide(config, *, state, questions, assets):
        calls.append(state)
        # "flaky" fails only its first request; "bad" fails every one.
        if state == "bad" or (state == "flaky" and calls.count("flaky") == 1):
            raise decisions.DecisionError("HTTP 503")
        return decisions.DecisionResult({"q": {"noul": state}}, 1, None)

    monkeypatch.setattr(decisions, "decide", decide)
    rows = decisions.decide_each(
        VERCEL,
        questions={},
        inputs=[decisions.DecisionInput(s, {}) for s in ("ok", "flaky", "bad")],
    )

    assert rows[:2] == [
        decisions.DecisionResult({"q": {"noul": state}}, 1, None)
        for state in ("ok", "flaky")
    ]
    assert isinstance(rows[2], decisions.DecisionError)
    assert sorted(calls) == ["bad", "bad", "flaky", "flaky", "ok"]


def test_decide_each_skips_the_second_round_when_every_request_fails(
    monkeypatch,
) -> None:
    calls = []

    def decide(config, *, state, questions, assets):
        calls.append(state)
        raise decisions.DecisionError("HTTP 401")

    monkeypatch.setattr(decisions, "decide", decide)
    rows = decisions.decide_each(
        VERCEL,
        questions={},
        inputs=[decisions.DecisionInput(s, {}) for s in ("a", "b")],
    )
    assert len(calls) == 2
    assert all(isinstance(row, decisions.DecisionError) for row in rows)
