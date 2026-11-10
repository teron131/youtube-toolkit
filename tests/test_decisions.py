"""Decision requests through TypeSafe's SDK, and how each host's usage and errors are read."""

import json

import httpx2
import pytest
from typesafe_sdk import Noul, NoulAnswer

from youtube_toolkit import decisions
from youtube_toolkit.config import DecisionConfig

VERCEL = DecisionConfig("https://ai-gateway.vercel.sh/typesafe", "secret", "decider")
OPENROUTER = DecisionConfig("https://openrouter.ai/api", "secret", "decider")
QUESTIONS = {"shown": Noul(instructions="Is it shown?")}
ANSWERS = {"shown": {"type": "noul", "noul": 0.9}}


def serve(monkeypatch, reply):
    """`reply` maps each request's JSON body to the status and JSON body the host answers with; the requests sent are returned."""
    sent = []

    def handle(request):
        sent.append(request)
        status, body = reply(json.loads(request.content))
        return httpx2.Response(status, json=body)

    client = decisions.TypeSafeClient
    monkeypatch.setattr(
        decisions,
        "TypeSafeClient",
        lambda **kwargs: client(**kwargs, transport=httpx2.MockTransport(handle)),
    )
    return sent


@pytest.mark.parametrize(
    ("config", "payload", "tokens", "cost", "seconds"),
    [
        (
            VERCEL,
            {
                "usage": {"input_tokens": 353, "output_tokens": 3},
                "provider_metadata": {
                    "gateway": {
                        "cost": "0.00000353",
                        # A provider that failed before the one that answered.
                        "routing": {
                            "modelAttempts": [
                                {
                                    "providerAttempts": [
                                        {"startTime": 1000, "endTime": 1140},
                                        {"startTime": 1150, "endTime": 1512},
                                    ]
                                }
                            ]
                        },
                    }
                },
            },
            (353, 3),
            0.00000353,
            0.502,
        ),
        (
            OPENROUTER,
            {"usage": {"input_tokens": 90, "cost": 0.0000009}},
            (90, None),
            0.0000009,
            None,
        ),
    ],
)
def test_request_carries_typesafe_shape_and_reads_each_hosts_usage(
    monkeypatch, config, payload, tokens, cost, seconds
) -> None:
    sent = serve(
        monkeypatch,
        lambda _body: (200, {"model": "decider", "answers": ANSWERS, **payload}),
    )
    [result] = decisions.decide_each(
        config,
        questions=QUESTIONS,
        inputs=[
            decisions.DecisionInput(
                ["Transcript", "frame"], {"frame": decisions.image_asset(b"jpeg bytes")}
            )
        ],
    )

    assert result == decisions.DecisionResult(
        {"shown": NoulAnswer(noul=0.9)}, *tokens, cost, seconds
    )
    [request] = sent
    assert str(request.url) == f"{config.base_url}/v1/systemone"
    assert request.headers["Authorization"] == "Bearer secret"
    assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "state": ["Transcript", "frame"],
        "model": "decider",
        "questions": {"shown": {"type": "noul", "instructions": "Is it shown?"}},
        "assets": {
            "frame": {
                "type": "image",
                "format": "jpg",
                "data_base64": "anBlZyBieXRlcw==",
            }
        },
    }


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
    serve(monkeypatch, lambda _body: (400, body))
    [row] = decisions.decide_each(
        VERCEL, questions=QUESTIONS, inputs=[decisions.DecisionInput("s", {})]
    )

    assert isinstance(row, decisions.DecisionError)
    assert "ai-gateway.vercel.sh decision request failed" in str(row)
    assert f"400 {message}" in str(row)


def test_decide_each_retries_failures_once_after_the_burst_and_keeps_order(
    monkeypatch,
) -> None:
    states = []

    def reply(body):
        state = body["state"]
        states.append(state)
        # "flaky" fails only its first request; "bad" fails every one.
        if state == "bad" or (state == "flaky" and states.count("flaky") == 1):
            return 400, {"message": "Rejected"}
        answers = {"shown": {"type": "noul", "noul": len(state) / 10}}
        return 200, {"model": "decider", "answers": answers, "usage": {}}

    serve(monkeypatch, reply)
    rows = decisions.decide_each(
        VERCEL,
        questions=QUESTIONS,
        inputs=[decisions.DecisionInput(s, {}) for s in ("ok", "flaky", "bad")],
    )

    assert rows[:2] == [
        decisions.DecisionResult(
            {"shown": NoulAnswer(noul=len(state) / 10)}, None, None, None, None
        )
        for state in ("ok", "flaky")
    ]
    assert isinstance(rows[2], decisions.DecisionError)
    assert sorted(states) == ["bad", "bad", "flaky", "flaky", "ok"]


def test_decide_each_skips_the_second_round_when_every_request_fails(
    monkeypatch,
) -> None:
    sent = serve(monkeypatch, lambda _body: (401, {"message": "Invalid key"}))
    rows = decisions.decide_each(
        VERCEL,
        questions=QUESTIONS,
        inputs=[decisions.DecisionInput(s, {}) for s in ("a", "b")],
    )

    assert len(sent) == 2
    assert all(isinstance(row, decisions.DecisionError) for row in rows)
