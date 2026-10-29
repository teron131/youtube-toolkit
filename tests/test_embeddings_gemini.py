"""Native multimodal requests preserve alignment, reject partial vectors, and report only known token usage."""

import base64
from email.message import Message
import http.client
import io
import json
import threading
import urllib.error

import numpy as np
import pytest

from youtube_toolkit.embeddings import gemini as embeddings
from youtube_toolkit.embeddings.results import EmbeddingError, EmbeddingUsage


def vector(index=0):
    values = [0.0] * embeddings.DIMENSIONS
    values[index] = 2.0
    return values


def response(values, usage=None):
    payload = {"embeddings": [{"values": row} for row in values]}
    if usage is not None:
        payload["usageMetadata"] = usage
    return io.BytesIO(json.dumps(payload).encode())


def test_images_and_text_are_separate_deduplicated_inputs_with_aligned_rows(
    monkeypatch, tmp_path
) -> None:
    first = tmp_path / "first.jpg"
    copy = tmp_path / "copy.jpg"
    first.write_bytes(b"\xff\xd8same-image")
    copy.write_bytes(first.read_bytes())
    requests = []

    def respond(request, *, timeout):
        assert request.full_url == embeddings.ENDPOINT
        assert request.get_method() == "POST"
        assert request.get_header("X-goog-api-key") == "private-key"
        assert timeout == 90
        items = json.loads(request.data)["requests"]
        requests.append(items)
        for item in items:
            assert item["model"] == "models/gemini-embedding-2"
            assert item["outputDimensionality"] == 768
            assert len(item["content"]["parts"]) == 1
        assert items[0]["content"]["parts"] == [
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(first.read_bytes()).decode(),
                }
            }
        ]
        assert items[1]["content"]["parts"] == [{"text": "same"}]
        assert items[2]["content"]["parts"] == [{"text": "other"}]
        return response(
            [[3.0, 4.0, *([0.0] * 766)], vector(2), vector(3)],
            {
                "promptTokenCount": 263,
                "promptTokenDetails": [
                    {"modality": "IMAGE", "tokenCount": 258},
                    {"modality": "TEXT", "tokenCount": 5},
                ],
            },
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    client = embeddings.GeminiEmbeddingClient(" private-key ")
    result = client.embed([first, copy, first], ["same", "other", "same"])
    assert result.images.shape == result.texts.shape == (3, 768)
    assert np.allclose(result.images[:, :2], [[0.6, 0.8]] * 3)
    assert np.array_equal(result.texts[0], result.texts[2])
    assert result.texts[0, 2] == 1
    assert result.texts[1, 3] == 1
    assert np.allclose(np.linalg.norm(result.images, axis=1), 1)
    assert result.usage == EmbeddingUsage(258, 5, 263)
    client.embed([first], ["same", "other"])
    assert len(requests) == 2


def test_batches_are_bounded_and_completion_order_does_not_reorder_inputs(
    monkeypatch,
) -> None:
    lock = threading.Lock()
    first_started = threading.Event()
    second_started = threading.Event()
    active = 0
    maximum_active = 0
    sizes = []

    def respond(request, *, timeout):
        nonlocal active, maximum_active
        items = json.loads(request.data)["requests"]
        identifiers = [int(item["content"]["parts"][0]["text"]) for item in items]
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            sizes.append(len(items))
        if identifiers[0] == 0:
            first_started.set()
            assert second_started.wait(timeout=3)
        elif identifiers[0] == 32:
            second_started.set()
            assert first_started.wait(timeout=3)
        with lock:
            active -= 1
        return response(
            [vector(index) for index in identifiers],
            {
                "promptTokenCount": len(items),
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": len(items)}],
            },
        )

    monkeypatch.setattr(embeddings, "BATCH_SIZE", 32)
    monkeypatch.setattr(embeddings, "MAX_WORKERS", 2)
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    texts = [str(index) for index in range(68)] + ["0", "67"]
    result = embeddings.GeminiEmbeddingClient("key").embed([], texts)
    assert sorted(sizes) == [4, 32, 32]
    assert maximum_active == 2
    assert result.images.shape == (0, 768)
    assert result.texts.argmax(axis=1).tolist() == [*range(68), 0, 67]
    assert result.usage == EmbeddingUsage(0, 68, 68)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, EmbeddingUsage(None, None, None)),
        ({"promptTokenCount": 9}, EmbeddingUsage(None, None, 9)),
        (
            {
                "promptTokenCount": 9,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 9}],
            },
            EmbeddingUsage(None, 9, 9),
        ),
        (
            {
                "promptTokenDetails": [
                    {"modality": "IMAGE", "tokenCount": 258},
                    {"modality": "TEXT", "tokenCount": 9},
                ]
            },
            EmbeddingUsage(258, 9, None),
        ),
    ],
)
def test_missing_usage_is_not_fabricated(
    monkeypatch, tmp_path, usage, expected
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"jpeg")
    monkeypatch.setattr(
        embeddings.urllib.request,
        "urlopen",
        lambda _request, **_kwargs: response([vector(), vector()], usage),
    )
    result = embeddings.GeminiEmbeddingClient("key").embed([image], ["caption"])
    assert result.usage == expected


def test_partial_batch_usage_keeps_aggregate_unknown(monkeypatch) -> None:
    def respond(request, *, timeout):
        items = json.loads(request.data)["requests"]
        usage = (
            {
                "promptTokenCount": 32,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 32}],
            }
            if len(items) == 32
            else None
        )
        return response([vector()] * len(items), usage)

    monkeypatch.setattr(embeddings, "BATCH_SIZE", 32)
    monkeypatch.setattr(embeddings, "MAX_WORKERS", 2)
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    result = embeddings.GeminiEmbeddingClient("key").embed(
        [], [str(index) for index in range(33)]
    )
    assert result.usage == EmbeddingUsage(None, None, None)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"embeddings": []},
        {"embeddings": [{"values": [1.0, 0.0]}]},
        {"embeddings": [{"values": [0.0] * 768}]},
        {"embeddings": [{"values": [float("nan"), *([0.0] * 767)]}]},
        {"embeddings": [{"values": [float("inf"), *([0.0] * 767)]}]},
        {"embeddings": [{"values": [True, *([0.0] * 767)]}]},
        {"embeddings": [{"values": ["private-input", *([0.0] * 767)]}]},
        {
            "embeddings": [{"values": vector()}],
            "usageMetadata": {"promptTokenCount": -1},
        },
    ],
)
def test_malformed_responses_fail_without_returning_partial_vectors(
    monkeypatch, payload
) -> None:
    monkeypatch.setattr(
        embeddings.urllib.request,
        "urlopen",
        lambda _request, **_kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    with pytest.raises(EmbeddingError) as caught:
        embeddings.GeminiEmbeddingClient("private-key").embed([], ["private-input"])
    assert "private-key" not in str(caught.value)
    assert "private-input" not in str(caught.value)


def test_one_failed_batch_fails_the_call(monkeypatch) -> None:
    def respond(request, *, timeout):
        items = json.loads(request.data)["requests"]
        return response([vector()] * len(items) if len(items) == 32 else [])

    monkeypatch.setattr(embeddings, "BATCH_SIZE", 32)
    monkeypatch.setattr(embeddings, "MAX_WORKERS", 2)
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    with pytest.raises(EmbeddingError, match="vector count"):
        embeddings.GeminiEmbeddingClient("key").embed(
            [], [str(index) for index in range(33)]
        )


def test_transient_http_failure_retries_and_then_succeeds(monkeypatch) -> None:
    attempts = 0
    delays = []

    def respond(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.HTTPError(
                embeddings.ENDPOINT, 429, "private-error-body", Message(), None
            )
        return response([vector()])

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)
    result = embeddings.GeminiEmbeddingClient("key").embed([], ["caption"])
    assert result.texts.shape == (1, 768)
    assert attempts == 3
    assert delays == [1, 2]


def test_auth_failure_is_not_retried_and_keeps_a_safe_error_message(
    monkeypatch,
) -> None:
    calls = []
    failure = urllib.error.HTTPError(
        embeddings.ENDPOINT, 401, "private-key private-input", Message(), None
    )

    def respond(*_args, **_kwargs):
        calls.append(True)
        raise failure

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    with pytest.raises(EmbeddingError, match="HTTP 401") as caught:
        embeddings.GeminiEmbeddingClient("private-key").embed([], ["private-input"])
    assert len(calls) == 1
    assert caught.value.__cause__ is failure
    assert "private-key" not in str(caught.value)
    assert "private-input" not in str(caught.value)


def test_http_failure_names_google_reason_code_without_message_text(
    monkeypatch,
) -> None:
    body = {
        "error": {
            "code": 400,
            "message": "API key not valid for private-input",
            "status": "INVALID_ARGUMENT",
            "details": [{"@type": "ErrorInfo", "reason": "API_KEY_INVALID"}],
        }
    }

    def respond(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            embeddings.ENDPOINT,
            400,
            "Bad Request",
            Message(),
            io.BytesIO(json.dumps(body).encode()),
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    with pytest.raises(EmbeddingError) as caught:
        embeddings.GeminiEmbeddingClient("key").embed([], ["private-input"])
    assert str(caught.value) == (
        "Gemini embedding request failed (HTTP 400, API_KEY_INVALID)"
    )


def test_response_cut_off_mid_body_is_retried(monkeypatch) -> None:
    attempts = 0

    def respond(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.IncompleteRead(b"partial")
        return response([vector()])

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", lambda _seconds: None)
    result = embeddings.GeminiEmbeddingClient("key").embed([], ["caption"])
    assert result.texts.shape == (1, 768)
    assert attempts == 2


def test_network_failure_is_bounded_and_preserves_cause(monkeypatch) -> None:
    calls = []
    failure = urllib.error.URLError("private connection detail")

    def respond(*_args, **_kwargs):
        calls.append(True)
        raise failure

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", lambda _delay: None)
    with pytest.raises(EmbeddingError, match="connection failed") as caught:
        embeddings.GeminiEmbeddingClient("key").embed([], ["caption"])
    assert len(calls) == 3
    assert caught.value.__cause__ is failure
    assert "private connection detail" not in str(caught.value)


def test_invalid_json_is_not_retried(monkeypatch) -> None:
    calls = []

    def respond(*_args, **_kwargs):
        calls.append(True)
        return io.BytesIO(b"invalid private response")

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", respond)
    with pytest.raises(EmbeddingError, match="not valid JSON"):
        embeddings.GeminiEmbeddingClient("key").embed([], ["caption"])
    assert len(calls) == 1


def test_empty_inputs_make_no_requests(monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("empty inputs must not call Gemini"),
    )
    result = embeddings.GeminiEmbeddingClient("key").embed([], [])
    assert result.images.shape == result.texts.shape == (0, 768)
    assert result.usage == EmbeddingUsage(0, 0, 0)
