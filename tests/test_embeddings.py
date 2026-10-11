"""Embedding transport, environment configuration, and transcript chunking contracts."""

import json

import httpx2
from openai import OpenAI
import pytest

from youtube_toolkit.config import (
    DEFAULT_EMBEDDING_CONCURRENCY,
    MAX_EMBEDDING_CONCURRENCY,
    EmbeddingConfig,
    embedding_concurrency,
)
from youtube_toolkit.search.embeddings import EmbeddingClient, EmbeddingError
from youtube_toolkit.search.semantic import split_transcript_chunks


def test_split_transcript_chunks_uses_overlap() -> None:
    chunks = split_transcript_chunks(
        "abcdefghijklmnopqrstuvwxyz",
        chunk_chars=10,
        overlap_chars=3,
    )

    assert chunks == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_embedding_client_uses_config_and_one_sdk_client(monkeypatch) -> None:
    """The real SDK sends configured credentials, endpoint, dimensions, and float encoding through one reused client."""
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(200, json={"data": [{"embedding": [0.25, 0.75]}]})

    clients = []

    def create_client(**kwargs):
        client = OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))
        )
        clients.append(client)
        return client

    config = EmbeddingConfig(
        model="text-embedding-test",
        api_key="embedding-key",
        base_url="https://embedding.example/v1/",
        output_dimensionality=512,
    )
    monkeypatch.setattr("youtube_toolkit.search.embeddings.OpenAI", create_client)

    with EmbeddingClient(config) as client:
        assert client.embed("hello") == [0.25, 0.75]
        assert client.embed("again") == [0.25, 0.75]
    assert len(clients) == 1
    assert len(requests) == 2
    assert str(requests[0].url) == "https://embedding.example/v1/embeddings"
    assert requests[0].headers["Authorization"] == "Bearer embedding-key"
    assert json.loads(requests[0].content) == {
        "model": "text-embedding-test",
        "input": "hello",
        "dimensions": 512,
        "encoding_format": "float",
    }
    assert clients[0].timeout == 60
    assert clients[0].max_retries == 2
    assert clients[0].is_closed()


def test_embedding_request_retries_transient_network_failures(monkeypatch) -> None:
    """SDK retries retain three total attempts and close the client after recovery."""
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx2.ConnectError("temporary outage", request=request)
        return httpx2.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

    def create_client(**kwargs):
        return OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))
        )

    monkeypatch.setattr("youtube_toolkit.search.embeddings.OpenAI", create_client)
    monkeypatch.setattr("openai._base_client.time.sleep", lambda _delay: None)
    config = EmbeddingConfig(
        "text-embedding-test", "embedding-key", "https://embedding.example/v1", 2
    )
    with EmbeddingClient(config) as client:
        assert client.embed("hello") == [1.0, 0.0]
    assert attempts == 3


def test_embedding_batches_share_text_without_losing_results_or_failures(
    monkeypatch,
) -> None:
    """Lists share one request per distinct text, keep partial failures, and fetch fresh embeddings on later calls."""
    requests = []

    def respond(request):
        text = json.loads(request.content)["input"]
        requests.append(text)
        if text == "unavailable":
            return httpx2.Response(400, json={"error": {"message": "Rejected text"}})
        if text == "empty":
            return httpx2.Response(200, json={"data": [{"embedding": []}]})
        return httpx2.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

    def create_client(**kwargs):
        return OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))
        )

    monkeypatch.setattr("youtube_toolkit.search.embeddings.OpenAI", create_client)
    config = EmbeddingConfig("test", "embedding-key", "https://embedding.example/v1")
    with EmbeddingClient(config) as client:
        assert client.embed([]) == ({}, {})
        assert requests == []
        vectors, failures = client.embed(
            ["shared title", "shared title", "unavailable", "unavailable", "empty"]
        )
        assert vectors == {"shared title": [1.0, 0.0]}
        assert set(failures) == {"unavailable", "empty"}
        assert "Rejected text" in failures["unavailable"]
        assert failures["empty"] == "Embedding request returned an empty vector"
        assert sorted(requests) == ["empty", "shared title", "unavailable"]
        assert client.embed(["shared title"]) == (
            {"shared title": [1.0, 0.0]},
            {},
        )
    assert requests.count("shared title") == 2


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (401, {"error": {"message": "Invalid key"}}),
        (200, {"data": []}),
        (200, {"data": [{"embedding": []}]}),
    ],
)
def test_embedding_failures_remain_search_errors(monkeypatch, status, payload) -> None:
    """API rejection and missing vectors stay within the caller's existing error contract."""

    def create_client(**kwargs):
        return OpenAI(
            **kwargs,
            http_client=httpx2.Client(
                transport=httpx2.MockTransport(
                    lambda _request: httpx2.Response(status, json=payload)
                )
            ),
        )

    monkeypatch.setattr("youtube_toolkit.search.embeddings.OpenAI", create_client)
    config = EmbeddingConfig("test", "embedding-key", "https://embedding.example/v1")
    with (
        EmbeddingClient(config) as client,
        pytest.raises(
            EmbeddingError, match=r"Invalid key|did not include values|empty vector"
        ),
    ):
        client.embed("hello")


def test_embedding_config_normalizes_values() -> None:
    config = EmbeddingConfig(
        model=" embedding-model ",
        api_key=" embedding-key ",
        base_url=" https://embedding.example/v1/ ",
    )

    assert config.model == "embedding-model"
    assert config.api_key == "embedding-key"
    assert config.base_url == "https://embedding.example/v1"


def test_embedding_concurrency_defaults_and_caps(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_CONCURRENCY", raising=False)
    assert embedding_concurrency(50) == DEFAULT_EMBEDDING_CONCURRENCY
    assert embedding_concurrency(3) == 3

    monkeypatch.setenv("EMBEDDING_CONCURRENCY", "999")
    assert embedding_concurrency(80) == MAX_EMBEDDING_CONCURRENCY
