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
from youtube_toolkit.embeddings import EmbeddingError, embed_text
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


def test_embed_text_uses_embedding_config(monkeypatch) -> None:
    """The real SDK sends configured credentials, endpoint, dimensions, and float encoding."""
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
    monkeypatch.setattr("youtube_toolkit.embeddings.OpenAI", create_client)

    assert embed_text("hello", config=config) == [0.25, 0.75]
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

    monkeypatch.setattr("youtube_toolkit.embeddings.OpenAI", create_client)
    monkeypatch.setattr("openai._base_client.time.sleep", lambda _delay: None)
    config = EmbeddingConfig(
        "text-embedding-test", "embedding-key", "https://embedding.example/v1", 2
    )
    assert embed_text("hello", config=config) == [1.0, 0.0]
    assert attempts == 3


@pytest.mark.parametrize(
    ("status", "payload"),
    [(401, {"error": {"message": "Invalid key"}}), (200, {"data": []})],
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

    monkeypatch.setattr("youtube_toolkit.embeddings.OpenAI", create_client)
    config = EmbeddingConfig("test", "embedding-key", "https://embedding.example/v1")
    with pytest.raises(EmbeddingError, match=r"Invalid key|did not include values"):
        embed_text("hello", config=config)


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
