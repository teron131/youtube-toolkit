"""Embedding transport, environment configuration, and transcript chunking contracts."""

import urllib.error

from youtube_toolkit.config import (
    DEFAULT_EMBEDDING_CONCURRENCY,
    MAX_EMBEDDING_CONCURRENCY,
    EmbeddingConfig,
    embedding_concurrency,
)
from youtube_toolkit.embeddings import embed_text, post_embedding_json
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
    calls = {}

    def fake_post_embedding_json(url, config, payload):
        calls["url"] = url
        calls["config"] = config
        calls["payload"] = payload
        return {"data": [{"embedding": [0.25, 0.75]}]}

    config = EmbeddingConfig(
        model="text-embedding-test",
        api_key="embedding-key",
        base_url="https://embedding.example/v1/",
        output_dimensionality=512,
    )
    monkeypatch.setattr(
        "youtube_toolkit.embeddings.post_embedding_json", fake_post_embedding_json
    )

    assert embed_text("hello", config=config) == [0.25, 0.75]
    assert calls == {
        "url": "https://embedding.example/v1/embeddings",
        "config": config,
        "payload": {
            "model": "text-embedding-test",
            "input": "hello",
            "dimensions": 512,
        },
    }


def test_embedding_request_retries_transient_network_failures(monkeypatch) -> None:
    """Embedding requests should isolate transient failures behind three total attempts."""
    attempts = 0
    delays: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"data": [{"embedding": [1.0, 0.0]}]}'

    def fake_urlopen(_request, *, timeout: int):
        nonlocal attempts
        assert timeout == 60
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()

    config = EmbeddingConfig(
        model="text-embedding-test",
        api_key="embedding-key",
        base_url="https://embedding.example/v1/",
        output_dimensionality=2,
    )
    monkeypatch.setattr(
        "youtube_toolkit.embeddings.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("youtube_toolkit.execution.time.sleep", delays.append)

    assert post_embedding_json(
        "https://embedding.example/v1/embeddings",
        config,
        {
            "model": config.model,
            "input": "hello",
            "dimensions": 2,
        },
    ) == {"data": [{"embedding": [1.0, 0.0]}]}
    assert attempts == 3
    assert delays == [0.25, 0.5]


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
