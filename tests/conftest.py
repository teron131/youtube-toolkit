"""Keeps every test off the user's saved settings, environment keys, and installed local embedding models."""

import os

import pytest

from youtube_toolkit import config
from youtube_toolkit.embeddings import local as local_embeddings


@pytest.fixture(autouse=True)
def isolate_saved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "user_config_path", lambda *_args, **_kwargs: tmp_path)
    for name in list(os.environ):
        if name.endswith("_API_KEY") or name.startswith(
            ("EMBEDDING_", "STT_", "DECISION_", "LLM_")
        ):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def without_local_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(local_embeddings, "find_spec", lambda _name: None)
