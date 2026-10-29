"""Keep each test off the user's configuration: saved settings live in the test's temporary directory, API keys and model settings from the environment are cleared, and the local embedding route stays off unless a test switches it on, whatever extras this environment has."""

import os

import pytest

from youtube_toolkit import config
from youtube_toolkit.embeddings import local as local_embeddings


@pytest.fixture(autouse=True)
def isolate_saved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "user_config_path", lambda *_args, **_kwargs: tmp_path)
    for name in list(os.environ):
        if name.endswith("_API_KEY") or name.startswith(
            ("EMBEDDING_", "STT_", "DECISION_")
        ):
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def without_local_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(local_embeddings, "find_spec", lambda _name: None)
