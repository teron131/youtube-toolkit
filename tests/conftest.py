"""Keep saved credentials in each test's temporary directory so checks never read or modify the user's configuration, and keep the local embedding route off unless a test switches it on, whatever extras this environment has."""

import pytest

from youtube_toolkit import config
from youtube_toolkit.embeddings import local as local_embeddings


@pytest.fixture(autouse=True)
def isolate_saved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "user_config_path", lambda *_args, **_kwargs: tmp_path)


@pytest.fixture(autouse=True)
def without_local_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(local_embeddings, "find_spec", lambda _name: None)
