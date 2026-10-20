"""Keep saved credentials in each test's temporary directory so checks never read or modify the user's configuration."""

import pytest

from youtube_toolkit import config


@pytest.fixture(autouse=True)
def isolate_saved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "user_config_path", lambda *_args, **_kwargs: tmp_path)
