"""Saved app configuration must preserve credentials, honor environment overrides, and never alter the shell environment."""

import os
from pathlib import Path

import pytest

from youtube_toolkit import config
from youtube_toolkit.stt.cloud import cloud_stt_status


def test_saved_settings_are_used_without_exporting_them(monkeypatch) -> None:
    """Both model routes consume saved settings while environment values retain precedence."""
    settings = {
        "EMBEDDING_API_KEY": "saved-embedding-key",
        "EMBEDDING_MODEL": "saved-model",
        "EMBEDDING_DIMENSIONS": "256",
        "STT_PROVIDER": "gemini",
        "GEMINI_API_KEY": "saved-speech-key",
    }
    for name in settings:
        monkeypatch.delenv(name, raising=False)
    before = dict(os.environ)

    assert config.config_path().name == ".env"
    assert config.read_config() == {}
    assert not config.config_path().exists()
    config.save_config(settings)
    embedding = config.embedding_config_from_env()
    assert embedding.api_key == "saved-embedding-key"
    assert embedding.model == "saved-model"
    assert embedding.output_dimensionality == 256
    assert cloud_stt_status()["provider"] == "Gemini"
    assert cloud_stt_status()["key_set"] is True
    assert dict(os.environ) == before

    monkeypatch.setenv("EMBEDDING_API_KEY", "environment-key")
    assert config.config_value("EMBEDDING_API_KEY") == "environment-key"
    monkeypatch.setenv("EMBEDDING_API_KEY", "  ")
    assert config.config_value("EMBEDDING_API_KEY") == "saved-embedding-key"


def test_saved_keys_have_private_permissions_and_atomic_replacement(
    monkeypatch,
) -> None:
    """A failed replacement leaves the old config intact and removes the temporary secret file."""
    path = config.save_config({"EMBEDDING_API_KEY": "old-key"})
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()

    def reject_replace(_source: Path, _target: Path) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", reject_replace)
    with pytest.raises(OSError, match="replacement failed"):
        config.save_config({"EMBEDDING_API_KEY": "new-key"})
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".config-*"))


@pytest.mark.parametrize(
    "content",
    [
        b"EMBEDDING_API_KEY='private-key",
        b"EMBEDDING_MODEL=valid\nEMBEDDING_API_KEY='private-key' extra",
        b"INVALID-KEY=private-key",
        b"\xffprivate-key",
    ],
)
def test_invalid_config_reports_the_file_without_its_contents(content: bytes) -> None:
    config.config_path().write_bytes(content)
    with pytest.raises(ValueError, match=r"configuration|Configuration") as error:
        config.read_config()
    assert str(config.config_path()) in str(error.value)
    assert "private-key" not in str(error.value)


def test_dotenv_comments_quotes_and_exports_are_read_without_expansion(
    monkeypatch,
) -> None:
    """Standard dotenv syntax is accepted while variable-like text in a key remains literal."""
    monkeypatch.setenv("EXAMPLE_SECRET", "must-not-be-expanded")
    config.config_path().write_text(
        "# YouTube Toolkit settings\n"
        "export EMBEDDING_API_KEY='literal-${EXAMPLE_SECRET}'\n"
        "EMBEDDING_MODEL=example-model # optional model\n"
        'STT_LOCAL_MODEL="example/model with spaces"\n'
        "EMPTY=\n"
        "UNASSIGNED\n",
        encoding="utf-8",
    )
    before = dict(os.environ)
    assert config.read_config() == {
        "EMBEDDING_API_KEY": "literal-${EXAMPLE_SECRET}",
        "EMBEDDING_MODEL": "example-model",
        "STT_LOCAL_MODEL": "example/model with spaces",
        "EMPTY": "",
    }
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "value",
    [
        "plain-key",
        "key with 'quotes' and \\\\backslashes\\",
        'key#hash=${NAME}="quoted"\nnext line',
    ],
)
def test_dotenv_saved_values_round_trip_without_changing_secrets(value: str) -> None:
    path = config.save_config({"EMBEDDING_API_KEY": value})
    assert path.read_text(encoding="utf-8").startswith("EMBEDDING_API_KEY='")
    assert config.read_config() == {"EMBEDDING_API_KEY": value}
