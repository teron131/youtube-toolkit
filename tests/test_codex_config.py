"""Codex sandbox setup tests cover safe TOML preservation and idempotent writable roots."""

from pathlib import Path
import stat

import tomlkit
import pytest

from youtube_toolkit.codex_config import configure_codex_permissions


def test_codex_setup_preserves_config_and_adds_root_once(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    writable_root = tmp_path / "Library" / "Application Support" / "youtube-toolkit"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '# Keep this comment.\nmodel = "gpt-5.6-sol"\n\n[sandbox_workspace_write]\nnetwork_access = true\nwritable_roots = ["/existing/path"]\n',
        encoding="utf-8",
    )
    config_path.chmod(0o640)

    preview = configure_codex_permissions(
        config_path=config_path,
        writable_root=writable_root,
        apply=False,
    )

    assert preview["status"] == "would_configure"
    assert writable_root.exists() is False
    assert config_path.read_text(encoding="utf-8").endswith('writable_roots = ["/existing/path"]\n')

    configured = configure_codex_permissions(
        config_path=config_path,
        writable_root=writable_root,
        apply=True,
    )
    first_write = config_path.read_text(encoding="utf-8")
    configured_document = tomlkit.parse(first_write)

    assert configured["status"] == "configured"
    assert first_write.startswith("# Keep this comment.\n")
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o640
    assert configured_document["sandbox_workspace_write"]["network_access"] is True
    assert list(configured_document["sandbox_workspace_write"]["writable_roots"]) == [
        "/existing/path",
        str(writable_root),
    ]
    assert writable_root.is_dir()

    repeated = configure_codex_permissions(
        config_path=config_path,
        writable_root=writable_root,
        apply=True,
    )

    assert repeated["status"] == "already_configured"
    assert config_path.read_text(encoding="utf-8") == first_write


def test_codex_setup_creates_missing_config_only_when_confirmed(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    writable_root = tmp_path / "youtube-toolkit"

    preview = configure_codex_permissions(
        config_path=config_path,
        writable_root=writable_root,
        apply=False,
    )

    assert preview["status"] == "would_configure"
    assert config_path.exists() is False

    configured = configure_codex_permissions(
        config_path=config_path,
        writable_root=writable_root,
        apply=True,
    )
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))

    assert configured["status"] == "configured"
    assert list(document["sandbox_workspace_write"]["writable_roots"]) == [str(writable_root)]
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert writable_root.is_dir()


def test_codex_setup_recognizes_equivalent_root_without_rewriting(tmp_path: Path) -> None:
    config_path = tmp_path / ".codex" / "config.toml"
    writable_root = tmp_path / "state" / "youtube-toolkit"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        f'[sandbox_workspace_write]\nwritable_roots = ["{writable_root}/"]\n',
        encoding="utf-8",
    )
    original = config_path.read_bytes()

    result = configure_codex_permissions(
        config_path=config_path,
        writable_root=writable_root,
        apply=True,
    )

    assert result["status"] == "already_configured"
    assert config_path.read_bytes() == original
    assert writable_root.is_dir()


def test_codex_setup_reports_config_directory_creation_failure(tmp_path: Path) -> None:
    config_parent = tmp_path / "not-a-directory"
    config_parent.write_text("occupied", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not write Codex config"):
        configure_codex_permissions(
            config_path=config_parent / "config.toml",
            writable_root=tmp_path / "youtube-toolkit",
            apply=True,
        )
