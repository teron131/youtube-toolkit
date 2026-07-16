"""Codex integration safely grants the active toolkit storage directory as a writable root."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
from typing import Literal, TypedDict

import tomlkit
from tomlkit.items import Array, Table

from .storage.connection import resolve_storage_dir


class CodexSetupResult(TypedDict):
    """Public result for previewing or applying the Codex sandbox integration."""

    config: str
    writable_root: str
    status: Literal["already_configured", "would_configure", "configured"]


def codex_config_path() -> Path:
    """Resolve Codex config from its explicit home or the standard user directory."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser().resolve() / "config.toml"
    return Path.home() / ".codex" / "config.toml"


def configure_codex_permissions(
    *,
    apply: bool,
    config_path: Path | None = None,
    writable_root: Path | None = None,
) -> CodexSetupResult:
    """Grant the active storage root once while preserving the rest of Codex config."""
    resolved_config_path = (config_path or codex_config_path()).expanduser().resolve()
    resolved_writable_root = (writable_root or resolve_storage_dir()).expanduser().resolve()
    document = read_codex_config(resolved_config_path)
    sandbox_config = sandbox_workspace_write_table(document)
    writable_roots = sandbox_config.get("writable_roots")
    already_configured = any(Path(root).expanduser().resolve() == resolved_writable_root for root in writable_root_values(writable_roots))
    if apply:
        try:
            resolved_writable_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RuntimeError(f"Could not create Codex writable root {resolved_writable_root}: {error}") from error
    if already_configured:
        status = "already_configured"
    elif not apply:
        status = "would_configure"
    else:
        if writable_roots is None:
            sandbox_config["writable_roots"] = [str(resolved_writable_root)]
        else:
            writable_roots.append(str(resolved_writable_root))
        write_codex_config(resolved_config_path, tomlkit.dumps(document))
        status = "configured"
    return {
        "config": str(resolved_config_path),
        "writable_root": str(resolved_writable_root),
        "status": status,
    }


def read_codex_config(config_path: Path) -> tomlkit.TOMLDocument:
    """Parse existing Codex TOML without losing its comments or formatting."""
    if not config_path.exists():
        return tomlkit.document()
    try:
        return tomlkit.parse(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read Codex config {config_path}: {error}") from error


def sandbox_workspace_write_table(document: tomlkit.TOMLDocument) -> Table:
    """Return or create the table that owns workspace-write sandbox settings."""
    existing = document.get("sandbox_workspace_write")
    if existing is None:
        created = tomlkit.table()
        document["sandbox_workspace_write"] = created
        return created
    if not isinstance(existing, Table):
        raise ValueError("Codex sandbox_workspace_write config must be a TOML table")
    return existing


def writable_root_values(value: object) -> list[str]:
    """Validate the existing writable-root contract before modifying user config."""
    if value is None:
        return []
    if not isinstance(value, Array):
        raise ValueError("Codex sandbox_workspace_write.writable_roots must be an array")
    roots = value.unwrap()
    if not all(isinstance(root, str) for root in roots):
        raise ValueError("Codex sandbox_workspace_write.writable_roots must contain only paths")
    return roots


def write_codex_config(config_path: Path, content: str) -> None:
    """Atomically replace Codex config while retaining its existing file mode."""
    temporary_path: Path | None = None
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = stat.S_IMODE(config_path.stat().st_mode) if config_path.exists() else 0o600
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=config_path.parent,
            encoding="utf-8",
            prefix=f".{config_path.name}.",
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        temporary_path.chmod(existing_mode)
        temporary_path.replace(config_path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not write Codex config {config_path}: {error}") from error
