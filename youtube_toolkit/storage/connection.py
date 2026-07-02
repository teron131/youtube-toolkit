"""SQLite connection and schema setup for YouTube toolkit state."""

from __future__ import annotations

import os
from pathlib import Path

import turso

STORAGE_DIRNAME = ".cache"
APP_STATE_DIRNAME = "youtube-toolkit"
DATABASE_FILENAME = "youtube.sqlite3"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS channels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    youtube_channel_id TEXT,
    channel_handle TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_channels_youtube_channel_id
ON channels (youtube_channel_id)
WHERE youtube_channel_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_channels_channel_handle
ON channels (channel_handle)
WHERE channel_handle IS NOT NULL;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    youtube_video_id TEXT NOT NULL UNIQUE,
    channel_id INTEGER,
    title TEXT,
    upload_date TEXT,
    view_count INTEGER,
    transcript TEXT,
    summary TEXT,
    embedding_content_hash TEXT,
    embedding F32_BLOB,
    FOREIGN KEY (channel_id) REFERENCES channels (id)
);

CREATE INDEX IF NOT EXISTS idx_videos_channel_id
ON videos (channel_id);

CREATE TABLE IF NOT EXISTS channels_watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (channel_id) REFERENCES channels (id)
);

CREATE TABLE IF NOT EXISTS watchlist_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL UNIQUE,
    discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos (id)
);

CREATE TABLE IF NOT EXISTS channels_blacklist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (channel_id) REFERENCES channels (id)
);

"""


def get_connection() -> turso.Connection:
    """Open the toolkit SQLite database and ensure the schema exists."""
    storage_dir = resolve_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    connection = turso.connect(str(storage_dir / DATABASE_FILENAME))
    connection.row_factory = turso.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_SQL)
    return connection


def resolve_storage_dir() -> Path:
    """Resolve the directory that owns the toolkit SQLite database."""
    configured_storage_home = os.environ.get("YOUTUBE_TOOLKIT_STORAGE_HOME")
    if configured_storage_home:
        return Path(configured_storage_home).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / ".git").exists() or (source_root / "pyproject.toml").exists():
        return source_root / STORAGE_DIRNAME
    return _resolve_user_state_dir()


def _resolve_user_state_dir() -> Path:
    """Resolve the per-user app state directory for installed CLI usage."""
    return Path.home() / "Library" / "Application Support" / APP_STATE_DIRNAME
