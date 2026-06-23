"""SQLite connection and schema setup for repo-local YouTube toolkit state."""

from __future__ import annotations

import os
from pathlib import Path

import turso

STORAGE_DIRNAME = ".cache"
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

CREATE TABLE IF NOT EXISTS channels_blacklist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (channel_id) REFERENCES channels (id)
);

"""


def get_connection() -> turso.Connection:
    storage_dir = resolve_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    connection = turso.connect(str(storage_dir / DATABASE_FILENAME))
    connection.row_factory = turso.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_SQL)
    return connection


def resolve_storage_dir() -> Path:
    for environment_variable in (
        "YOUTUBE_TOOLKIT_STORAGE_HOME",
        "YOUTUBE_TOOLKIT_STATE_HOME",
        "YOUTUBE_MEMORY_HOME",
    ):
        configured_home = os.environ.get(environment_variable)
        if configured_home:
            return Path(configured_home).expanduser().resolve()

    current_repo_root = find_repo_root(Path.cwd())
    if current_repo_root is not None:
        return current_repo_root / STORAGE_DIRNAME

    package_repo_root = find_repo_root(Path(__file__))
    if package_repo_root is not None:
        return package_repo_root / STORAGE_DIRNAME
    return Path(__file__).resolve().parents[2] / STORAGE_DIRNAME


def find_repo_root(start_dir: Path) -> Path | None:
    current_dir = start_dir.resolve()
    for candidate_dir in (current_dir, *current_dir.parents):
        if (candidate_dir / ".git").exists():
            return candidate_dir
    return None
