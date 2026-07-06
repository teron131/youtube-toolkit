"""LanceDB runtime connection, storage-path setup, and table inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import lancedb
from lancedb.table import LanceTable
import pyarrow as pa

STORAGE_DIRNAME = ".cache"
APP_STATE_DIRNAME = "youtube-toolkit"
DATABASE_FILENAME = "youtube.lancedb"
DATABASE_SCHEMA_REPAIR = "Run youtube setup database to create a fresh LanceDB store"
MAX_TABLE_SCAN_ROWS = 100_000


class DatabaseSchemaHealthResult(TypedDict):
    path: str
    ok: bool
    status: Literal["ready", "missing", "invalid", "unreadable"]
    detail: str
    repair: NotRequired[str]


BASE_TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "channels": pa.schema(
        [
            pa.field("youtube_channel_id", pa.string()),
            pa.field("channel_handle", pa.string()),
        ]
    ),
    "videos": pa.schema(
        [
            pa.field("youtube_video_id", pa.string()),
            pa.field("channel_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("upload_date", pa.string()),
            pa.field("duration", pa.float64()),
            pa.field("view_count", pa.int64()),
            pa.field("like_count", pa.int64()),
            pa.field("transcript", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("metadata_fetched_at", pa.string()),
            pa.field("transcript_fetched_at", pa.string()),
        ]
    ),
    "channels_watchlist": pa.schema([pa.field("channel_id", pa.string())]),
    "watchlist_videos": pa.schema(
        [
            pa.field("video_id", pa.string()),
            pa.field("discovered_at", pa.string()),
        ]
    ),
    "channels_blacklist": pa.schema([pa.field("channel_id", pa.string())]),
}


def vector_schema(
    *,
    id_field: str,
    vector_dimensions: int,
    extra_fields: list[pa.Field] | None = None,
) -> pa.Schema:
    return pa.schema(
        [
            pa.field(id_field, pa.string()),
            *(extra_fields or []),
            pa.field("embedding_content_hash", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), vector_dimensions)),
        ]
    )


class LanceConnection:
    """Small storage facade so domain modules do not depend on LanceDB query objects."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path
        self.database = lancedb.connect(path)
        self.ensure_base_tables()

    def __enter__(self) -> LanceConnection:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def ensure_base_tables(self) -> None:
        for table_name, schema in BASE_TABLE_SCHEMAS.items():
            self.ensure_table(table_name, schema)

    def ensure_table(
        self,
        table_name: str,
        schema: pa.Schema,
    ) -> LanceTable:
        if table_name in self.table_names():
            return self.database.open_table(table_name)
        return self.database.create_table(table_name, schema=schema)

    def ensure_video_embedding_table(
        self,
        vector_dimensions: int,
    ) -> LanceTable:
        return self.ensure_table(
            "video_embeddings",
            vector_schema(
                id_field="video_id",
                vector_dimensions=vector_dimensions,
            ),
        )

    def ensure_transcript_chunk_table(
        self,
        vector_dimensions: int,
    ) -> LanceTable:
        return self.ensure_table(
            "video_transcript_chunks",
            vector_schema(
                id_field="video_id",
                vector_dimensions=vector_dimensions,
                extra_fields=[
                    pa.field("chunk_index", pa.int64()),
                    pa.field("text", pa.string()),
                ],
            ),
        )

    def table_names(self) -> set[str]:
        return set(self.database.list_tables().tables)

    def table(
        self,
        table_name: str,
    ) -> LanceTable:
        return self.database.open_table(table_name)

    def optional_table(
        self,
        table_name: str,
    ) -> LanceTable | None:
        if table_name not in self.table_names():
            return None
        return self.table(table_name)

    def rows(
        self,
        table_name: str,
    ) -> list[dict[str, Any]]:
        table = self.optional_table(table_name)
        if table is None:
            return []
        return table.search().limit(MAX_TABLE_SCAN_ROWS).to_list()

    def find_row(
        self,
        table_name: str,
        key: str,
        value: Any,
    ) -> dict[str, Any] | None:
        return next((row for row in self.rows(table_name) if row[key] == value), None)

    def upsert_rows(
        self,
        table_name: str,
        key: str | list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        self.table(table_name).merge_insert(key).when_matched_update_all().when_not_matched_insert_all().execute(rows)

    def delete_where(
        self,
        table_name: str,
        where_sql: str,
    ) -> None:
        table = self.optional_table(table_name)
        if table is None:
            return
        table.delete(where_sql)


def get_connection() -> LanceConnection:
    """Open the toolkit LanceDB store and ensure base tables exist."""
    storage_dir = resolve_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    return LanceConnection(storage_dir / DATABASE_FILENAME)


def resolve_storage_dir() -> Path:
    """Resolve the directory that owns the toolkit LanceDB store."""
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


def database_schema_health(database_path: Path | None = None) -> DatabaseSchemaHealthResult:
    path = database_path or resolve_storage_dir() / DATABASE_FILENAME
    return inspect_database_schema(path)


def inspect_database_schema(database_path: Path) -> DatabaseSchemaHealthResult:
    """Inspect the persisted LanceDB table shape without creating or changing the store."""
    if not database_path.exists():
        return {
            "path": str(database_path),
            "ok": False,
            "status": "missing",
            "detail": "LanceDB store is missing",
            "repair": "Run youtube setup database to create a fresh LanceDB store",
        }
    try:
        database = lancedb.connect(database_path)
        table_names = set(database.list_tables().tables)
    except Exception as error:
        return {
            "path": str(database_path),
            "ok": False,
            "status": "unreadable",
            "detail": f"LanceDB store could not be inspected: {error}",
            "repair": DATABASE_SCHEMA_REPAIR,
        }

    missing_tables = sorted(set(BASE_TABLE_SCHEMAS) - table_names)
    if missing_tables:
        return {
            "path": str(database_path),
            "ok": False,
            "status": "invalid",
            "detail": f"missing tables: {', '.join(missing_tables)}",
            "repair": DATABASE_SCHEMA_REPAIR,
        }
    return {
        "path": str(database_path),
        "ok": True,
        "status": "ready",
        "detail": "LanceDB store has expected base tables",
    }


def filter_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def filter_in_values(values: list[str]) -> str:
    return ", ".join(filter_literal(value) for value in values)
