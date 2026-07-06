"""LanceDB runtime connection, storage-path setup, and table inspection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import lancedb
from lancedb.index import BTree, FTS
from lancedb.table import LanceTable
import pyarrow as pa

STORAGE_DIRNAME = ".cache"
APP_STATE_DIRNAME = "youtube-toolkit"
DATABASE_FILENAME = "youtube.lancedb"
DATABASE_SCHEMA_REPAIR = "Run youtube setup database to create a fresh LanceDB store"
MAX_TABLE_SCAN_ROWS = 100_000
TRANSCRIPT_CHUNK_TEXT_INDEX = "transcript_chunk_text_idx"
SCALAR_INDEX_COLUMNS = {
    "channels": (
        "youtube_channel_id",
        "channel_handle",
    ),
    "videos": (
        "youtube_video_id",
        "channel_id",
        "upload_date",
        "view_count",
        "like_count",
    ),
    "channels_watchlist": ("channel_id",),
    "watchlist_videos": ("video_id",),
    "channels_blacklist": ("channel_id",),
    "video_embeddings": ("video_id",),
    "video_transcript_chunks": (
        "video_id",
        "chunk_index",
    ),
}


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
        self.ensure_base_scalar_indexes()
        self.ensure_optional_table_indexes()

    def __enter__(self) -> LanceConnection:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def ensure_base_tables(self) -> None:
        for table_name, schema in BASE_TABLE_SCHEMAS.items():
            self.ensure_table(table_name, schema)

    def ensure_base_scalar_indexes(self) -> None:
        for table_name in BASE_TABLE_SCHEMAS:
            self.ensure_scalar_indexes(table_name)

    def ensure_optional_table_indexes(self) -> None:
        self.ensure_scalar_indexes("video_embeddings")
        self.ensure_scalar_indexes("video_transcript_chunks")
        self.ensure_transcript_chunk_text_index()

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
        table = self.ensure_table(
            "video_embeddings",
            vector_schema(
                id_field="video_id",
                vector_dimensions=vector_dimensions,
            ),
        )
        self.ensure_scalar_indexes("video_embeddings")
        return table

    def ensure_transcript_chunk_table(
        self,
        vector_dimensions: int,
    ) -> LanceTable:
        table = self.ensure_table(
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
        self.ensure_scalar_indexes("video_transcript_chunks")
        return table

    def ensure_scalar_indexes(
        self,
        table_name: str,
        *,
        refresh: bool = False,
    ) -> None:
        table = self.optional_table(table_name)
        if table is None:
            return
        index_names = {index.name for index in table.list_indices()}
        for column_name in SCALAR_INDEX_COLUMNS.get(table_name, ()):
            index_name = f"{table_name}_{column_name}_idx"
            if not refresh and index_name in index_names:
                continue
            table.create_index(
                column_name,
                config=BTree(),
                name=index_name,
                replace=True,
            )

    def ensure_transcript_chunk_text_index(self) -> None:
        table = self.optional_table("video_transcript_chunks")
        if table is None:
            return
        if any(index.name == TRANSCRIPT_CHUNK_TEXT_INDEX for index in table.list_indices()):
            return
        table.create_index(
            "text",
            config=FTS(base_tokenizer="icu"),
            name=TRANSCRIPT_CHUNK_TEXT_INDEX,
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
        *,
        where_sql: str | None = None,
        limit: int = MAX_TABLE_SCAN_ROWS,
    ) -> list[dict[str, Any]]:
        table = self.optional_table(table_name)
        if table is None:
            return []
        query = table.search()
        if where_sql is not None:
            query = query.where(where_sql)
        return query.limit(limit).to_list()

    def find_row(
        self,
        table_name: str,
        key: str,
        value: Any,
    ) -> dict[str, Any] | None:
        rows = self.rows(
            table_name,
            where_sql=filter_equal(key, value),
            limit=1,
        )
        return rows[0] if rows else None

    def upsert_rows(
        self,
        table_name: str,
        key: str | list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        self.table(table_name).merge_insert(key).when_matched_update_all().when_not_matched_insert_all().execute(rows)
        self.ensure_scalar_indexes(table_name, refresh=True)

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


def filter_equal(
    column_name: str,
    value: Any,
) -> str:
    if value is None:
        return f"{column_name} IS NULL"
    if isinstance(value, bool):
        literal = "TRUE" if value else "FALSE"
    elif isinstance(value, int | float):
        literal = str(value)
    else:
        literal = filter_literal(str(value))
    return f"{column_name} = {literal}"


def filter_in_values(values: list[str]) -> str:
    return ", ".join(filter_literal(value) for value in values)
