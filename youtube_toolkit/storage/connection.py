"""Turso runtime connection, storage-path setup, and schema inspection."""

from __future__ import annotations

from contextlib import closing
from functools import cache
from importlib import resources
import os
from pathlib import Path
import sqlite3
from typing import Literal, NotRequired, TypedDict

import turso

STORAGE_DIRNAME = ".cache"
APP_STATE_DIRNAME = "youtube-toolkit"
DATABASE_FILENAME = "youtube.sqlite3"
SCHEMA_RESOURCE = "schema.sql"
DATABASE_SCHEMA_REPAIR = "Run youtube setup database to archive the current DB and create a fresh schema"


class DatabaseSchemaHealthResult(TypedDict):
    path: str
    ok: bool
    status: Literal["ready", "missing", "invalid", "unreadable"]
    detail: str
    repair: NotRequired[str]


def get_connection() -> turso.Connection:
    """Open the toolkit Turso database and ensure the schema exists."""
    storage_dir = resolve_storage_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    connection = turso.connect(str(storage_dir / DATABASE_FILENAME))
    connection.row_factory = turso.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(schema_sql())
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


def database_schema_health(database_path: Path | None = None) -> DatabaseSchemaHealthResult:
    path = database_path or resolve_storage_dir() / DATABASE_FILENAME
    return inspect_database_schema(path)


def inspect_database_schema(database_path: Path) -> DatabaseSchemaHealthResult:
    """Inspect the persisted DB shape without creating or changing the file."""
    if not database_path.exists():
        return {
            "path": str(database_path),
            "ok": False,
            "status": "missing",
            "detail": "Database file is missing",
            "repair": "Run youtube setup database to create a fresh schema",
        }
    try:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
            issues = database_schema_issues(connection)
    except sqlite3.Error as error:
        return {
            "path": str(database_path),
            "ok": False,
            "status": "unreadable",
            "detail": f"Database could not be inspected: {error}",
            "repair": DATABASE_SCHEMA_REPAIR,
        }
    if issues:
        return {
            "path": str(database_path),
            "ok": False,
            "status": "invalid",
            "detail": "; ".join(issues),
            "repair": DATABASE_SCHEMA_REPAIR,
        }
    return {
        "path": str(database_path),
        "ok": True,
        "status": "ready",
        "detail": "Schema has expected tables, columns, indexes, and foreign keys",
    }


def database_schema_issues(connection: sqlite3.Connection) -> list[str]:
    issues: list[str] = []
    expected_tables, expected_indexes, expected_foreign_keys = expected_schema_shape()
    existing_tables = schema_table_names(connection)
    unexpected_tables = sorted(existing_tables - set(expected_tables))
    if unexpected_tables:
        issues.append(f"unexpected tables: {', '.join(unexpected_tables)}")
    for table_name in sorted(expected_tables):
        expected_columns = expected_tables[table_name]
        if table_name not in existing_tables:
            issues.append(f"{table_name} table is missing")
            continue
        issues.extend(
            table_column_issues(
                connection,
                table_name,
                expected_columns,
            )
        )
        issues.extend(
            table_index_issues(
                connection,
                table_name,
                expected_indexes.get(table_name, set()),
            )
        )
        issues.extend(
            table_foreign_key_issues(
                connection,
                table_name,
                expected_foreign_keys.get(table_name, set()),
            )
        )

    foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_violations:
        issues.append(f"foreign key check failed for {len(foreign_key_violations)} row(s)")
    return issues


@cache
def schema_sql() -> str:
    """Load the packaged SQL schema used to create and validate the DB."""
    return resources.files(__package__).joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")


@cache
def expected_schema_shape() -> tuple[
    dict[str, dict[str, tuple[str, int]]],
    dict[str, set[str]],
    dict[str, set[tuple[str, str, str]]],
]:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema_sql())
        table_names = schema_table_names(connection)
        columns = {
            table_name: {str(row[1]): (str(row[2]).upper(), int(row[5])) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()} for table_name in table_names
        }
        indexes = {
            table_name: {str(row[1]) for row in connection.execute(f"PRAGMA index_list({table_name})").fetchall() if not str(row[1]).startswith("sqlite_autoindex_")}
            for table_name in table_names
        }
        foreign_keys = {
            table_name: {(str(row[3]), str(row[2]), str(row[4])) for row in connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()} for table_name in table_names
        }
        return columns, indexes, foreign_keys


def schema_table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            """,
        ).fetchall()
    }


def table_column_issues(
    connection: sqlite3.Connection,
    table_name: str,
    expected_columns: dict[str, tuple[str, int]],
) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    actual = {str(row[1]): (str(row[2]).upper(), int(row[5])) for row in rows}
    issues: list[str] = []
    missing = sorted(set(expected_columns) - set(actual))
    extra = sorted(set(actual) - set(expected_columns))
    if missing:
        issues.append(f"{table_name} missing columns: {', '.join(missing)}")
    if extra:
        issues.append(f"{table_name} has unexpected columns: {', '.join(extra)}")
    for column_name, (expected_type, expected_pk) in expected_columns.items():
        if column_name not in actual:
            continue
        actual_type, actual_pk = actual[column_name]
        if actual_type != expected_type:
            issues.append(f"{table_name}.{column_name} type is {actual_type}, expected {expected_type}")
        if actual_pk != expected_pk:
            issues.append(f"{table_name}.{column_name} primary key rank is {actual_pk}, expected {expected_pk}")
    return issues


def table_index_issues(
    connection: sqlite3.Connection,
    table_name: str,
    expected_indexes: set[str],
) -> list[str]:
    existing_indexes = {str(row[1]) for row in connection.execute(f"PRAGMA index_list({table_name})").fetchall()}
    missing = sorted(expected_indexes - existing_indexes)
    if not missing:
        return []
    return [f"{table_name} missing indexes: {', '.join(missing)}"]


def table_foreign_key_issues(
    connection: sqlite3.Connection,
    table_name: str,
    expected_foreign_keys: set[tuple[str, str, str]],
) -> list[str]:
    existing_foreign_keys = {(str(row[3]), str(row[2]), str(row[4])) for row in connection.execute(f"PRAGMA foreign_key_list({table_name})").fetchall()}
    missing = sorted(expected_foreign_keys - existing_foreign_keys)
    if not missing:
        return []
    missing_labels = [f"{column}->{foreign_table}.{foreign_column}" for column, foreign_table, foreign_column in missing]
    return [f"{table_name} missing foreign keys: {', '.join(missing_labels)}"]
