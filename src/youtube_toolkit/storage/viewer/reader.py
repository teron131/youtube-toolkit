"""Read live LanceDB metadata and rows for the local viewer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from pathlib import Path
import re
from threading import Lock
from typing import Any, cast

import lancedb
from lancedb.db import DBConnection
from lancedb.query import ColumnOrdering
from lancedb.table import Table
import pyarrow as pa

from ...identity import extract_video_id
from ...search.request import RESULT_MIN_SIMILARITY
from ..connection import filter_contains_case_insensitive, filter_equal, filter_in_values
from ..embeddings import EmbeddingSearchError, embed_text
from ..semantic import search_transcript_chunk_scores_in_table

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
MAX_OFFSET = 10_000_000
SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SEMANTIC_VIDEO_TABLE = "videos"
TRANSCRIPT_CHUNK_TABLE = "video_transcript_chunks"
VIDEO_ID_COLUMN = "youtube_video_id"
SEMANTIC_CACHE_SIZE = 32
UNPARSED_SEARCH_VALUE = object()
EmbedQuery = Callable[[str], list[float]]


class ViewerQueryError(ValueError):
    """A viewer request named an unsupported table, column, filter, or page."""


class TableNotFoundError(ViewerQueryError):
    """A requested table is not present in the connected LanceDB store."""


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    """Internal semantic outcome lets literal search survive unavailable embedding infrastructure."""

    status: str
    scores: dict[str, float]
    detail: str | None = None

    def payload(self) -> dict[str, Any]:
        """Expose enough status for the browser to describe the active search behavior."""
        payload: dict[str, Any] = {
            "status": self.status,
            "matches": len(self.scores),
        }
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ReadOnlyLanceViewer:
    """Read current table metadata and rows without invoking storage setup or maintenance."""

    def __init__(
        self,
        database_path: Path,
        *,
        embed_query: EmbedQuery = embed_text,
    ) -> None:
        self.database_path = database_path.resolve()
        self.embed_query = embed_query
        self._semantic_score_cache: dict[tuple[int, str], dict[str, float]] = {}
        self._semantic_cache_lock = Lock()
        if not self.database_path.is_dir():
            raise RuntimeError(f"LanceDB store does not exist: {self.database_path}")

    def overview(self) -> dict[str, Any]:
        """Return live database and per-table counts for navigation."""
        database = self._connect()
        tables = []
        total_rows = 0
        total_table_bytes = 0
        total_indexes = 0
        for table_name in sorted(database.list_tables().tables):
            table = open_pinned_table(database, table_name)
            stats = cast(dict[str, Any], table.stats())
            row_count = int(stats["num_rows"])
            table_bytes = int(stats["total_bytes"])
            index_count = int(stats["num_indices"])
            total_rows += row_count
            total_table_bytes += table_bytes
            total_indexes += index_count
            tables.append(
                {
                    "name": table_name,
                    "rows": row_count,
                    "columns": len(table.schema.names),
                    "bytes": table_bytes,
                    "indexes": index_count,
                }
            )
        return {
            "path": str(self.database_path),
            "bytes": directory_size(self.database_path),
            "table_bytes": total_table_bytes,
            "table_count": len(tables),
            "row_count": total_rows,
            "index_count": total_indexes,
            "tables": tables,
        }

    def table_detail(self, table_name: str) -> dict[str, Any]:
        """Return one table's schema, indexes, and statistics."""
        database = self._connect()
        table = open_database_table(database, table_name)
        table_names = set(database.list_tables().tables)
        schema = [field_payload(field) for field in table.schema]
        indexes = [
            {
                "name": index.name,
                "type": index.index_type,
                "columns": index.columns,
                "created_at": json_value(index.created_at),
                "indexed_rows": index.num_indexed_rows,
                "unindexed_rows": index.num_unindexed_rows,
                "bytes": index.size_bytes,
            }
            for index in table.list_indices()
        ]
        return {
            "name": table_name,
            "schema": schema,
            "indexes": indexes,
            "stats": json_value(table.stats()),
            "semantic_search": table_name == SEMANTIC_VIDEO_TABLE and TRANSCRIPT_CHUNK_TABLE in table_names,
        }

    def table_rows(
        self,
        table_name: str,
        *,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
        sort_column: str | None = None,
        sort_direction: str = "asc",
        search_text: str | None = None,
        filter_column: str | None = None,
        filter_operator: str | None = None,
        filter_value: str | None = None,
    ) -> dict[str, Any]:
        """Return one bounded live row page with optional validated sorting and filtering."""
        if offset < 0:
            raise ViewerQueryError("offset must be zero or greater")
        if offset > MAX_OFFSET:
            raise ViewerQueryError(f"offset must be {MAX_OFFSET} or less")
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ViewerQueryError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        if sort_direction not in {"asc", "desc"}:
            raise ViewerQueryError("sort direction must be asc or desc")

        database = self._connect()
        table = open_database_table(database, table_name)
        fields = {field.name: field for field in table.schema}
        if sort_column is not None:
            sort_field = require_field(fields, sort_column)
            if not is_sortable(sort_field.type):
                raise ViewerQueryError(f"column cannot be sorted: {sort_column}")

        search_sql = build_search(fields, search_text)
        filter_sql = build_filter(fields, filter_column, filter_operator, filter_value)
        semantic_search = self._semantic_video_search(
            database,
            table_name,
            search_text,
        )
        if semantic_search.scores and search_sql is not None and sort_column is None:
            return blended_video_page(
                table,
                offset=offset,
                limit=limit,
                literal_search_sql=search_sql,
                filter_sql=filter_sql,
                semantic_search=semantic_search,
                search_text=search_text or "",
            )
        if semantic_search.scores:
            semantic_ids_sql = f"{VIDEO_ID_COLUMN} IN ({filter_in_values(list(semantic_search.scores))})"
            search_sql = combine_alternatives(search_sql, semantic_ids_sql)
        where_sql = combine_filters(search_sql, filter_sql)
        query = table.search()
        if where_sql is not None:
            query = query.where(where_sql)
        if sort_column is not None:
            query = query.order_by(
                [
                    ColumnOrdering(
                        column_name=sort_column,
                        ascending=sort_direction == "asc",
                        nulls_first=False,
                    )
                ]
            )
        rows = query.offset(offset).limit(limit).to_list()
        return {
            "table": table_name,
            "offset": offset,
            "limit": limit,
            "total": table.count_rows(where_sql),
            "search": search_text or "",
            "semantic_search": semantic_search.payload(),
            "rows": json_value(rows),
        }

    def _semantic_video_search(
        self,
        database: DBConnection,
        table_name: str,
        search_text: str | None,
    ) -> SemanticSearchResult:
        """Use existing transcript embeddings when appropriate and degrade to literal results when the API is unavailable."""
        if table_name != SEMANTIC_VIDEO_TABLE or search_text is None or not search_text.strip():
            return SemanticSearchResult("not_applicable", {})
        query = search_text.strip()
        if not semantic_query_candidate(query):
            return SemanticSearchResult("skipped", {}, "The query is a literal value or video reference")
        if TRANSCRIPT_CHUNK_TABLE not in database.list_tables().tables:
            return SemanticSearchResult("unavailable", {}, "No transcript embedding index is stored")
        chunk_table = open_pinned_table(database, TRANSCRIPT_CHUNK_TABLE)
        cache_key = (chunk_table.version, query)
        with self._semantic_cache_lock:
            cached_scores = self._semantic_score_cache.get(cache_key)
        if cached_scores is not None:
            return SemanticSearchResult("applied", cached_scores)
        try:
            scores = search_transcript_chunk_scores_in_table(
                chunk_table,
                query,
                None,
                min_similarity=RESULT_MIN_SIMILARITY,
                embed_text=self.embed_query,
            )
        except EmbeddingSearchError as error:
            return SemanticSearchResult("unavailable", {}, str(error))
        with self._semantic_cache_lock:
            if len(self._semantic_score_cache) >= SEMANTIC_CACHE_SIZE:
                self._semantic_score_cache.pop(next(iter(self._semantic_score_cache)))
            self._semantic_score_cache[cache_key] = scores
        return SemanticSearchResult("applied", scores)

    def _connect(self) -> DBConnection:
        """Open the existing database directly so inspection cannot run toolkit migrations."""
        if not self.database_path.is_dir():
            raise RuntimeError(f"LanceDB store does not exist: {self.database_path}")
        return lancedb.connect(
            self.database_path,
            read_consistency_interval=timedelta(0),
            manifest_enabled=False,
        )


def open_database_table(database: DBConnection, table_name: str) -> Table:
    """Validate a live table name before pinning the request to its latest version."""
    if table_name not in database.list_tables().tables:
        raise TableNotFoundError(f"table does not exist: {table_name}")
    return open_pinned_table(database, table_name)


def open_pinned_table(database: DBConnection, table_name: str) -> Table:
    """Pin one request to the version that was current when the table was opened."""
    latest = database.open_table(table_name)
    return database.open_table(table_name, version=latest.version)


def semantic_query_candidate(query: str) -> bool:
    """Reserve embedding calls for natural-language queries rather than scalar values and exact video references."""
    return any(character.isalpha() for character in query) and extract_video_id(query) is None


def blended_video_page(
    table: Table,
    *,
    offset: int,
    limit: int,
    literal_search_sql: str,
    filter_sql: str | None,
    semantic_search: SemanticSearchResult,
    search_text: str,
) -> dict[str, Any]:
    """Page literal video hits first, then append distinct semantic hits in similarity order."""
    literal_where_sql = cast(str, combine_filters(literal_search_sql, filter_sql))
    semantic_ids_sql = f"{VIDEO_ID_COLUMN} IN ({filter_in_values(list(semantic_search.scores))})"
    semantic_where_sql = cast(str, combine_filters(semantic_ids_sql, filter_sql))
    literal_total = table.count_rows(literal_where_sql)

    literal_rows: list[dict[str, Any]] = []
    if offset < literal_total:
        literal_rows = table.search().where(literal_where_sql).offset(offset).limit(limit).to_list()

    overlap_where_sql = cast(str, combine_filters(literal_where_sql, semantic_ids_sql))
    overlap_rows = table.search().where(overlap_where_sql).select([VIDEO_ID_COLUMN]).limit(len(semantic_search.scores)).to_list()
    literal_semantic_ids = {row[VIDEO_ID_COLUMN] for row in overlap_rows}
    semantic_rows = table.search().where(semantic_where_sql).limit(len(semantic_search.scores)).to_list()
    semantic_rows_by_id = {row[VIDEO_ID_COLUMN]: row for row in semantic_rows}
    semantic_only_rows = [
        semantic_rows_by_id[video_id]
        for video_id in sorted(
            semantic_search.scores,
            key=semantic_search.scores.__getitem__,
            reverse=True,
        )
        if video_id not in literal_semantic_ids and video_id in semantic_rows_by_id
    ]

    semantic_offset = max(0, offset - literal_total)
    remaining = limit - len(literal_rows)
    page_rows = [
        *literal_rows,
        *semantic_only_rows[semantic_offset : semantic_offset + remaining],
    ]
    return {
        "table": SEMANTIC_VIDEO_TABLE,
        "offset": offset,
        "limit": limit,
        "total": literal_total + len(semantic_only_rows),
        "search": search_text,
        "semantic_search": semantic_search.payload(),
        "rows": json_value(page_rows),
    }


def field_payload(field: pa.Field) -> dict[str, Any]:
    """Describe one Arrow field for browser controls and structure inspection."""
    return {
        "name": field.name,
        "type": str(field.type),
        "nullable": field.nullable,
        "sortable": is_sortable(field.type),
        "filterable": is_filterable(field.type),
        "text": is_text(field.type),
    }


def is_text(data_type: pa.DataType) -> bool:
    """Return whether a field supports case-insensitive substring filtering."""
    return pa.types.is_string(data_type) or pa.types.is_large_string(data_type)


def is_filterable(data_type: pa.DataType) -> bool:
    """Limit equality filtering to scalar types Lance SQL can compare directly."""
    return is_text(data_type) or pa.types.is_boolean(data_type) or pa.types.is_integer(data_type) or pa.types.is_floating(data_type)


def is_sortable(data_type: pa.DataType) -> bool:
    """Exclude nested raw values from sort controls."""
    return not (
        pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type) or pa.types.is_struct(data_type) or pa.types.is_map(data_type)
    )


def require_field(fields: dict[str, pa.Field], column_name: str) -> pa.Field:
    """Return a trusted schema field or reject the caller-provided column name."""
    field = fields.get(column_name)
    if field is None:
        raise ViewerQueryError(f"column does not exist: {column_name}")
    return field


def build_filter(
    fields: dict[str, pa.Field],
    column_name: str | None,
    operator: str | None,
    raw_value: str | None,
) -> str | None:
    """Translate validated browser filter controls into a Lance SQL predicate."""
    if column_name is None and operator is None and raw_value is None:
        return None
    if column_name is None or operator is None:
        raise ViewerQueryError("filter column and operator are required together")

    field = require_field(fields, column_name)
    if SAFE_SQL_IDENTIFIER.fullmatch(column_name) is None:
        raise ViewerQueryError(f"column cannot be used in a SQL filter: {column_name}")
    if operator == "is_null":
        return f"{column_name} IS NULL"
    if operator == "is_not_null":
        return f"{column_name} IS NOT NULL"
    if raw_value is None or raw_value == "":
        raise ViewerQueryError("filter value is required")
    if operator == "contains":
        if not is_text(field.type):
            raise ViewerQueryError(f"contains is only supported for text columns: {column_name}")
        return filter_contains_case_insensitive(column_name, raw_value)
    if operator != "equals":
        raise ViewerQueryError(f"unsupported filter operator: {operator}")
    if not is_filterable(field.type):
        raise ViewerQueryError(f"column cannot be filtered by equality: {column_name}")
    return filter_equal(column_name, parse_scalar_filter(field.type, raw_value))


def build_search(
    fields: dict[str, pa.Field],
    raw_value: str | None,
) -> str | None:
    """Search text by substring and scalar columns by an exact parsed value."""
    if raw_value is None or not raw_value.strip():
        return None
    searchable_fields = {name: field for name, field in fields.items() if is_filterable(field.type) and SAFE_SQL_IDENTIFIER.fullmatch(name) is not None}
    if not searchable_fields:
        raise ViewerQueryError("table has no searchable scalar columns")

    query = raw_value.strip()
    predicates = [filter_contains_case_insensitive(column_name, query) for column_name, field in searchable_fields.items() if is_text(field.type)]
    for column_name, field in searchable_fields.items():
        if is_text(field.type):
            continue
        parsed_value = parse_search_scalar(field.type, query)
        if parsed_value is not UNPARSED_SEARCH_VALUE:
            predicates.append(filter_equal(column_name, parsed_value))
    return "(" + " OR ".join(predicates) + ")" if predicates else "FALSE"


def parse_search_scalar(
    data_type: pa.DataType,
    raw_value: str,
) -> object:
    """Parse a free-form query only when it is an exact scalar value for the field type."""
    try:
        if pa.types.is_boolean(data_type):
            normalized = raw_value.casefold()
            return normalized == "true" if normalized in {"true", "false"} else UNPARSED_SEARCH_VALUE
        if pa.types.is_integer(data_type):
            return int(raw_value)
        if pa.types.is_floating(data_type):
            value = float(raw_value)
            return value if math.isfinite(value) else UNPARSED_SEARCH_VALUE
    except ValueError:
        return UNPARSED_SEARCH_VALUE
    return UNPARSED_SEARCH_VALUE


def combine_filters(*filters: str | None) -> str | None:
    """Combine independently validated viewer predicates without changing precedence."""
    active_filters = [filter_sql for filter_sql in filters if filter_sql is not None]
    if not active_filters:
        return None
    return " AND ".join(f"({filter_sql})" for filter_sql in active_filters)


def combine_alternatives(*filters: str | None) -> str | None:
    """Combine independent search strategies so a row may match either one."""
    active_filters = [filter_sql for filter_sql in filters if filter_sql is not None]
    if not active_filters:
        return None
    return " OR ".join(f"({filter_sql})" for filter_sql in active_filters)


def parse_scalar_filter(data_type: pa.DataType, raw_value: str) -> str | bool | int | float:
    """Parse one equality value according to the selected Arrow scalar type."""
    try:
        if pa.types.is_boolean(data_type):
            normalized = raw_value.casefold()
            if normalized not in {"true", "false"}:
                raise ViewerQueryError("boolean filters must be true or false")
            return normalized == "true"
        if pa.types.is_integer(data_type):
            return int(raw_value)
        if pa.types.is_floating(data_type):
            return float(raw_value)
    except ValueError as error:
        raise ViewerQueryError(f"invalid {data_type} filter value: {raw_value}") from error
    return raw_value


def json_value(value: Any) -> Any:
    """Normalize LanceDB and Arrow values into strict JSON without hiding raw fields."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    return str(value)


def directory_size(path: Path) -> int:
    """Measure the current on-disk database directory without failing on concurrent file replacement."""
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total
