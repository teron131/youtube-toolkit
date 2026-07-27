"""Read-only LanceDB viewer tests cover live metadata, raw rows, CLI wiring, and HTTP delivery."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.client import HTTPConnection
import json
from pathlib import Path
from threading import Thread
from typing import cast

import lancedb
from lancedb.index import BTree
import pyarrow as pa
import pytest

from youtube_toolkit.cli import commands
from youtube_toolkit.cli.__main__ import main
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.embeddings import EmbeddingError
from youtube_toolkit.storage import viewer as storage_viewer
from youtube_toolkit.storage.viewer import ReadOnlyLanceViewer, ViewerQueryError, create_database_viewer_server


@pytest.fixture
def viewer_database(tmp_path: Path) -> Path:
    """Create a small typed table with an index and fixed-size vectors for viewer tests."""
    database_path = tmp_path / "viewer.lancedb"
    database = lancedb.connect(database_path)
    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("category", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("embedding", pa.list_(pa.float32(), 3)),
        ]
    )
    data = pa.Table.from_pylist(
        [
            {"name": "Alpha", "category": "first", "score": 1.25, "embedding": [1.0, 2.0, 3.0]},
            {"name": "Beta", "category": "target", "score": 3.75, "embedding": [4.0, 5.0, 6.0]},
            {"name": "Gamma", "category": "third", "score": 2.5, "embedding": [7.0, 8.0, 9.0]},
        ],
        schema=schema,
    )
    table = database.create_table("records", data=data)
    table.create_index("name", config=BTree(), name="records_name_idx")
    return database_path


def test_viewer_reads_live_database_without_creating_or_mutating_storage(
    viewer_database: Path,
    tmp_path: Path,
) -> None:
    database = lancedb.connect(viewer_database)
    versions_before = {name: database.open_table(name).version for name in database.list_tables().tables}
    viewer = ReadOnlyLanceViewer(viewer_database)

    overview = viewer.overview()
    viewer.table_detail("records")
    viewer.table_rows("records")

    versions_after = {name: database.open_table(name).version for name in database.list_tables().tables}
    assert overview["path"] == str(viewer_database)
    assert overview["table_count"] == 1
    assert overview["row_count"] == 3
    assert overview["tables"][0]["rows"] == 3
    assert versions_after == versions_before

    database.open_table("records").add([{"name": "Delta", "category": "fourth", "score": 4.5, "embedding": [10.0, 11.0, 12.0]}])
    assert viewer.overview()["row_count"] == 4

    missing_path = tmp_path / "missing.lancedb"
    with pytest.raises(RuntimeError, match="LanceDB store does not exist"):
        ReadOnlyLanceViewer(missing_path)
    assert not missing_path.exists()


def test_viewer_exposes_schema_indexes_and_complete_raw_rows(viewer_database: Path) -> None:
    viewer = ReadOnlyLanceViewer(viewer_database)

    detail = viewer.table_detail("records")
    page = viewer.table_rows("records", offset=0, limit=1, sort_column="score", sort_direction="desc")

    embedding_field = next(field for field in detail["schema"] if field["name"] == "embedding")
    assert embedding_field == {
        "name": "embedding",
        "type": "fixed_size_list<item: float>[3]",
        "nullable": True,
        "sortable": False,
        "filterable": False,
        "text": False,
    }
    assert detail["indexes"] == [
        {
            "name": "records_name_idx",
            "type": "BTree",
            "columns": ["name"],
            "created_at": detail["indexes"][0]["created_at"],
            "indexed_rows": 3,
            "unindexed_rows": 0,
            "bytes": detail["indexes"][0]["bytes"],
        }
    ]
    assert page["total"] == 3
    assert page["rows"][0]["name"] == "Beta"
    assert page["rows"][0]["embedding"] == pytest.approx([4.0, 5.0, 6.0])


def test_viewer_validates_filters_sorting_and_pagination(viewer_database: Path) -> None:
    viewer = ReadOnlyLanceViewer(viewer_database)

    contains_page = viewer.table_rows(
        "records",
        filter_column="name",
        filter_operator="contains",
        filter_value="BET",
    )
    score_page = viewer.table_rows(
        "records",
        filter_column="score",
        filter_operator="equals",
        filter_value="3.75",
    )
    search_page = viewer.table_rows("records", search_text="TARGET")
    numeric_search_page = viewer.table_rows("records", search_text="3.75")
    combined_page = viewer.table_rows(
        "records",
        search_text="target",
        filter_column="score",
        filter_operator="equals",
        filter_value="3.75",
    )
    sorted_page = viewer.table_rows("records", offset=1, limit=1, sort_column="score", sort_direction="desc")

    assert [row["name"] for row in contains_page["rows"]] == ["Beta"]
    assert [row["name"] for row in score_page["rows"]] == ["Beta"]
    assert [row["name"] for row in search_page["rows"]] == ["Beta"]
    assert [row["name"] for row in numeric_search_page["rows"]] == ["Beta"]
    assert [row["name"] for row in combined_page["rows"]] == ["Beta"]
    assert [row["name"] for row in sorted_page["rows"]] == ["Gamma"]
    with pytest.raises(ViewerQueryError, match="column cannot be sorted"):
        viewer.table_rows("records", sort_column="embedding")
    with pytest.raises(ViewerQueryError, match="column does not exist"):
        viewer.table_rows("records", filter_column="missing", filter_operator="equals", filter_value="value")
    with pytest.raises(ViewerQueryError, match="limit must be between"):
        viewer.table_rows("records", limit=101)
    with pytest.raises(ViewerQueryError, match="offset must be"):
        viewer.table_rows("records", offset=-1)


def test_viewer_blends_literal_scalar_and_read_only_semantic_video_search(tmp_path: Path) -> None:
    """Video search keeps exact scalar hits while reusing stored transcript vectors without changing table versions."""
    database_path = tmp_path / "semantic-viewer.lancedb"
    database = lancedb.connect(database_path)
    database.create_table(
        "videos",
        data=pa.Table.from_pylist(
            [
                {"youtube_video_id": "literal-hit", "title": "Agent memory field notes", "duration": 42.0},
                {"youtube_video_id": "semantic-hit", "title": "Durable application state", "duration": 186.0},
                {"youtube_video_id": "unrelated", "title": "Cooking dinner", "duration": 60.0},
            ],
            schema=pa.schema(
                [
                    pa.field("youtube_video_id", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("duration", pa.float64()),
                ]
            ),
        ),
    )
    database.create_table(
        "video_transcript_chunks",
        data=pa.Table.from_pylist(
            [
                {"video_id": "literal-hit", "chunk_index": 0, "text": "literal", "embedding": [0.0, 1.0, 0.0]},
                {"video_id": "semantic-hit", "chunk_index": 0, "text": "semantic", "embedding": [1.0, 0.0, 0.0]},
                {"video_id": "unrelated", "chunk_index": 0, "text": "other", "embedding": [-1.0, 0.0, 0.0]},
            ],
            schema=pa.schema(
                [
                    pa.field("video_id", pa.string()),
                    pa.field("chunk_index", pa.int64()),
                    pa.field("text", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32(), 3)),
                ]
            ),
        ),
    )
    versions_before = {name: database.open_table(name).version for name in database.list_tables().tables}
    embedding_queries: list[str] = []

    def embed_query(query: str) -> list[float]:
        embedding_queries.append(query)
        return [1.0, 0.0, 0.0]

    viewer = ReadOnlyLanceViewer(database_path, embed_query=embed_query)
    semantic_page = viewer.table_rows("videos", search_text="agent memory")
    semantic_second_page = viewer.table_rows("videos", offset=1, limit=1, search_text="agent memory")
    assert embedding_queries == ["agent memory"]
    embedding_queries.clear()
    numeric_page = viewer.table_rows("videos", search_text="186")

    assert viewer.table_detail("videos")["semantic_search"] is True
    assert [row["youtube_video_id"] for row in semantic_page["rows"]] == ["literal-hit", "semantic-hit"]
    assert [row["youtube_video_id"] for row in semantic_second_page["rows"]] == ["semantic-hit"]
    assert semantic_page["semantic_search"] == {"status": "applied", "matches": 1}
    assert [row["youtube_video_id"] for row in numeric_page["rows"]] == ["semantic-hit"]
    assert numeric_page["semantic_search"]["status"] == "skipped"
    assert embedding_queries == []

    def unavailable_embedding(_query: str) -> list[float]:
        raise EmbeddingError("Embedding API is not configured")

    fallback_page = ReadOnlyLanceViewer(database_path, embed_query=unavailable_embedding).table_rows(
        "videos",
        search_text="agent memory",
    )
    assert [row["youtube_video_id"] for row in fallback_page["rows"]] == ["literal-hit"]
    assert fallback_page["semantic_search"] == {
        "status": "unavailable",
        "matches": 0,
        "detail": "Embedding API is not configured",
    }
    versions_after = {name: database.open_table(name).version for name in database.list_tables().tables}
    assert versions_after == versions_before


def test_viewer_http_server_delivers_live_page_and_get_only_json(viewer_database: Path) -> None:
    database = lancedb.connect(viewer_database)
    version_before = database.open_table("records").version
    server = create_database_viewer_server(viewer_database, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)

    def request(path: str, *, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection(host, port, timeout=5)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    try:
        page_status, page_headers, page_body = request("/")
        page = page_body.decode()
        assert page_status == HTTPStatus.OK
        assert page_headers["Cache-Control"] == "no-store"
        assert page_headers["X-Content-Type-Options"] == "nosniff"
        assert "LanceDB Viewer" in page
        assert 'id="rowInspector"' in page
        assert 'id="versionsPanel"' not in page
        assert "Apply one column rule" not in page
        assert "<dialog" not in page
        assert '<link rel="stylesheet" href="/viewer.css">' in page
        assert '<script src="/viewer.js" defer></script>' in page
        assert 'id="themeToggle"' in page
        assert "'unsafe-inline'" not in page_headers["Content-Security-Policy"]

        stylesheet_status, stylesheet_headers, stylesheet_body = request("/viewer.css")
        script_status, script_headers, script_body = request("/viewer.js")
        assert stylesheet_status == HTTPStatus.OK
        assert stylesheet_headers["Content-Type"] == "text/css; charset=utf-8"
        assert b".app" in stylesheet_body
        assert b':root[data-theme="light"]' in stylesheet_body
        assert script_status == HTTPStatus.OK
        assert script_headers["Content-Type"] == "text/javascript; charset=utf-8"
        assert b"loadOverview" in script_body
        assert b"lancedb-viewer-theme" in script_body

        overview_status, _, overview_body = request("/api/overview")
        rows_status, _, rows_body = request("/api/tables/records/rows?limit=1&offset=0")
        overview = json.loads(overview_body)
        payload = json.loads(rows_body)
        assert overview_status == HTTPStatus.OK
        assert rows_status == HTTPStatus.OK
        assert overview["table_count"] == 1
        assert payload["total"] == 3
        assert len(payload["rows"]) == 1

        write_status, _, write_body = request("/api/tables/records", method="POST")
        assert write_status == HTTPStatus.METHOD_NOT_ALLOWED
        assert json.loads(write_body) == {"error": "viewer API is read-only"}
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert database.open_table("records").version == version_before


def test_database_view_cli_forwards_active_path_without_json_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage_home = tmp_path / "state"
    calls: list[tuple[Path, int, bool]] = []
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))
    monkeypatch.setattr(
        storage_viewer,
        "serve_database_viewer",
        lambda path, *, port, open_browser: calls.append((path, port, open_browser)),
    )

    parsed = build_parser().parse_args(["db", "view", "--port", "8042", "--no-open"])
    assert parsed.command == "db"
    assert parsed.database_command == "view"
    assert parsed.port == 8042
    assert parsed.no_open is True
    assert commands.DatabaseCommands.view(argparse.Namespace(port=8042, no_open=True)) is None
    assert main(["db", "view", "--port", "8043", "--no-open"]) == 0
    assert calls == [
        (storage_home / "youtube.lancedb", 8042, False),
        (storage_home / "youtube.lancedb", 8043, False),
    ]
    assert capsys.readouterr().out == ""
