from pathlib import Path

import lancedb
import pytest

import youtube_toolkit.storage.connection as storage_connection
from youtube_toolkit import EmbeddingConfig
from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.storage import (
    DATABASE_FILENAME,
    STORAGE_DIRNAME,
    analyze_transcript_chunk_search,
    auto_index_transcript_chunks,
    auto_index_video_embedding,
    database_schema_health,
    embed_text_with_gemini,
    explain_transcript_chunk_search,
    get_connection,
    get_stored_video,
    index_search_candidate_titles,
    index_transcript_chunks,
    index_video_embedding,
    index_video_title_embedding,
    index_video_title_embeddings,
    resolve_storage_dir,
    search_candidate_title_scores,
    search_stored_videos,
    search_transcript_chunks,
    search_transcript_chunk_scores,
    search_video_title_scores,
    split_transcript_chunks,
    store_search_candidates,
    upsert_watchlist_channel,
    upsert_video,
    upsert_video_summary,
)
from youtube_toolkit.storage.connection import SCALAR_INDEX_COLUMNS, TRANSCRIPT_CHUNK_TEXT_INDEX


def test_default_storage_path_is_repo_cache(monkeypatch, tmp_path) -> None:
    toolkit_root = Path(__file__).resolve().parents[1]
    outside_repo = tmp_path / "other"
    outside_repo.mkdir()
    (outside_repo / ".git").mkdir()
    monkeypatch.chdir(outside_repo)
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STORAGE_HOME", raising=False)

    assert STORAGE_DIRNAME == ".cache"
    assert DATABASE_FILENAME == "youtube.lancedb"
    assert resolve_storage_dir() == toolkit_root / ".cache"


def test_installed_storage_path_is_user_app_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(storage_connection, "__file__", str(installed_connection_file(tmp_path)))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STORAGE_HOME", raising=False)

    assert resolve_storage_dir() == tmp_path / "home" / "Library" / "Application Support" / "youtube-toolkit"


def installed_connection_file(tmp_path: Path) -> Path:
    return tmp_path / "tool" / "site-packages" / "youtube_toolkit" / "storage" / "connection.py"


def test_database_schema_health_reports_missing_db(tmp_path) -> None:
    db_path = tmp_path / "missing.lancedb"

    assert database_schema_health(db_path) == {
        "path": str(db_path),
        "ok": False,
        "status": "missing",
        "detail": "LanceDB store is missing",
        "repair": "Run youtube setup database to create a fresh LanceDB store",
    }


def test_database_schema_health_accepts_current_schema(monkeypatch, tmp_path) -> None:
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    with get_connection():
        pass

    assert database_schema_health(storage_home / DATABASE_FILENAME) == {
        "path": str(storage_home / DATABASE_FILENAME),
        "ok": True,
        "status": "ready",
        "detail": "LanceDB store has expected base tables",
    }


def test_connection_creates_base_scalar_indexes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        for table_name, column_names in SCALAR_INDEX_COLUMNS.items():
            if table_name not in connection.table_names():
                continue
            index_names = {index.name for index in connection.table(table_name).list_indices()}
            assert {f"{table_name}_{column_name}_idx" for column_name in column_names}.issubset(index_names)


def test_database_schema_health_warns_on_missing_lance_tables(tmp_path) -> None:
    db_path = tmp_path / DATABASE_FILENAME
    database = lancedb.connect(db_path)
    database.create_table("legacy_videos", data=[{"id": "legacy"}])

    payload = database_schema_health(db_path)

    assert payload["path"] == str(db_path)
    assert payload["ok"] is False
    assert payload["status"] == "invalid"
    assert "missing tables:" in payload["detail"]
    assert payload["repair"] == "Run youtube setup database to create a fresh LanceDB store"


def test_upsert_video_converts_title_to_hong_kong_traditional(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI商业发展暗藏机会",
            },
        )
        row = connection.find_row("videos", "youtube_video_id", "JpSHyEIZ_bo")

    assert row is not None
    assert row["title"] == "AI商業發展暗藏機會"


def test_upsert_video_returns_existing_values_for_sparse_update(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes",
                "upload_date": "20260621",
            },
        )
        payload = upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
            },
            transcript="既有逐字稿",
        )

    assert payload is not None
    assert {
        key: payload[key]
        for key in (
            "video_id",
            "title",
            "upload_date",
            "duration",
            "view_count",
            "like_count",
            "channel_id",
            "channel_handle",
            "transcript",
        )
    } == {
        "video_id": "JpSHyEIZ_bo",
        "title": "AI market notes",
        "upload_date": "2026-06-21",
        "duration": None,
        "view_count": None,
        "like_count": None,
        "channel_id": None,
        "channel_handle": None,
        "transcript": "既有逐字稿",
    }
    assert payload["metadata_fetched_at"] is not None
    assert payload["transcript_fetched_at"] is not None


def test_upsert_video_stores_metadata_refresh_fields(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        payload = upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes",
                "duration": 123,
                "view_count": 1000,
                "like_count": 50,
            },
        )
        updated_payload = upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes refreshed",
            },
        )

    assert payload is not None
    assert payload["duration"] == 123.0
    assert payload["view_count"] == 1000
    assert payload["like_count"] == 50
    assert payload["metadata_fetched_at"] is not None
    assert payload["transcript_fetched_at"] is None
    assert updated_payload is not None
    assert updated_payload["title"] == "AI market notes refreshed"
    assert updated_payload["duration"] == 123.0
    assert updated_payload["view_count"] == 1000
    assert updated_payload["like_count"] == 50


def test_upsert_video_stores_youtube_channel_id(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        payload = upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "channel_id": "UC123",
                "channel_handle": "@OpenAI",
            },
        )
        row = connection.find_row("videos", "youtube_video_id", "JpSHyEIZ_bo")

    assert row is not None
    assert row["channel_id"] == "UC123"
    assert payload is not None
    assert payload["channel_id"] == "UC123"
    assert payload["channel_handle"] == "@OpenAI"


def test_upsert_video_links_known_channel_from_handle_only_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        payload = upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "channel_handle": "@OpenAI",
            },
        )

    assert payload is not None
    assert payload["channel_id"] == "UC123"
    assert payload["channel_handle"] == "@OpenAI"


def test_upsert_video_clears_embedding_when_channel_handle_changes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
                "channel_id": "UC123",
                "channel_handle": "@OldChannel",
            },
            transcript="晶片需求和 GPU 投资回升",
        )
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)

        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "channel_id": "UC123",
                "channel_handle": "@NewChannel",
            },
        )
        row = connection.find_row("video_embeddings", "video_id", "JpSHyEIZ_bo")

    assert row is None


def test_upsert_video_clears_embedding_when_channel_id_changes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
                "channel_id": "UC123",
                "channel_handle": "@OldChannel",
            },
            transcript="晶片需求和 GPU 投资回升",
        )
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)

        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "channel_id": "UC456",
                "channel_handle": "@NewChannel",
            },
        )
        row = connection.find_row("video_embeddings", "video_id", "JpSHyEIZ_bo")

    assert row is None


def test_upsert_video_keeps_embedding_when_upload_date_changes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
                "upload_date": "20260620",
            },
            transcript="晶片需求和 GPU 投资回升",
        )
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)

        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "upload_date": "20260621",
            },
        )

    with get_connection() as connection:
        row = connection.find_row("video_embeddings", "video_id", "JpSHyEIZ_bo")

    assert row is not None
    assert row["embedding_content_hash"] is not None
    assert row["embedding"] is not None


def test_get_stored_video_returns_one_payload_by_url(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes",
            },
            transcript="逐字稿",
        )
        payload = get_stored_video(
            connection,
            "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
            require_transcript=True,
        )

    assert payload is not None
    assert payload["video_id"] == "JpSHyEIZ_bo"
    assert payload["title"] == "AI market notes"
    assert payload["transcript"] == "逐字稿"


def test_search_stored_videos_finds_by_id_and_url(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes",
                "upload_date": "20260621",
            },
            transcript="既有逐字稿",
        )
        expected = {
            "video_id": "JpSHyEIZ_bo",
            "title": "AI market notes",
            "upload_date": "2026-06-21",
            "duration": None,
            "view_count": None,
            "like_count": None,
            "channel_id": None,
            "channel_handle": None,
            "transcript": "既有逐字稿",
        }
        id_results = search_stored_videos(connection, "JpSHyEIZ_bo")
        url_results = search_stored_videos(connection, "https://www.youtube.com/watch?v=JpSHyEIZ_bo")
        assert [{key: result[key] for key in expected} for result in id_results] == [expected]
        assert [{key: result[key] for key in expected} for result in url_results] == [expected]
        assert id_results[0]["metadata_fetched_at"] is not None
        assert id_results[0]["transcript_fetched_at"] is not None


def test_search_stored_videos_finds_keyword_across_row(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes",
            },
            transcript="逐字稿提到半导体机会",
        )
        results = search_stored_videos(connection, "半导体")

    assert len(results) == 1
    assert results[0]["video_id"] == "JpSHyEIZ_bo"
    assert results[0]["title"] == "AI market notes"
    assert results[0]["transcript"] == "逐字稿提到半导体机会"


def test_search_stored_videos_can_require_transcript(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "No transcript yet",
            },
        )
        assert search_stored_videos(connection, "No transcript") != []
        assert search_stored_videos(connection, "No transcript", require_transcript=True) == []


def test_search_stored_videos_defaults_to_upload_date_sort(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Market update",
                "upload_date": "20260620",
            },
            transcript="same keyword",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Market update",
                "upload_date": "20260618",
            },
            transcript="same keyword",
        )
        results = search_stored_videos(
            connection,
            "Market",
        )

    assert [result["video_id"] for result in results] == [
        "JpSHyEIZ_bo",
        "dQw4w9WgXcQ",
    ]


def test_search_stored_videos_rejects_relevance_sort_without_semantic_search(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection, pytest.raises(ValueError, match="relevance sort"):
        search_stored_videos(
            connection,
            "Market",
            sort="relevance",
        )


def test_search_stored_videos_semantic_search_uses_saved_embeddings(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
                "channel_id": "UC123",
                "channel_handle": "@MarketChannel",
            },
            transcript="晶片需求和 GPU 投资回升",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Kitchen notes",
            },
            transcript="番茄酱和意大利面",
        )
        embedded_texts: list[str] = []

        def capture_embed(text: str) -> list[float]:
            embedded_texts.append(text)
            return fake_embed(text)

        embedding_payload = index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=capture_embed)
        assert embedding_payload is not None
        assert embedding_payload.pop("content_hash")
        assert embedded_texts == ["Semiconductor cycle\n@MarketChannel\n晶片需求和 GPU 投资回升"]
        assert embedding_payload == {
            "video_id": "JpSHyEIZ_bo",
            "dimensions": 2,
        }
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

    with get_connection() as connection:
        stored_embedding = connection.find_row("video_embeddings", "video_id", "JpSHyEIZ_bo")

    assert stored_embedding is not None
    assert stored_embedding["embedding_content_hash"]
    assert stored_embedding["embedding"] == pytest.approx(fake_embed("GPU capex"))

    with get_connection() as connection:
        results = search_stored_videos(
            connection,
            "GPU capex",
            semantic=True,
            embed_text=fake_embed,
        )

    assert results[0]["video_id"] == "JpSHyEIZ_bo"
    assert results[0]["semantic_score"] == pytest.approx(1.0)


def test_video_title_embedding_scores_title_matches(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Introducing Claude Fable 5",
            },
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Interview with the research team",
            },
        )
        index_video_title_embedding(connection, "JpSHyEIZ_bo", embed_text=title_embed)
        index_video_title_embedding(connection, "dQw4w9WgXcQ", embed_text=title_embed)
        scores = search_video_title_scores(
            connection,
            "product announcement",
            [
                "JpSHyEIZ_bo",
                "dQw4w9WgXcQ",
            ],
            embed_text=title_embed,
        )

    assert scores["JpSHyEIZ_bo"] > scores["dQw4w9WgXcQ"]


def test_video_title_embeddings_are_indexed_concurrently(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    from threading import Barrier, Lock

    barrier = Barrier(2)
    lock = Lock()
    active_count = 0
    max_active_count = 0

    def embed_title(text: str) -> list[float]:
        nonlocal active_count, max_active_count
        with lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        try:
            barrier.wait(timeout=2)
        finally:
            with lock:
                active_count -= 1
        return fake_embed(text)

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "GPU capex update",
            },
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Semiconductor cycle",
            },
        )
        results = index_video_title_embeddings(
            connection,
            [
                "JpSHyEIZ_bo",
                "dQw4w9WgXcQ",
            ],
            embed_text=embed_title,
        )

    assert max_active_count == 2
    assert {result["video_id"] for result in results if result["status"] == "indexed"} == {
        "JpSHyEIZ_bo",
        "dQw4w9WgXcQ",
    }


def test_search_candidate_title_scores_use_staged_rows(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        store_search_candidates(
            connection,
            "search-session",
            "product announcement",
            [
                {
                    "id": "JpSHyEIZ_bo",
                    "source": "youtube",
                    "title": "Introducing Claude Fable 5",
                    "view_count": 10_000,
                    "duration": 120,
                },
                {
                    "id": "dQw4w9WgXcQ",
                    "source": "youtube",
                    "title": "Interview with the research team",
                    "view_count": 10_000,
                    "duration": 600,
                },
            ],
        )
        index_search_candidate_titles(
            connection,
            "search-session",
            embed_text=title_embed,
        )
        scores = search_candidate_title_scores(
            connection,
            "search-session",
            "product announcement",
            limit=2,
            min_similarity=0.3,
            embed_text=title_embed,
        )
        durable_video = get_stored_video(connection, "JpSHyEIZ_bo")

    assert scores["JpSHyEIZ_bo"] > scores.get("dQw4w9WgXcQ", 0)
    assert durable_video is None


def test_split_transcript_chunks_uses_overlap() -> None:
    chunks = split_transcript_chunks(
        "abcdefghijklmnopqrstuvwxyz",
        chunk_chars=10,
        overlap_chars=3,
    )

    assert chunks == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_transcript_chunk_index_scores_best_matching_chunk(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "GPU topic",
                "channel_id": "UC123",
                "channel_handle": "@Chips",
                "upload_date": "20260703",
                "duration": 600,
                "view_count": 40_000,
                "like_count": 2_000,
            },
            transcript="first chunk about pasta " * 80 + "second chunk about GPU capex " * 80,
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Kitchen topic",
            },
            transcript="only pasta sauce " * 80,
        )
        index_transcript_chunks(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
        index_transcript_chunks(connection, "dQw4w9WgXcQ", embed_text=fake_embed)
        scores = search_transcript_chunk_scores(
            connection,
            "GPU capex",
            [
                "JpSHyEIZ_bo",
                "dQw4w9WgXcQ",
            ],
            embed_text=fake_embed,
        )
        all_video_scores = search_transcript_chunk_scores(
            connection,
            "GPU capex",
            None,
            embed_text=fake_embed,
        )
        filtered_chunks = search_transcript_chunks(
            connection,
            "GPU capex",
            None,
            where_sql="view_count >= 10000 AND duration <= 700",
            columns=[
                "video_id",
                "chunk_index",
                "text",
            ],
            embed_text=fake_embed,
        )
        blocked_chunks = search_transcript_chunks(
            connection,
            "GPU capex",
            None,
            where_sql="view_count >= 100000",
            embed_text=fake_embed,
        )
        explain_plan = explain_transcript_chunk_search(
            connection,
            "GPU capex",
            [
                "JpSHyEIZ_bo",
                "dQw4w9WgXcQ",
            ],
            embed_text=fake_embed,
        )
        analyze_plan = analyze_transcript_chunk_search(
            connection,
            "GPU capex",
            [
                "JpSHyEIZ_bo",
                "dQw4w9WgXcQ",
            ],
            embed_text=fake_embed,
        )
        transcript_chunk_indices = connection.table("video_transcript_chunks").list_indices()
        chunk_row = connection.find_row("video_transcript_chunks", "video_id", "JpSHyEIZ_bo")
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "view_count": 60_000,
                "like_count": 3_000,
            },
        )
        updated_chunk_row = connection.find_row("video_transcript_chunks", "video_id", "JpSHyEIZ_bo")

    assert any(index.name == TRANSCRIPT_CHUNK_TEXT_INDEX for index in transcript_chunk_indices)
    assert chunk_row is not None
    assert chunk_row["title"] == "GPU topic"
    assert chunk_row["channel_id"] == "UC123"
    assert chunk_row["channel_handle"] == "@Chips"
    assert chunk_row["upload_date"] == "2026-07-03"
    assert chunk_row["duration"] == 600.0
    assert chunk_row["view_count"] == 40_000
    assert chunk_row["like_count"] == 2_000
    assert updated_chunk_row is not None
    assert updated_chunk_row["view_count"] == 60_000
    assert updated_chunk_row["like_count"] == 3_000
    assert scores["JpSHyEIZ_bo"] > 0
    assert scores["dQw4w9WgXcQ"] < scores["JpSHyEIZ_bo"]
    assert set(all_video_scores) == {
        "JpSHyEIZ_bo",
        "dQw4w9WgXcQ",
    }
    assert filtered_chunks
    assert filtered_chunks[0]["video_id"] == "JpSHyEIZ_bo"
    assert "similarity" in filtered_chunks[0]
    assert "title" not in filtered_chunks[0]
    assert blocked_chunks == []
    assert "LanceRead" in explain_plan
    assert "AnalyzeExec" in analyze_plan


def test_index_video_embedding_persists_after_connection_close(
    monkeypatch,
    tmp_path,
) -> None:
    storage_home = tmp_path / "state"
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(storage_home))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
            },
            transcript="晶片需求和 GPU 投资回升",
        )

    with get_connection() as connection:
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)

    with get_connection() as connection:
        row = connection.find_row("video_embeddings", "video_id", "JpSHyEIZ_bo")

    assert row is not None
    assert row["embedding_content_hash"] is not None
    assert row["embedding"] == pytest.approx(fake_embed("GPU capex"))


def test_auto_index_video_embedding_respects_config_toggle(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EMBEDDING_AUTO_INDEX", "false")

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
            },
            transcript="晶片需求和 GPU 投资回升",
        )

        assert auto_index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed) == {
            "status": "skipped",
            "video_id": "JpSHyEIZ_bo",
            "detail": "Auto embedding is disabled",
        }


def test_auto_index_transcript_chunks_respects_config_toggle(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EMBEDDING_AUTO_INDEX", "false")

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "GPU topic",
            },
            transcript="GPU earnings",
        )
        assert auto_index_transcript_chunks(connection, "JpSHyEIZ_bo", embed_text=fake_embed) == {
            "status": "skipped",
            "video_id": "JpSHyEIZ_bo",
            "detail": "Auto embedding is disabled",
        }


def test_auto_index_video_embedding_reports_failures(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EMBEDDING_AUTO_INDEX", "true")

    def fail_embed(_text: str):
        raise RuntimeError("embedding unavailable")

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Semiconductor cycle",
            },
            transcript="晶片需求和 GPU 投资回升",
        )

        assert auto_index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fail_embed) == {
            "status": "failed",
            "video_id": "JpSHyEIZ_bo",
            "error": "embedding unavailable",
        }


def test_auto_index_video_embedding_reports_invalid_toggle(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EMBEDDING_AUTO_INDEX", "maybe")

    with get_connection() as connection:
        assert auto_index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed) == {
            "status": "failed",
            "video_id": "JpSHyEIZ_bo",
            "error": "Model config value must be a boolean: EMBEDDING_AUTO_INDEX",
        }


def test_embed_text_with_gemini_uses_embedding_config(monkeypatch) -> None:
    calls = {}

    def fake_post_embedding_json(url, config, payload):
        calls["url"] = url
        calls["config"] = config
        calls["payload"] = payload
        return {"embedding": {"values": [0.25, 0.75]}}

    config = EmbeddingConfig(
        model="gemini-embedding-test",
        api_key="embedding-key",
        base_url="https://embedding.example/v1beta/",
        output_dimensionality=512,
    )
    monkeypatch.setattr("youtube_toolkit.storage.semantic.post_embedding_json", fake_post_embedding_json)

    assert embed_text_with_gemini("hello", config=config) == [0.25, 0.75]
    assert calls == {
        "url": "https://embedding.example/v1beta/models/gemini-embedding-test:embedContent",
        "config": config,
        "payload": {
            "model": "models/gemini-embedding-test",
            "content": {"parts": [{"text": "hello"}]},
            "output_dimensionality": 512,
        },
    }


def test_upsert_video_summary_stores_summary(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Market GPU note",
            },
            transcript="GPU earnings",
        )
        summary_video_id = upsert_video_summary(
            connection,
            "JpSHyEIZ_bo",
            "GPU earnings improved.",
        )
        row = connection.find_row("videos", "youtube_video_id", "JpSHyEIZ_bo")

    assert summary_video_id == "JpSHyEIZ_bo"
    assert row is not None
    assert row["summary"] == "GPU earnings improved."


def test_upsert_video_summary_updates_summary(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Market GPU note",
            },
            transcript="GPU earnings transcript",
        )
        upsert_video_summary(
            connection,
            "JpSHyEIZ_bo",
            "Old summary.",
        )
        summary_video_id = upsert_video_summary(
            connection,
            "JpSHyEIZ_bo",
            "New summary.",
        )
        row = connection.find_row("videos", "youtube_video_id", "JpSHyEIZ_bo")

    assert summary_video_id == "JpSHyEIZ_bo"
    assert row is not None
    assert row["summary"] == "New summary."


def test_video_summary_is_removed_when_transcript_changes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        metadata = {
            "id": "JpSHyEIZ_bo",
            "title": "Market GPU note",
        }
        upsert_video(
            connection,
            metadata,
            transcript="Old transcript",
        )
        upsert_video_summary(
            connection,
            "JpSHyEIZ_bo",
            "Old summary.",
        )
        upsert_video(
            connection,
            metadata,
            transcript="New transcript",
        )
        summary_row = connection.find_row("videos", "youtube_video_id", "JpSHyEIZ_bo")

    assert summary_row is not None
    assert summary_row["summary"] is None


def test_search_stored_videos_can_sort_by_relevance_score(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "Market GPU note",
            },
            transcript="GPU earnings",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Market kitchen note",
            },
            transcript="pasta sauce",
        )
    with get_connection() as connection:
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
    with get_connection() as connection:
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

    with get_connection() as connection:
        results = search_stored_videos(
            connection,
            "GPU capex",
            semantic=True,
            embed_text=fake_embed,
            sort="relevance",
        )

    assert results[0]["video_id"] == "JpSHyEIZ_bo"
    assert results[0]["semantic_score"] > results[1]["semantic_score"]


def test_search_stored_videos_keeps_upload_date_sort_with_semantic_expansion(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "GPU note",
                "upload_date": "20260618",
            },
            transcript="GPU earnings",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Market kitchen note",
                "upload_date": "20260620",
            },
            transcript="pasta sauce",
        )
    with get_connection() as connection:
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
    with get_connection() as connection:
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

    with get_connection() as connection:
        results = search_stored_videos(
            connection,
            "GPU capex",
            semantic=True,
            embed_text=fake_embed,
            sort="upload_date",
        )

    assert [result["video_id"] for result in results] == [
        "dQw4w9WgXcQ",
        "JpSHyEIZ_bo",
    ]
    assert results[1]["semantic_score"] > results[0]["semantic_score"]


def test_search_stored_videos_semantic_search_is_lightweight_without_index(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_video(
            connection,
            {
                "id": "JpSHyEIZ_bo",
                "title": "AI market notes",
            },
            transcript="逐字稿",
        )
        assert search_stored_videos(connection, "GPU capex", semantic=True, embed_text=failing_embed) == []


def fake_embed(text: str) -> list[float]:
    lowered_text = text.lower()
    if "gpu" in lowered_text or "semiconductor" in lowered_text or "晶片" in lowered_text:
        return [1.0, 0.0]
    return [0.0, 1.0]


def title_embed(text: str) -> list[float]:
    lowered_text = text.lower()
    if any(term in lowered_text for term in ("product", "introducing", "launch")):
        return [1.0, 0.0]
    return [0.0, 1.0]


def failing_embed(_text: str) -> list[float]:
    raise AssertionError("semantic search should not embed when no saved embeddings exist")
