import pytest

from youtube_toolkit import EmbeddingConfig
from youtube_toolkit.storage import (
    DATABASE_FILENAME,
    STORAGE_DIRNAME,
    embed_text_with_gemini,
    find_repo_root,
    get_connection,
    index_video_embedding,
    resolve_storage_dir,
    search_stored_videos,
    upsert_video,
    upsert_video_summary,
)


def test_default_storage_path_is_repo_cache(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STORAGE_HOME", raising=False)
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STATE_HOME", raising=False)
    monkeypatch.delenv("YOUTUBE_MEMORY_HOME", raising=False)

    assert STORAGE_DIRNAME == ".cache"
    assert DATABASE_FILENAME == "youtube.sqlite3"
    assert resolve_storage_dir() == repo_root / ".cache"


def test_find_repo_root_returns_nearest_git_parent(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert find_repo_root(nested) == repo_root


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
        row = connection.execute(
            """
            SELECT title
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()

    assert row is not None
    assert row["title"] == "AI商業發展暗藏機會"


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
                "view_count": 123,
            },
            transcript="既有逐字稿",
        )
        expected = {
            "video_id": "JpSHyEIZ_bo",
            "database_row_id": 1,
            "title": "AI market notes",
            "upload_date": "2026-06-21",
            "view_count": 123,
            "channel_id": None,
            "channel_handle": None,
            "transcript": "既有逐字稿",
        }
        assert search_stored_videos(connection, "JpSHyEIZ_bo") == [expected]
        assert search_stored_videos(connection, "https://www.youtube.com/watch?v=JpSHyEIZ_bo") == [expected]


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


def test_search_stored_videos_defaults_to_view_sort(
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
                "view_count": 10,
            },
            transcript="same keyword",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Market update",
                "upload_date": "20260618",
                "view_count": 500,
            },
            transcript="same keyword",
        )
        results = search_stored_videos(
            connection,
            "Market",
        )

    assert [result["video_id"] for result in results] == [
        "dQw4w9WgXcQ",
        "JpSHyEIZ_bo",
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
                "view_count": 500,
            },
            transcript="晶片需求和 GPU 投资回升",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Kitchen notes",
                "view_count": 10,
            },
            transcript="番茄酱和意大利面",
        )
        embedding_payload = index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
        stored_embedding = connection.execute(
            """
            SELECT embedding_content_hash, typeof(embedding) AS embedding_type
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()
        assert embedding_payload is not None
        assert embedding_payload.pop("content_hash")
        assert stored_embedding is not None
        assert stored_embedding["embedding_content_hash"]
        assert stored_embedding["embedding_type"] == "blob"
        assert embedding_payload == {
            "video_id": "JpSHyEIZ_bo",
            "database_row_id": 1,
            "dimensions": 2,
        }
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

        results = search_stored_videos(
            connection,
            "GPU capex",
            semantic=True,
            embed_text=fake_embed,
        )

    assert results[0]["video_id"] == "JpSHyEIZ_bo"
    assert results[0]["semantic_score"] == pytest.approx(1.0)


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
        row_id = upsert_video_summary(
            connection,
            "JpSHyEIZ_bo",
            "GPU earnings improved.",
        )
        row = connection.execute(
            """
            SELECT summary
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()

    assert row_id == 1
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
        row_id = upsert_video_summary(
            connection,
            "JpSHyEIZ_bo",
            "New summary.",
        )
        row = connection.execute(
            """
            SELECT summary
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()

    assert row_id == 1
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
        summary_row = connection.execute(
            """
            SELECT summary
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()

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
                "view_count": 10,
            },
            transcript="GPU earnings",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Market kitchen note",
                "view_count": 500,
            },
            transcript="pasta sauce",
        )
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

        results = search_stored_videos(
            connection,
            "GPU capex",
            semantic=True,
            embed_text=fake_embed,
            sort="relevance",
        )

    assert results[0]["video_id"] == "JpSHyEIZ_bo"
    assert results[0]["semantic_score"] > results[1]["semantic_score"]


def test_search_stored_videos_keeps_view_sort_with_semantic_expansion(
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
                "view_count": 10,
            },
            transcript="GPU earnings",
        )
        upsert_video(
            connection,
            {
                "id": "dQw4w9WgXcQ",
                "title": "Market kitchen note",
                "view_count": 500,
            },
            transcript="pasta sauce",
        )
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

        results = search_stored_videos(
            connection,
            "GPU capex",
            semantic=True,
            embed_text=fake_embed,
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


def failing_embed(_text: str) -> list[float]:
    raise AssertionError("semantic search should not embed when no saved embeddings exist")
