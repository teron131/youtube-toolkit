from pathlib import Path

import pytest

import youtube_toolkit.storage.connection as storage_connection
from youtube_toolkit import EmbeddingConfig
from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.storage import (
    DATABASE_FILENAME,
    STORAGE_DIRNAME,
    embed_text_with_gemini,
    get_connection,
    get_stored_video,
    index_video_embedding,
    resolve_storage_dir,
    search_stored_videos,
    upsert_watchlist_channel,
    upsert_video,
    upsert_video_summary,
)


def test_default_storage_path_is_repo_cache(monkeypatch, tmp_path) -> None:
    toolkit_root = Path(__file__).resolve().parents[1]
    outside_repo = tmp_path / "other"
    outside_repo.mkdir()
    (outside_repo / ".git").mkdir()
    monkeypatch.chdir(outside_repo)
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STORAGE_HOME", raising=False)

    assert STORAGE_DIRNAME == ".cache"
    assert DATABASE_FILENAME == "youtube.sqlite3"
    assert resolve_storage_dir() == toolkit_root / ".cache"


def test_installed_storage_path_is_user_app_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(storage_connection, "__file__", str(installed_connection_file(tmp_path)))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("YOUTUBE_TOOLKIT_STORAGE_HOME", raising=False)

    assert resolve_storage_dir() == tmp_path / "home" / "Library" / "Application Support" / "youtube-toolkit"


def installed_connection_file(tmp_path: Path) -> Path:
    return tmp_path / "tool" / "site-packages" / "youtube_toolkit" / "storage" / "connection.py"


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

    assert payload == {
        "video_id": "JpSHyEIZ_bo",
        "title": "AI market notes",
        "upload_date": "2026-06-21",
        "channel_id": None,
        "channel_handle": None,
        "transcript": "既有逐字稿",
    }


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
        row = connection.execute(
            """
            SELECT channel_id
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()

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
        summary_video_id = upsert_video_summary(
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
        row = connection.execute(
            """
            SELECT summary
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("JpSHyEIZ_bo",),
        ).fetchone()

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
        index_video_embedding(connection, "JpSHyEIZ_bo", embed_text=fake_embed)
        index_video_embedding(connection, "dQw4w9WgXcQ", embed_text=fake_embed)

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


def failing_embed(_text: str) -> list[float]:
    raise AssertionError("semantic search should not embed when no saved embeddings exist")
