"""Search contracts exercise discovery, filtering, and ephemeral semantic ranking."""

from collections.abc import Callable
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

import pytest

from youtube_toolkit.config import EmbeddingConfig
from youtube_toolkit.schemas import TranscriptChunk, TranscriptDocument, VideoMetadata
import youtube_toolkit.search.discovery as search_discovery
from youtube_toolkit.search.embeddings import EmbeddingClient
from youtube_toolkit.search.preprocessing import VideoResult
from youtube_toolkit.search.request import (
    ChannelTarget,
    SearchRequest,
    parse_duration,
)
import youtube_toolkit.search.semantic as search_semantic
import youtube_toolkit.search.workflow as search_workflow
from youtube_toolkit.search.workflow import search_videos


def use_embeddings(monkeypatch, embed: Callable[[str], list[float]]) -> None:
    """Replace only the embedding transport, retaining production batch execution and failure handling."""

    def create(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=embed(kwargs["input"]))])

    monkeypatch.setattr(
        "youtube_toolkit.search.embeddings.OpenAI",
        lambda **_kwargs: SimpleNamespace(
            embeddings=SimpleNamespace(create=create), close=lambda: None
        ),
    )
    config = EmbeddingConfig("test", "embedding-key", "https://embedding.example/v1")
    monkeypatch.setattr(
        search_semantic, "EmbeddingClient", lambda: EmbeddingClient(config)
    )


@pytest.fixture(autouse=True)
def search_transcript_side_effects(monkeypatch) -> dict[str, Any]:
    fetched_video_ids: list[str] = []
    supplied_metadata: dict[str, dict | None] = {}

    def fake_fetch_live_transcript(
        video_id: str,
        *,
        options=None,
        metadata: dict | None = None,
    ) -> TranscriptDocument:
        assert options is not None
        assert not options.allow_stt
        fetched_video_ids.append(video_id)
        supplied_metadata[video_id] = metadata
        return TranscriptDocument(
            metadata=VideoMetadata(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
            ),
            language="en",
            source="automatic_captions",
            chunks=[
                TranscriptChunk(
                    start_seconds=0,
                    end_seconds=1,
                    text=f"Transcript for {video_id}",
                )
            ],
        )

    monkeypatch.setattr(search_semantic, "fetch_transcript", fake_fetch_live_transcript)
    use_embeddings(monkeypatch, lambda _text: [1.0, 0.0])
    return {
        "fetched": fetched_video_ids,
        "metadata": supplied_metadata,
    }


def video_results(request: SearchRequest) -> list[VideoResult]:
    """Run a search the test knows returns videos, so video-only fields type-check."""
    return cast(list[VideoResult], search_videos(request))


def test_parse_duration_accepts_seconds_and_colon_values() -> None:
    assert parse_duration("90") == 90
    assert parse_duration("1:30") == 90
    assert parse_duration("1:02:03") == 3723


def test_parse_duration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration("1::30")


def test_native_youtube_search_does_not_fetch_transcripts(
    monkeypatch,
    search_transcript_side_effects,
) -> None:
    calls = {}

    def fake_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        calls["query"] = query
        calls["fetch_limit"] = fetch_limit
        calls["sort"] = sort
        return [
            {
                "id": "aaaaaaaaaaa",
                "channel": "OpenAI",
                "view_count": 6_000,
                "duration": 600,
            },
            {
                "id": "bbbbbbbbbbb",
                "channel": "OpenAI",
                "view_count": 7_000,
                "duration": 600,
            },
            {
                "id": "ccccccccccc",
                "channel": "OpenAI",
                "view_count": 8_000,
                "duration": 600,
            },
            {
                "id": "ddddddddddd",
                "channel": "OpenAI",
                "view_count": 9_000,
                "duration": 600,
            },
        ]

    monkeypatch.setattr(search_discovery, "search_youtube", fake_search_youtube)

    assert search_videos(
        SearchRequest(
            queries=("agent skills",),
            limit=3,
            min_views=5000,
            include_channel=("@OpenAI",),
        )
    ) == [
        {
            "type": "video",
            "video_id": "aaaaaaaaaaa",
            "title": None,
            "channel_id": None,
            "channel_handle": None,
            "upload_date": None,
            "duration": 600,
            "view_count": 6_000,
            "like_count": None,
            "source": "youtube",
        },
        {
            "type": "video",
            "video_id": "bbbbbbbbbbb",
            "title": None,
            "channel_id": None,
            "channel_handle": None,
            "upload_date": None,
            "duration": 600,
            "view_count": 7_000,
            "like_count": None,
            "source": "youtube",
        },
        {
            "type": "video",
            "video_id": "ccccccccccc",
            "title": None,
            "channel_id": None,
            "channel_handle": None,
            "upload_date": None,
            "duration": 600,
            "view_count": 8_000,
            "like_count": None,
            "source": "youtube",
        },
    ]
    assert calls["query"] == "agent skills"
    assert calls["fetch_limit"] == 9
    assert calls["sort"] == "relevance"
    assert search_transcript_side_effects["fetched"] == []


def test_default_filters_run_before_youtube_date_metadata_fetch(
    monkeypatch,
    search_transcript_side_effects,
) -> None:
    def fake_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        assert fetch_limit == 60
        assert sort == "relevance"
        return [
            {
                "id": "aaaaaaaaaaa",
                "title": "Too small",
                "view_count": 500,
                "duration": 600,
            },
            {
                "id": "bbbbbbbbbbb",
                "title": "New enough",
                "view_count": 5_000,
                "duration": 600,
            },
        ]

    metadata_calls: list[str] = []

    class FakeYtDlpClient:
        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            return {
                "id": video_id,
                "title": "New enough",
                "upload_date": "20260702",
                "view_count": 5_000,
                "duration": 600,
            }

    monkeypatch.setattr(search_discovery, "search_youtube", fake_search_youtube)
    monkeypatch.setattr(search_workflow, "YtDlpClient", FakeYtDlpClient)

    results = video_results(
        SearchRequest(
            queries=("Claude Fable 5",),
            limit=5,
            min_views=1_000,
            uploaded_after="2026-07-01",
        )
    )

    assert metadata_calls == ["bbbbbbbbbbb"]
    assert [result["video_id"] for result in results] == ["bbbbbbbbbbb"]
    assert results[0]["upload_date"] == "2026-07-02"
    assert search_transcript_side_effects["fetched"] == []


def test_youtube_relevance_search_interlaces_queries(
    monkeypatch,
    search_transcript_side_effects,
) -> None:
    calls = []
    ready = Barrier(3)
    results_by_query = {
        "alpha": [
            {"id": "aaaaaaaaaaa", "title": "A1"},
            {"id": "ddddddddddd", "title": "A2"},
        ],
        "beta": [
            {"id": "bbbbbbbbbbb", "title": "B1"},
            {"id": "aaaaaaaaaaa", "title": "Duplicate ignored"},
            {"id": "eeeeeeeeeee", "title": "B3"},
        ],
        "gamma": [
            {"id": "ccccccccccc", "title": "C1"},
        ],
    }

    def fake_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        calls.append((query, fetch_limit, sort))
        ready.wait(timeout=2)
        return results_by_query[query]

    monkeypatch.setattr(search_discovery, "search_youtube", fake_search_youtube)

    results = video_results(
        SearchRequest(
            queries=("alpha", "beta", "gamma", "alpha"),
            limit=5,
            min_views=None,
            min_duration=None,
            max_duration=None,
        )
    )

    assert [result["video_id"] for result in results] == [
        "aaaaaaaaaaa",
        "bbbbbbbbbbb",
        "ccccccccccc",
        "ddddddddddd",
        "eeeeeeeeeee",
    ]
    assert [result["title"] for result in results] == [
        "A1",
        "B1",
        "C1",
        "A2",
        "B3",
    ]
    assert sorted(calls) == [
        ("alpha", 5, "relevance"),
        ("beta", 5, "relevance"),
        ("gamma", 5, "relevance"),
    ]
    assert search_transcript_side_effects["fetched"] == []


def test_youtube_popularity_search_dedupes_and_sorts_by_popularity(
    monkeypatch,
    search_transcript_side_effects,
) -> None:
    results_by_query = {
        "alpha": [
            {"id": "aaaaaaaaaaa", "view_count": 10},
            {"id": "bbbbbbbbbbb", "view_count": 100},
        ],
        "beta": [
            {"id": "ccccccccccc", "view_count": 50},
            {"id": "aaaaaaaaaaa", "view_count": 999},
        ],
    }

    def fake_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        assert fetch_limit == 5
        assert sort == "popularity"
        return results_by_query[query]

    monkeypatch.setattr(search_discovery, "search_youtube", fake_search_youtube)

    results = search_videos(
        SearchRequest(
            queries=("alpha", "beta"),
            limit=3,
            sort="popularity",
            min_views=None,
            min_duration=None,
            max_duration=None,
        )
    )

    assert [result["video_id"] for result in results] == [
        "bbbbbbbbbbb",
        "ccccccccccc",
        "aaaaaaaaaaa",
    ]
    assert search_transcript_side_effects["fetched"] == []


def test_channel_search_selection_accepts_channel_id_and_handle() -> None:
    assert ChannelTarget.from_ref("UC123") == ChannelTarget(
        youtube_channel_id="UC123",
        channel_handle=None,
    )
    assert ChannelTarget.from_ref("OpenAI") == ChannelTarget(
        youtube_channel_id=None,
        channel_handle=None,
        channel_name="OpenAI",
    )
    assert ChannelTarget.from_ref("https://www.youtube.com/@OpenAI") == ChannelTarget(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )


def test_channel_search_fetches_channel_listing(
    monkeypatch,
) -> None:
    calls = []

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        calls.append((channel_ref, options.last))
        return {
            "channel_id": "UC123",
            "channel_handle": "@OpenAI",
            "entries": [
                {
                    "id": "JpSHyEIZ_bo",
                    "title": "OpenAI agents",
                    "upload_date": "20260706",
                    "view_count": 25_000,
                    "duration": 600,
                }
            ],
        }

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    results = search_videos(
        SearchRequest(
            queries=(),
            limit=30,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert calls == [("@OpenAI", 30)]
    assert results == [
        {
            "type": "video",
            "video_id": "JpSHyEIZ_bo",
            "title": "OpenAI agents",
            "channel_id": "UC123",
            "channel_handle": "@OpenAI",
            "upload_date": "20260706",
            "duration": 600,
            "view_count": 25_000,
            "like_count": None,
            "source": "channel",
        }
    ]


def test_channel_search_resolves_names_before_fetching_listings(monkeypatch) -> None:
    search_queries = []
    listing_calls = []

    def fake_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        search_queries.append((query, fetch_limit, sort))
        return [
            {
                "id": "JpSHyEIZ_bo",
                "title": "Introducing Claude",
                "channel": "Anthropic",
                "channel_id": "UCrDwWp7EBBv4NwvScIpBDOA",
                "uploader_id": "@anthropic-ai",
            }
        ]

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        listing_calls.append((channel_ref, options.last))
        return {
            "channel_id": "UCrDwWp7EBBv4NwvScIpBDOA",
            "entries": [
                {
                    "id": "JpSHyEIZ_bo",
                    "title": "Introducing Claude",
                    "upload_date": "20260706",
                    "view_count": 25_000,
                    "duration": 600,
                }
            ],
        }

    monkeypatch.setattr(search_discovery, "search_youtube", fake_search_youtube)
    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    results = video_results(
        SearchRequest(
            queries=(),
            limit=5,
            channels=(ChannelTarget.from_ref("Anthropic"),),
        )
    )

    assert search_queries == [("Anthropic", 10, "relevance")]
    assert listing_calls == [("UCrDwWp7EBBv4NwvScIpBDOA", 5)]
    assert results[0]["channel_handle"] == "@anthropic-ai"


def test_channel_search_uses_explicit_handles_without_name_lookup(monkeypatch) -> None:
    listing_calls = []

    def fail_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        raise AssertionError("explicit handles should not search channel names")

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        listing_calls.append((channel_ref, options.last))
        return {
            "channel_id": "UC123",
            "channel_handle": "@Anthropic",
            "entries": [
                {
                    "id": "JpSHyEIZ_bo",
                    "title": "Exact handle result",
                    "upload_date": "20260706",
                    "view_count": 25_000,
                    "duration": 600,
                }
            ],
        }

    monkeypatch.setattr(search_discovery, "search_youtube", fail_search_youtube)
    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    search_videos(
        SearchRequest(
            queries=(),
            limit=5,
            channels=(ChannelTarget.from_ref("@Anthropic"),),
        )
    )

    assert listing_calls == [("@Anthropic", 5)]


def test_channel_search_dedupes_and_sorts_by_recency(monkeypatch) -> None:
    listings = {
        "@OpenAI": {
            "channel_id": "UC123",
            "channel_handle": "@OpenAI",
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "First source wins",
                    "upload_date": "20260702",
                    "view_count": 25_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Missing date",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "not-a-video-id",
                    "title": "Ignored",
                },
            ],
        },
        "@Anthropic": {
            "channel_id": "UC456",
            "channel_handle": "@Anthropic",
            "entries": [
                {
                    "id": "ccccccccccc",
                    "title": "Newest",
                    "upload_date": "20260703",
                    "view_count": 35_000,
                    "duration": 600,
                },
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Duplicate ignored",
                    "upload_date": "20260704",
                    "view_count": 40_000,
                    "duration": 600,
                },
            ],
        },
    }

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return listings[channel_ref]

    metadata_calls: list[str] = []

    class UnavailableYtDlpClient:
        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_workflow, "YtDlpClient", UnavailableYtDlpClient)

    results = video_results(
        SearchRequest(
            queries=(),
            limit=30,
            channels=(
                ChannelTarget.from_ref("@OpenAI"),
                ChannelTarget.from_ref("@Anthropic"),
            ),
        )
    )

    assert [result["video_id"] for result in results] == [
        "ccccccccccc",
        "aaaaaaaaaaa",
        "bbbbbbbbbbb",
    ]
    assert results[1]["title"] == "First source wins"
    assert metadata_calls == ["bbbbbbbbbbb"]


def test_channel_search_can_sort_by_popularity(monkeypatch) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Low popularity",
                    "upload_date": "20260703",
                    "view_count": 10,
                    "like_count": 10,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "High popularity",
                    "upload_date": "20260701",
                    "view_count": 100,
                    "like_count": 10,
                    "duration": 600,
                },
                {
                    "id": "ddddddddddd",
                    "title": "Higher likes at same views",
                    "upload_date": "20260701",
                    "view_count": 100,
                    "like_count": 20,
                    "duration": 600,
                },
                {
                    "id": "ccccccccccc",
                    "title": "Missing popularity",
                    "upload_date": "20260702",
                },
            ],
        }

    metadata_calls: list[str] = []

    class UnavailableYtDlpClient:
        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_workflow, "YtDlpClient", UnavailableYtDlpClient)

    results = search_videos(
        SearchRequest(
            queries=(),
            limit=30,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            sort="popularity",
            min_views=None,
            min_duration=None,
            max_duration=None,
        )
    )

    assert [result["video_id"] for result in results] == [
        "ddddddddddd",
        "bbbbbbbbbbb",
        "aaaaaaaaaaa",
        "ccccccccccc",
    ]
    assert metadata_calls == ["ccccccccccc"]


def test_channel_search_fetches_metadata_missing_from_flat_listings(
    monkeypatch,
) -> None:
    """Flat channel listings lack views and dates, so known filters run first and survivors are hydrated."""

    def fake_fetch_channel_videos(channel_ref: str, *, options) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {"id": "aaaaaaaaaaa", "title": "Older talk", "duration": 600},
                {"id": "bbbbbbbbbbb", "title": "Short clip", "duration": 45},
                {"id": "ccccccccccc", "title": "Unpopular talk", "duration": 600},
                {"id": "ddddddddddd", "title": "Newer talk", "duration": 600},
            ],
        }

    full_metadata = {
        "aaaaaaaaaaa": {"upload_date": "20260701", "view_count": 20_000},
        "ccccccccccc": {"upload_date": "20260703", "view_count": 50},
        "ddddddddddd": {"upload_date": "20260702", "view_count": 30_000},
    }
    metadata_calls: list[str] = []

    class FakeYtDlpClient:
        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            return {"id": video_id, "duration": 600, **full_metadata[video_id]}

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_workflow, "YtDlpClient", FakeYtDlpClient)

    results = video_results(
        SearchRequest(
            queries=(),
            limit=30,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            min_views=10_000,
            min_duration="3:00",
            max_duration="8:00:00",
        )
    )

    assert sorted(metadata_calls) == ["aaaaaaaaaaa", "ccccccccccc", "ddddddddddd"]
    assert [result["video_id"] for result in results] == [
        "ddddddddddd",
        "aaaaaaaaaaa",
    ]
    assert results[0]["upload_date"] == "2026-07-02"
    assert results[0]["view_count"] == 30_000


def test_channel_date_filter_stops_hydrating_after_listing_passes_cutoff(
    monkeypatch,
) -> None:
    """Newest-first channel listings stop fetching metadata after the batch that reaches an older video."""
    video_ids = [f"video{index:06d}" for index in range(20)]

    def fake_fetch_channel_videos(channel_ref: str, *, options) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {"id": video_id, "title": f"Talk {index}", "duration": 600}
                for index, video_id in enumerate(video_ids)
            ],
        }

    metadata_calls: list[str] = []

    class FakeYtDlpClient:
        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            day = 20 - video_ids.index(video_id)
            return {"id": video_id, "upload_date": f"202607{day:02d}", "duration": 600}

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_workflow, "YtDlpClient", FakeYtDlpClient)

    results = search_videos(
        SearchRequest(
            queries=(),
            limit=2,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            uploaded_after="2026-07-10",
        )
    )

    batch_size = search_workflow.MAX_CONCURRENT_METADATA_FETCHES
    assert sorted(metadata_calls) == video_ids[: 2 * batch_size]
    assert [result["video_id"] for result in results] == video_ids[:2]


def test_native_channel_query_filters_video_titles(monkeypatch) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "GPU capex discussion",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "General agent update",
                    "upload_date": "20260702",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        }

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    results = search_videos(
        SearchRequest(
            queries=("capex",),
            limit=30,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert [result["video_id"] for result in results] == ["aaaaaaaaaaa"]


def test_channel_search_can_sort_by_semantic_relevance(
    monkeypatch,
    search_transcript_side_effects,
) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Newest but weak match",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Best match",
                    "upload_date": "20260701",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "ccccccccccc",
                    "title": "Second best match",
                    "upload_date": "20260702",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        }

    def fake_embed_text(text: str) -> list[float]:
        scores = {
            "Transcript for aaaaaaaaaaa": 0.2,
            "Transcript for bbbbbbbbbbb": 0.9,
            "Transcript for ccccccccccc": 0.8,
        }
        score = scores.get(text, 1.0)
        return [score, (1 - score**2) ** 0.5]

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    use_embeddings(monkeypatch, fake_embed_text)

    results = search_videos(
        SearchRequest(
            queries=("GPU capex",),
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            limit=2,
        )
    )

    assert [result["video_id"] for result in results] == [
        "bbbbbbbbbbb",
        "ccccccccccc",
    ]
    assert [result["score"] for result in results] == pytest.approx([0.9, 0.8])
    assert sorted(search_transcript_side_effects["fetched"]) == [
        "aaaaaaaaaaa",
        "bbbbbbbbbbb",
        "ccccccccccc",
    ]


def test_semantic_channel_search_prefilters_by_title_embeddings(
    monkeypatch,
    search_transcript_side_effects,
) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "AI interview with a founder",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Introducing Claude Fable 5",
                    "view_count": 30_000,
                    "duration": 120,
                },
                {
                    "id": "ccccccccccc",
                    "title": "Research update",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "ddddddddddd",
                    "title": "Launch day for a new product",
                    "view_count": 30_000,
                    "duration": 180,
                },
                {
                    "id": "eeeeeeeeeee",
                    "title": "Office hours",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "ffffffffff0",
                    "title": "Meet the new assistant",
                    "view_count": 30_000,
                    "duration": 240,
                },
            ],
        }

    def fake_embed_text(text: str) -> list[float]:
        scores = {
            "AI interview with a founder": 0.1,
            "Introducing Claude Fable 5": 0.95,
            "Research update": 0.1,
            "Launch day for a new product": 0.9,
            "Office hours": 0.1,
            "Meet the new assistant": 0.8,
            "Transcript for bbbbbbbbbbb": 0.8,
            "Transcript for ddddddddddd": 0.9,
            "Transcript for ffffffffff0": 0.7,
        }
        score = scores.get(text, 1.0)
        return [score, (1 - score**2) ** 0.5]

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    use_embeddings(monkeypatch, fake_embed_text)

    results = search_videos(
        SearchRequest(
            queries=("product announcements",),
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@Anthropic"),),
            limit=2,
        )
    )

    assert sorted(search_transcript_side_effects["fetched"]) == [
        "bbbbbbbbbbb",
        "ddddddddddd",
        "ffffffffff0",
    ]
    assert [result["video_id"] for result in results] == [
        "ddddddddddd",
        "bbbbbbbbbbb",
    ]


def test_semantic_titles_share_embeddings_and_keep_per_video_warnings(
    monkeypatch,
) -> None:
    """Text-keyed batch results still rank every matching video and identify every failed video."""
    entries = [
        {"id": "aaaaaaaaaaa", "title": "Unavailable title"},
        {"id": "bbbbbbbbbbb", "title": "Unavailable title"},
        {"id": "ccccccccccc", "title": "Shared title"},
        {"id": "ddddddddddd", "title": "Shared title"},
    ]
    monkeypatch.setattr(
        search_discovery,
        "fetch_channel_videos",
        lambda _ref, **_kwargs: {"entries": entries},
    )
    embedded = []

    def embed(text):
        embedded.append(text)
        if text == "Unavailable title":
            raise RuntimeError("embedding unavailable")
        return [1.0, 0.0]

    use_embeddings(monkeypatch, embed)
    results = search_videos(
        SearchRequest(
            queries=("query",),
            limit=5,
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert [result["video_id"] for result in results] == ["ccccccccccc", "ddddddddddd"]
    assert embedded.count("Unavailable title") == 1
    assert embedded.count("Shared title") == 1
    assert all(
        result["warnings"]
        == [
            "title embedding skipped 2 item(s): aaaaaaaaaaa: embedding unavailable; bbbbbbbbbbb: embedding unavailable"
        ]
        for result in results
    )


def test_semantic_video_search_returns_only_scored_videos(
    monkeypatch,
    search_transcript_side_effects,
) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Scored match",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Unscored metadata",
                    "upload_date": "20260702",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        }

    def fake_embed_text(text: str) -> list[float]:
        scores = {
            "Transcript for aaaaaaaaaaa": 0.8,
            "Transcript for bbbbbbbbbbb": 0.4,
        }
        score = scores.get(text, 1.0)
        return [score, (1 - score**2) ** 0.5]

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    use_embeddings(monkeypatch, fake_embed_text)

    results = search_videos(
        SearchRequest(
            queries=("GPU capex",),
            limit=10,
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert [result["video_id"] for result in results] == ["aaaaaaaaaaa"]


def test_date_filter_runs_before_transcript_fetch(
    monkeypatch,
    search_transcript_side_effects,
) -> None:
    """Flat listings are hydrated for dates once, and passing videos reuse that metadata for transcripts."""

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Old match",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "New match",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        }

    full_metadata = {
        "aaaaaaaaaaa": {"id": "aaaaaaaaaaa", "upload_date": "20260629"},
        "bbbbbbbbbbb": {"id": "bbbbbbbbbbb", "upload_date": "20260702"},
    }
    metadata_calls: list[str] = []

    class FakeYtDlpClient:
        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            return full_metadata[video_id]

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_workflow, "YtDlpClient", FakeYtDlpClient)

    results = search_videos(
        SearchRequest(
            queries=("GPU capex",),
            limit=10,
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            uploaded_after="2026-07-01",
        )
    )

    assert [result["video_id"] for result in results] == ["bbbbbbbbbbb"]
    assert search_transcript_side_effects["fetched"] == ["bbbbbbbbbbb"]
    assert sorted(metadata_calls) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    assert search_transcript_side_effects["metadata"] == {
        "bbbbbbbbbbb": full_metadata["bbbbbbbbbbb"]
    }


def test_semantic_video_results_include_partial_transcript_warnings(
    monkeypatch,
    search_transcript_side_effects,
) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Indexed match",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Skipped match",
                    "upload_date": "20260702",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        }

    def fake_fetch_live_transcript(video_id: str) -> TranscriptDocument:
        if video_id == "bbbbbbbbbbb":
            raise RuntimeError("captions unavailable")
        return TranscriptDocument(
            metadata=VideoMetadata(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
            ),
            language="en",
            source="automatic_captions",
            chunks=[TranscriptChunk(start_seconds=0, end_seconds=1, text="GPU capex")],
        )

    def fake_fetch_transcript(
        video_id: str,
        *,
        options=None,
        metadata: dict | None = None,
    ) -> TranscriptDocument:
        return fake_fetch_live_transcript(video_id)

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_semantic, "fetch_transcript", fake_fetch_transcript)

    results = search_videos(
        SearchRequest(
            queries=("GPU capex",),
            limit=10,
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert [result["video_id"] for result in results] == ["aaaaaaaaaaa"]
    assert results[0]["warnings"] == [
        "transcript preparation skipped 1 item(s): bbbbbbbbbbb: captions unavailable"
    ]


def test_semantic_search_defaults_to_chunk_results(
    monkeypatch,
    search_transcript_side_effects,
) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "GPU capex discussion",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                }
            ],
        }

    def fake_fetch_live_transcript(video_id: str) -> TranscriptDocument:
        search_transcript_side_effects["fetched"].append(video_id)
        return TranscriptDocument(
            metadata=VideoMetadata(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
            ),
            language="en",
            source="automatic_captions",
            chunks=[
                TranscriptChunk(
                    start_seconds=0,
                    end_seconds=1,
                    text="The GPU capex cycle is accelerating.",
                )
            ],
        )

    def fake_embed_text(text: str) -> list[float]:
        score = 0.92 if text == "The GPU capex cycle is accelerating." else 1.0
        return [score, (1 - score**2) ** 0.5]

    def fake_fetch_transcript(
        video_id: str,
        *,
        options=None,
        metadata: dict | None = None,
    ) -> TranscriptDocument:
        return fake_fetch_live_transcript(video_id)

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_semantic, "fetch_transcript", fake_fetch_transcript)
    use_embeddings(monkeypatch, fake_embed_text)

    results = search_videos(
        SearchRequest(
            queries=("GPU capex",),
            mode="semantic",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            limit=2,
        )
    )

    assert results == [
        {
            "type": "chunk",
            "video_id": "aaaaaaaaaaa",
            "chunk_index": 0,
            "text": "The GPU capex cycle is accelerating.",
            "score": 0.92,
            "video": {
                "title": "GPU capex discussion",
                "channel_id": "UC123",
                "channel_handle": "@OpenAI",
                "upload_date": "2026-07-03",
                "duration": 600.0,
                "view_count": 30_000,
                "like_count": None,
            },
        }
    ]
    assert search_transcript_side_effects["fetched"] == ["aaaaaaaaaaa"]


def test_live_semantic_search_fetches_transcripts_concurrently(
    monkeypatch,
) -> None:
    all_fetches_started = Barrier(4)

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {"id": "aaaaaaaaaaa", "title": "GPT-5.6 analysis"},
                {"id": "bbbbbbbbbbb", "title": "Grok 4.5 analysis"},
                {"id": "ccccccccccc", "title": "Fable 5 analysis"},
                {"id": "ddddddddddd", "title": "Gemini analysis"},
            ],
        }

    def fake_fetch_live_transcript(
        video_id: str, *, options=None, metadata: dict | None = None
    ) -> TranscriptDocument:
        all_fetches_started.wait(timeout=1)
        return TranscriptDocument(
            metadata=VideoMetadata(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
            ),
            language="en",
            source="automatic_captions",
            chunks=[TranscriptChunk(start_seconds=0, end_seconds=1, text=video_id)],
        )

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_semantic, "fetch_transcript", fake_fetch_live_transcript)

    results = search_videos(
        SearchRequest(
            queries=("frontier model analysis",),
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
            limit=4,
            min_views=None,
            min_duration=None,
            max_duration=None,
        )
    )

    assert {result["video_id"] for result in results} == {
        "aaaaaaaaaaa",
        "bbbbbbbbbbb",
        "ccccccccccc",
        "ddddddddddd",
    }


def test_chunk_results_require_semantic_mode() -> None:
    with pytest.raises(ValueError, match="chunk return requires semantic mode"):
        search_videos(
            SearchRequest(
                queries=("GPU capex",),
                limit=10,
                return_type="chunk",
            )
        )


def test_search_query_is_optional_only_for_native_channel_search(monkeypatch) -> None:
    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [],
        }

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    search_videos(
        SearchRequest(
            queries=(),
            limit=10,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )
    with pytest.raises(ValueError, match="search query must be non-empty"):
        search_videos(
            SearchRequest(
                queries=(),
                limit=10,
            )
        )
    with pytest.raises(ValueError, match="search query must be non-empty"):
        search_videos(
            SearchRequest(
                queries=(),
                limit=10,
                mode="semantic",
                channels=(ChannelTarget.from_ref("@OpenAI"),),
            )
        )


def test_native_channel_search_keeps_titles_matching_any_query(monkeypatch) -> None:
    """Queries are equals, so a channel video passes when its title contains any of them."""

    def fake_fetch_channel_videos(channel_ref: str, *, options) -> dict:
        return {
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "GPU capex discussion",
                    "upload_date": "20260703",
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "General agent update",
                    "upload_date": "20260702",
                },
                {
                    "id": "ccccccccccc",
                    "title": "Office hours",
                    "upload_date": "20260701",
                },
            ]
        }

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    results = search_videos(
        SearchRequest(
            queries=("capex", "agent"),
            limit=10,
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert [result["video_id"] for result in results] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]


@pytest.mark.parametrize("queries", [("alpha", "beta"), ("alpha", "beta", "alpha")])
def test_semantic_search_scores_passages_by_their_closest_query(
    monkeypatch, queries
) -> None:
    """A passage matching only a later query still ranks, because no query outranks another."""

    def fake_fetch_channel_videos(channel_ref: str, *, options) -> dict:
        return {"entries": [{"id": "aaaaaaaaaaa", "title": "Talk"}]}

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    embedded = []

    def embed(text):
        embedded.append(text)
        return [1.0, 0.0] if text == "alpha" else [0.0, 1.0]

    use_embeddings(monkeypatch, embed)

    results = search_videos(
        SearchRequest(
            queries=queries,
            limit=5,
            mode="semantic",
            return_type="video",
            channels=(ChannelTarget.from_ref("@OpenAI"),),
        )
    )

    assert [result["video_id"] for result in results] == ["aaaaaaaaaaa"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert embedded.count("alpha") == 1
    assert embedded.count("beta") == 1


def test_native_youtube_search_rejects_recency() -> None:
    with pytest.raises(ValueError, match="does not support --sort recency"):
        search_videos(
            SearchRequest(
                queries=("agents",),
                limit=10,
                sort="recency",
            )
        )


@pytest.mark.parametrize(
    ("search_request", "message"),
    [
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                min_views=-1,
            ),
            "--min-views cannot be negative",
        ),
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                min_duration="invalid",
            ),
            "Invalid duration",
        ),
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                max_duration="invalid",
            ),
            "Invalid duration",
        ),
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                min_duration="2:00",
                max_duration="1:00",
            ),
            "--min-duration cannot be greater than --max-duration",
        ),
    ],
)
def test_search_request_rejects_invalid_metadata_filters(
    search_request: SearchRequest,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        search_videos(search_request)


@pytest.mark.parametrize(
    ("search_request", "message"),
    [
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                mode=cast(Any, "adjacent"),
            ),
            "search mode must be one of",
        ),
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                return_type=cast(Any, "document"),
            ),
            "search return type must be one of",
        ),
        (
            SearchRequest(
                queries=("agents",),
                limit=10,
                sort=cast(Any, "random"),
            ),
            "search sort must be one of",
        ),
    ],
)
def test_search_request_rejects_invalid_literal_values(
    search_request: SearchRequest,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        search_videos(search_request)


def test_semantic_search_reports_when_no_transcripts_can_be_prepared(
    monkeypatch,
) -> None:

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        return {
            "channel_id": "UC123",
            "channel_handle": channel_ref,
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "GPU capex discussion",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                }
            ],
        }

    def fail_fetch_transcript(
        video_id: str,
        *,
        options=None,
        metadata: dict | None = None,
    ) -> TranscriptDocument:
        raise RuntimeError("captions unavailable")

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )
    monkeypatch.setattr(search_semantic, "fetch_transcript", fail_fetch_transcript)

    with pytest.raises(RuntimeError, match="captions unavailable"):
        search_videos(
            SearchRequest(
                queries=("GPU capex",),
                limit=10,
                mode="semantic",
                channels=(ChannelTarget.from_ref("@OpenAI"),),
            )
        )


def test_channel_search_limits_after_merge_and_sort(monkeypatch) -> None:
    calls = []
    ready = Barrier(2)
    listings = {
        "@OpenAI": {
            "channel_id": "UC123",
            "channel_handle": "@OpenAI",
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Older",
                    "upload_date": "20260701",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Newest",
                    "upload_date": "20260704",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        },
        "@Anthropic": {
            "channel_id": "UC456",
            "channel_handle": "@Anthropic",
            "entries": [
                {
                    "id": "ccccccccccc",
                    "title": "Second newest",
                    "upload_date": "20260703",
                    "view_count": 30_000,
                    "duration": 600,
                },
                {
                    "id": "ddddddddddd",
                    "title": "Third newest",
                    "upload_date": "20260702",
                    "view_count": 30_000,
                    "duration": 600,
                },
            ],
        },
    }

    def fake_fetch_channel_videos(
        channel_ref: str,
        *,
        options,
    ) -> dict:
        calls.append((channel_ref, options.last))
        ready.wait(timeout=2)
        return listings[channel_ref]

    monkeypatch.setattr(
        search_discovery, "fetch_channel_videos", fake_fetch_channel_videos
    )

    results = search_videos(
        SearchRequest(
            queries=(),
            channels=(
                ChannelTarget.from_ref("@OpenAI"),
                ChannelTarget.from_ref("@Anthropic"),
            ),
            limit=2,
        )
    )

    assert sorted(calls) == [("@Anthropic", 2), ("@OpenAI", 2)]
    assert [result["video_id"] for result in results] == [
        "bbbbbbbbbbb",
        "ccccccccccc",
    ]


def test_search_result_keeps_raw_metadata_when_transcript_fetch_fails(
    monkeypatch,
    search_transcript_side_effects,
) -> None:
    def fake_search_youtube(
        query: str,
        *,
        fetch_limit: int,
        sort: str,
    ) -> list[dict]:
        return [
            {
                "id": "aaaaaaaaaaa",
                "title": "Metadata stays",
            }
        ]

    def fail_fetch_transcript(
        video_id: str,
        *,
        options=None,
        metadata: dict | None = None,
    ) -> TranscriptDocument:
        raise RuntimeError("captions unavailable")

    monkeypatch.setattr(search_discovery, "search_youtube", fake_search_youtube)
    monkeypatch.setattr(search_semantic, "fetch_transcript", fail_fetch_transcript)

    assert search_videos(
        SearchRequest(
            queries=("agents",),
            limit=1,
            min_views=None,
            min_duration=None,
            max_duration=None,
        )
    ) == [
        {
            "type": "video",
            "video_id": "aaaaaaaaaaa",
            "title": "Metadata stays",
            "channel_id": None,
            "channel_handle": None,
            "upload_date": None,
            "duration": None,
            "view_count": None,
            "like_count": None,
            "source": "youtube",
        }
    ]


@pytest.mark.parametrize("mode", ["native", "semantic"])
def test_search_preserves_provider_handle_for_filters_and_results(
    monkeypatch, mode
) -> None:
    """A yt-dlp uploader ID must remain usable when the display name differs and channel_handle is absent."""

    def discover(*args, **kwargs):
        return [
            {
                "id": "aaaaaaaaaaa",
                "title": "Agent example",
                "channel": "Different Display Name",
                "uploader_id": "@CreatorHandle",
            }
        ]

    monkeypatch.setattr(search_discovery, "search_youtube", discover)
    results = search_videos(
        SearchRequest(
            queries=("agents",), mode=mode, limit=1, include_channel=("@CreatorHandle",)
        )
    )
    assert len(results) == 1
    result = cast(dict[str, Any], results[0])
    metadata = result if mode == "native" else result["video"]
    assert metadata["channel_handle"] == "@CreatorHandle"
