"""Channel URL, list membership, and option validation tests."""

from datetime import date
import threading

import pytest

import youtube_toolkit.storage as storage
from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.identity import YouTubeRefs, normalize_channel_url, parse_channel_ref
from youtube_toolkit.schemas import ChannelIdentity, TranscriptDocument, VideoMetadata
from youtube_toolkit.storage import (
    get_connection,
    list_blacklist_channel_identities,
    list_watchlist_channel_refs,
    store_channel_listing,
    store_watchlist_channel_listing,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from youtube_toolkit.youtube.channels import (
    ChannelOptions,
    add_watchlist_channels,
    fetch_channel_videos,
    fetch_watchlist_transcripts,
    get_recent_channel_entries,
    list_recent_watchlist_videos,
    list_watchlist_channels,
    remove_blacklist_channels,
    remove_watchlist_channels,
    sync_watchlist,
    validate_channel_options,
)


def test_fetch_channel_videos_stays_stateless_without_storage(monkeypatch) -> None:
    """Direct channel retrieval returns provider metadata without opening storage."""

    class FakeClient:
        def channel_videos(
            self,
            channel_url: str,
            *,
            last: int | None,
            flat: bool = True,
        ) -> dict:
            assert channel_url == "https://www.youtube.com/@OpenAI/videos"
            assert last == 1
            assert flat is True
            return openai_listing()

    monkeypatch.setattr("youtube_toolkit.youtube.channels.capability_available", lambda _capability: False)
    monkeypatch.setattr(storage, "get_connection", lambda: pytest.fail("stateless channel retrieval opened storage"))

    assert fetch_channel_videos("@OpenAI", options=ChannelOptions(last=1), client=FakeClient()) == openai_listing()


def test_normalize_channel_url_from_common_inputs() -> None:
    """Normalize handles, ids, and channel URLs into videos-tab URLs."""
    assert normalize_channel_url("@example") == "https://www.youtube.com/@example/videos"
    assert normalize_channel_url("UCabc") == "https://www.youtube.com/channel/UCabc/videos"
    assert normalize_channel_url("https://www.youtube.com/@example/videos") == "https://www.youtube.com/@example/videos"


def test_parse_channel_ref_normalizes_public_channel_refs() -> None:
    """Normalize handles and channel URLs without fetching remote metadata."""
    assert parse_channel_ref("@OpenAI") == ChannelIdentity(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )
    assert parse_channel_ref("https://www.youtube.com/@OpenAI") == ChannelIdentity(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )
    assert parse_channel_ref("https://www.youtube.com/channel/UC123") == ChannelIdentity(
        youtube_channel_id="UC123",
        channel_handle=None,
    )


def test_channel_refs_normalize_refs_and_parse_identities() -> None:
    """Normalize one-or-many channel refs and parse URL, handle, or ID identity."""
    refs = YouTubeRefs.parse(["@OpenAI", "https://www.youtube.com/channel/UC123"])

    assert list(YouTubeRefs.parse("@OpenAI")) == ["@OpenAI"]
    assert list(YouTubeRefs.parse(None)) == []
    assert refs.channel_identities() == [
        ChannelIdentity(
            youtube_channel_id=None,
            channel_handle="@OpenAI",
        ),
        ChannelIdentity(
            youtube_channel_id="UC123",
            channel_handle=None,
        ),
    ]


def test_parser_accepts_watchlist_and_blacklist_remove() -> None:
    """Parse remove commands for both channel membership lists."""
    watchlist_args = build_parser().parse_args(["watchlist", "remove", "@OpenAI", "UC123"])
    blacklist_args = build_parser().parse_args(["blacklist", "remove", "@OpenAI", "UC123"])

    assert watchlist_args.watchlist_command == "remove"
    assert watchlist_args.channel_refs == ["@OpenAI", "UC123"]
    assert blacklist_args.blacklist_command == "remove"
    assert blacklist_args.channel_refs == ["@OpenAI", "UC123"]


def test_parser_accepts_watchlist_recent_and_scheduled_sync() -> None:
    """Parse agent-facing and scheduler-facing watchlist commands."""
    list_args = build_parser().parse_args(["watchlist", "list"])
    recent_args = build_parser().parse_args(["watchlist", "recent", "@OpenAI"])
    scheduled_args = build_parser().parse_args(["watchlist", "scheduled-sync", "--last", "5"])

    assert list_args.watchlist_command == "list"
    assert recent_args.watchlist_command == "recent"
    assert recent_args.channel_refs == ["@OpenAI"]
    assert recent_args.days == 3
    assert scheduled_args.watchlist_command == "scheduled-sync"
    assert scheduled_args.last == 5
    assert scheduled_args.days == 3


def test_add_watchlist_channels_accepts_single_and_list_channel_refs(
    monkeypatch,
    tmp_path,
) -> None:
    """Add one or many public channel refs through normalized stored identities."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    listings = {
        "@OpenAI": openai_listing(),
        "https://www.youtube.com/channel/UC456": anthropic_listing(),
    }

    def fake_fetch_channel_metadata(
        channel_ref: str,
        *,
        options: ChannelOptions,
    ) -> dict:
        assert options.last == 1
        return listings[channel_ref]

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_channel_metadata",
        fake_fetch_channel_metadata,
    )

    assert add_watchlist_channels("@OpenAI") == ["@OpenAI"]
    assert add_watchlist_channels(["https://www.youtube.com/channel/UC456"]) == ["@Anthropic"]

    with get_connection() as connection:
        assert list_watchlist_channel_refs(connection) == ["@Anthropic", "@OpenAI"]
        assert connection.rows("videos") == []

    assert list_watchlist_channels() == ["@Anthropic", "@OpenAI"]


def test_watchlist_membership_stores_youtube_channel_id(
    monkeypatch,
    tmp_path,
) -> None:
    """Watchlist membership uses the public YouTube channel ID as its key."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        channel_id = upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        membership_row = connection.find_row("channels_watchlist", "channel_id", "UC123")
        columns = connection.table("channels_watchlist").schema.names

    assert channel_id == "UC123"
    assert membership_row is not None
    assert membership_row["channel_id"] == "UC123"
    assert columns == ["channel_id"]


def test_channel_merge_keeps_one_real_watchlist_channel_id(
    monkeypatch,
    tmp_path,
) -> None:
    """Merging channel identities keeps membership on the final YouTube ID."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle=None,
            ),
        )
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UCOLD",
                channel_handle="@OpenAI",
            ),
        )
        store_watchlist_channel_listing(
            connection,
            {
                "channel_id": "UCOLD",
                "channel_handle": "@OpenAI",
                "entries": [
                    {
                        "id": "old-video",
                    }
                ],
            },
        )

        store_channel_listing(connection, openai_listing())

        channel_ids = sorted(row["channel_id"] for row in connection.rows("channels_watchlist"))
        video_rows = sorted(
            connection.rows("videos"),
            key=lambda row: row["youtube_video_id"],
        )

    assert channel_ids == ["UC123"]
    assert list_watchlist_channels() == ["@OpenAI"]
    assert [(row["youtube_video_id"], row["channel_id"]) for row in video_rows] == [
        ("old-video", "UC123"),
        ("video-openai", "UC123"),
    ]


def test_add_watchlist_channels_fetches_refs_concurrently(
    monkeypatch,
    tmp_path,
) -> None:
    """Fan out yt-dlp metadata fetches while preserving ordered membership writes."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    both_refs_started = threading.Barrier(2)

    def fake_fetch_channel_metadata(
        channel_ref: str,
        *,
        options: ChannelOptions,
    ) -> dict:
        assert options.last == 1
        both_refs_started.wait(timeout=1)
        return {
            "channel_id": f"UC{channel_ref.removeprefix('@')}",
            "channel_handle": channel_ref,
        }

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_channel_metadata",
        fake_fetch_channel_metadata,
    )

    assert add_watchlist_channels(["@OpenAI", "@Anthropic"]) == ["@OpenAI", "@Anthropic"]

    with get_connection() as connection:
        assert list_watchlist_channel_refs(connection) == ["@Anthropic", "@OpenAI"]


def test_remove_watchlist_channel_deletes_membership_only(
    monkeypatch,
    tmp_path,
) -> None:
    """Remove a watched channel while preserving the stored channel row."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        channel_id = upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )

    assert remove_watchlist_channels("@OpenAI") == [
        {
            "channel": "@OpenAI",
            "removed": True,
        }
    ]

    with get_connection() as connection:
        channel_row = connection.find_row("channels", "youtube_channel_id", channel_id)
        watched_channels = list_watchlist_channel_refs(connection)

    assert channel_row is not None
    assert watched_channels == []


def test_remove_watchlist_channel_deletes_watchlist_video_rows_only(
    monkeypatch,
    tmp_path,
) -> None:
    """Remove watched-channel video records while preserving stored catalog rows."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        store_watchlist_channel_listing(connection, openai_listing())

    assert remove_watchlist_channels(["@OpenAI"]) == [
        {
            "channel": "@OpenAI",
            "removed": True,
        }
    ]

    with get_connection() as connection:
        video_row = connection.find_row("videos", "youtube_video_id", "video-openai")
        watchlist_video_rows = connection.rows("watchlist_videos")

    assert video_row is not None
    assert watchlist_video_rows == []


def test_remove_watchlist_channel_resolves_ambiguous_url(
    monkeypatch,
    tmp_path,
) -> None:
    """Fetch ambiguous channel URLs before removing their stored membership."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )

    def fake_fetch_channel_metadata(
        channel_ref: str,
        *,
        options: ChannelOptions,
    ) -> dict:
        assert channel_ref == "https://www.youtube.com/c/OpenAI"
        assert options.last == 1
        return {
            "channel_id": "UC123",
            "channel_handle": "@OpenAI",
        }

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_channel_metadata",
        fake_fetch_channel_metadata,
    )

    assert remove_watchlist_channels(["https://www.youtube.com/c/OpenAI"]) == [
        {
            "channel": "@OpenAI",
            "removed": True,
        }
    ]


def test_remove_blacklist_channel_deletes_membership_only(
    monkeypatch,
    tmp_path,
) -> None:
    """Remove a blacklisted channel while preserving the stored channel row."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        channel_id = upsert_blacklist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )

    assert remove_blacklist_channels(["https://www.youtube.com/channel/UC123"]) == [
        {
            "channel": "UC123",
            "removed": True,
        }
    ]

    with get_connection() as connection:
        channel_row = connection.find_row("channels", "youtube_channel_id", channel_id)
        blacklisted_channels = list_blacklist_channel_identities(connection)

    assert channel_row is not None
    assert blacklisted_channels == []


def test_blacklist_membership_stores_youtube_channel_id(
    monkeypatch,
    tmp_path,
) -> None:
    """Blacklist membership uses the public YouTube channel ID as its key."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        channel_id = upsert_blacklist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        membership_row = connection.find_row("channels_blacklist", "channel_id", "UC123")
        columns = connection.table("channels_blacklist").schema.names

    assert channel_id == "UC123"
    assert membership_row is not None
    assert membership_row["channel_id"] == "UC123"
    assert columns == ["channel_id"]


def test_remove_missing_channel_reports_no_membership(
    monkeypatch,
    tmp_path,
) -> None:
    """Report a no-op when the requested channel is not in the list."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    assert remove_watchlist_channels(["@Missing"]) == [
        {
            "channel": "@Missing",
            "removed": False,
        }
    ]


def test_store_watchlist_channel_listing_records_only_new_video_ids(
    monkeypatch,
    tmp_path,
) -> None:
    """Record newly discovered watched videos without re-recording known IDs."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        assert store_watchlist_channel_listing(connection, openai_listing()) == ["video-openai"]
        assert store_watchlist_channel_listing(connection, openai_listing()) == []

        rows = connection.rows("watchlist_videos")

    assert len(rows) == 1
    assert rows[0]["video_id"] == "video-openai"


def test_store_watchlist_channel_listing_records_existing_catalog_video(
    monkeypatch,
    tmp_path,
) -> None:
    """Watchlist discovery is separate from the global video catalog."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        store_channel_listing(connection, openai_listing())
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )

        assert store_watchlist_channel_listing(connection, openai_listing()) == ["video-openai"]
        assert store_watchlist_channel_listing(connection, openai_listing()) == []


def test_list_recent_watchlist_videos_filters_by_objective_recency(
    monkeypatch,
    tmp_path,
) -> None:
    """Return recent watched videos through an objective date window."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.cutoff_date_for_days",
        lambda days: date(2026, 6, 29),
    )

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC456",
                channel_handle="@Anthropic",
            ),
        )
        store_watchlist_channel_listing(connection, openai_listing())
        store_watchlist_channel_listing(connection, anthropic_listing())

    videos = list_recent_watchlist_videos(
        channel_refs="@OpenAI",
        days=3,
    )

    assert [video["video_id"] for video in videos] == ["video-openai"]
    assert videos[0]["transcript_status"] == "missing"
    assert "transcript" not in videos[0]


def test_list_recent_watchlist_videos_rejects_non_positive_days(
    monkeypatch,
    tmp_path,
) -> None:
    """Reject non-positive objective recency windows."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with pytest.raises(ValueError, match="--days"):
        list_recent_watchlist_videos(days=0)


def test_unlimited_days_window_uses_bounded_first_fetch(monkeypatch) -> None:
    """A None last window has no fixed cap but still fetches bounded date windows."""
    calls = []
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.cutoff_date_for_days",
        lambda days: date(2026, 6, 29),
    )

    class FakeClient:
        def channel_videos(
            self,
            channel_url: str,
            *,
            last: int | None,
            flat: bool = True,
        ) -> dict:
            calls.append((channel_url, last, flat))
            return {
                "entries": [
                    {
                        "id": "recent",
                        "upload_date": "20260630",
                    },
                    {
                        "id": "old",
                        "upload_date": "20260601",
                    },
                ]
            }

    metadata = get_recent_channel_entries(
        FakeClient(),
        "https://www.youtube.com/@OpenAI/videos",
        options=ChannelOptions(last=None, days=3),
    )

    assert calls == [("https://www.youtube.com/@OpenAI/videos", 10, False)]
    assert [entry["id"] for entry in metadata["entries"]] == ["recent"]


def test_sync_watchlist_marks_new_videos_and_hydrates_missing(
    monkeypatch,
    tmp_path,
) -> None:
    """Sync watched channels through video-ID discovery and transcript hydration."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )

    def fake_fetch_channel_metadata(
        channel_ref: str,
        *,
        options: ChannelOptions,
    ) -> dict:
        assert channel_ref == "@OpenAI"
        assert options.last == 3
        return openai_listing()

    def fake_fetch_watchlist_transcripts(video_ids: list[str]):
        assert video_ids == ["video-openai"]
        return [
            {
                "status": "stored",
                "video_id": "video-openai",
            }
        ]

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_channel_metadata",
        fake_fetch_channel_metadata,
    )
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_watchlist_transcripts",
        fake_fetch_watchlist_transcripts,
    )

    result = sync_watchlist(options=ChannelOptions(last=3))

    assert result == {
        "channels": ["@OpenAI"],
        "new_videos": ["video-openai"],
        "transcripts": [
            {
                "status": "stored",
                "video_id": "video-openai",
            }
        ],
        "embeddings": [],
    }


def test_sync_watchlist_fetches_channels_concurrently(
    monkeypatch,
    tmp_path,
) -> None:
    """Fan out watched-channel yt-dlp fetches before ordered storage updates."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with get_connection() as connection:
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC123",
                channel_handle="@OpenAI",
            ),
        )
        upsert_watchlist_channel(
            connection,
            ChannelIdentity(
                youtube_channel_id="UC456",
                channel_handle="@Anthropic",
            ),
        )

    both_refs_started = threading.Barrier(2)
    stored_refs: list[str] = []

    listings = {
        "@Anthropic": anthropic_listing(),
        "@OpenAI": openai_listing(),
    }

    def fake_fetch_channel_metadata(
        channel_ref: str,
        *,
        options: ChannelOptions,
    ) -> dict:
        assert options.last == 2
        both_refs_started.wait(timeout=1)
        return listings[channel_ref]

    def fake_store_watchlist_channel_listing(
        connection,
        metadata: dict,
    ) -> list[str]:
        stored_refs.append(metadata["channel_handle"])
        return [metadata["entries"][0]["id"]]

    def fake_fetch_watchlist_transcripts(video_ids: list[str]):
        assert video_ids == []
        return []

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_channel_metadata",
        fake_fetch_channel_metadata,
    )
    monkeypatch.setattr(
        storage,
        "store_watchlist_channel_listing",
        fake_store_watchlist_channel_listing,
    )
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_watchlist_transcripts",
        fake_fetch_watchlist_transcripts,
    )

    result = sync_watchlist(options=ChannelOptions(last=2))

    assert result["channels"] == ["@Anthropic", "@OpenAI"]
    assert result["new_videos"] == ["video-anthropic", "video-openai"]
    assert result["embeddings"] == []
    assert stored_refs == ["@Anthropic", "@OpenAI"]


def test_fetch_watchlist_transcripts_fetches_videos_concurrently(monkeypatch) -> None:
    """Fan out transcript acquisition for missing watchlist videos."""
    both_videos_started = threading.Barrier(2)

    def fake_fetch_transcript(
        video_id: str,
        *,
        auto_index: bool,
    ) -> TranscriptDocument:
        assert auto_index is False
        both_videos_started.wait(timeout=1)
        if video_id == "video-failed":
            raise RuntimeError("caption missing")
        return TranscriptDocument(
            metadata=VideoMetadata(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
            ),
            language="stored",
            source="database",
            chunks=[],
            sections=[],
        )

    def fake_auto_index_video_embedding(connection, video_id: str):
        return {
            "status": "indexed",
            "video_id": video_id,
            "dimensions": 768,
        }

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_transcript",
        fake_fetch_transcript,
    )
    monkeypatch.setattr(
        storage,
        "auto_index_video_embedding",
        fake_auto_index_video_embedding,
    )

    transcripts = fetch_watchlist_transcripts(
        [
            "video-openai",
            "video-failed",
        ]
    )

    assert transcripts == [
        {
            "status": "stored",
            "video_id": "video-openai",
            "embedding": {
                "status": "indexed",
                "video_id": "video-openai",
                "dimensions": 768,
            },
        },
        {
            "status": "failed",
            "video_id": "video-failed",
            "error": "caption missing",
        },
    ]


def test_validate_channel_options_rejects_non_positive_values() -> None:
    """Reject non-positive channel fetch window values."""
    with pytest.raises(ValueError, match="--last"):
        validate_channel_options(ChannelOptions(last=0))

    with pytest.raises(ValueError, match="--days"):
        validate_channel_options(ChannelOptions(days=0))


def openai_listing() -> dict:
    return {
        "channel_id": "UC123",
        "channel_handle": "@OpenAI",
        "entries": [
            {
                "id": "video-openai",
                "title": "OpenAI update",
                "upload_date": "20260630",
            }
        ],
    }


def anthropic_listing() -> dict:
    return {
        "channel_id": "UC456",
        "channel_handle": "@Anthropic",
        "entries": [
            {
                "id": "video-anthropic",
                "title": "Anthropic update",
                "upload_date": "20260629",
            }
        ],
    }
