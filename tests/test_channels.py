"""Channel URL, list membership, and option validation tests."""

import pytest

from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.identity import YouTubeRefs, normalize_channel_url, parse_channel_ref
from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.storage import (
    consume_new_watchlist_videos,
    get_connection,
    list_blacklist_channel_identities,
    list_watchlist_channel_refs,
    store_watchlist_channel_listing,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from youtube_toolkit.youtube.channels import (
    ChannelOptions,
    add_watchlist_channels,
    remove_blacklist_channels,
    remove_watchlist_channels,
    sync_watchlist,
    validate_channel_options,
    watch_new_videos,
)


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


def test_parser_accepts_watchlist_new_and_scheduled_sync() -> None:
    """Parse agent-facing and scheduler-facing watchlist commands."""
    new_args = build_parser().parse_args(["watchlist", "new", "--channel", "@OpenAI"])
    scheduled_args = build_parser().parse_args(["watchlist", "scheduled-sync", "--last", "5"])

    assert new_args.watchlist_command == "new"
    assert new_args.channel_refs == ["@OpenAI"]
    assert scheduled_args.watchlist_command == "scheduled-sync"
    assert scheduled_args.last == 5


def test_add_watchlist_channels_accepts_single_and_batch_channel_refs(
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

    assert add_watchlist_channels("@OpenAI") == [
        {
            "channel_ref": "@OpenAI",
            "channel": "@OpenAI",
        },
    ]
    assert add_watchlist_channels(["https://www.youtube.com/channel/UC456"]) == [
        {
            "channel_ref": "https://www.youtube.com/channel/UC456",
            "channel": "@Anthropic",
        },
    ]

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
            "channel_ref": "@OpenAI",
            "channel": "@OpenAI",
            "removed": True,
        }
    ]

    with get_connection() as connection:
        channel_row = connection.execute(
            """
            SELECT id
            FROM channels
            WHERE id = ?
            """,
            (channel_id,),
        ).fetchone()
        watched_channels = list_watchlist_channel_refs(connection)

    assert channel_row is not None
    assert watched_channels == []


def test_remove_watchlist_channel_deletes_queue_only(
    monkeypatch,
    tmp_path,
) -> None:
    """Remove a watched channel queue while preserving stored video rows."""
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
            "channel_ref": "@OpenAI",
            "channel": "@OpenAI",
            "removed": True,
        }
    ]

    with get_connection() as connection:
        video_row = connection.execute(
            """
            SELECT youtube_video_id
            FROM videos
            WHERE youtube_video_id = ?
            """,
            ("video-openai",),
        ).fetchone()
        queue_rows = connection.execute("SELECT * FROM watchlist_videos").fetchall()

    assert video_row is not None
    assert queue_rows == []


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
            "channel_ref": "https://www.youtube.com/c/OpenAI",
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
            "channel_ref": "https://www.youtube.com/channel/UC123",
            "channel": "UC123",
            "removed": True,
        }
    ]

    with get_connection() as connection:
        channel_row = connection.execute(
            """
            SELECT id
            FROM channels
            WHERE id = ?
            """,
            (channel_id,),
        ).fetchone()
        blacklisted_channels = list_blacklist_channel_identities(connection)

    assert channel_row is not None
    assert blacklisted_channels == []


def test_remove_missing_channel_reports_no_membership(
    monkeypatch,
    tmp_path,
) -> None:
    """Report a no-op when the requested channel is not in the list."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    assert remove_watchlist_channels(["@Missing"]) == [
        {
            "channel_ref": "@Missing",
            "channel": "@Missing",
            "removed": False,
        }
    ]


def test_store_watchlist_channel_listing_marks_only_new_video_ids(
    monkeypatch,
    tmp_path,
) -> None:
    """Queue newly discovered watched videos without re-queuing known IDs."""
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

        rows = connection.execute("SELECT seen_at FROM watchlist_videos").fetchall()

    assert len(rows) == 1
    assert rows[0]["seen_at"] is None


def test_watch_new_videos_consumes_only_scoped_channels(
    monkeypatch,
    tmp_path,
) -> None:
    """Return and clear new rows only for the requested channel scope."""
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
        store_watchlist_channel_listing(connection, openai_listing())
        store_watchlist_channel_listing(connection, anthropic_listing())

    videos = watch_new_videos(channel_refs="@OpenAI")

    assert [video["video_id"] for video in videos] == ["video-openai"]
    assert videos[0]["transcript_status"] == "missing"

    with get_connection() as connection:
        remaining = consume_new_watchlist_videos(
            connection,
            channel_identities=[
                ChannelIdentity(
                    youtube_channel_id=None,
                    channel_handle="@Anthropic",
                )
            ],
        )

    assert [video["video_id"] for video in remaining] == ["video-anthropic"]


def test_watch_new_videos_requires_explicit_scope(
    monkeypatch,
    tmp_path,
) -> None:
    """Reject ambiguous consumes that could clear the wrong queue."""
    monkeypatch.setenv("YOUTUBE_TOOLKIT_STORAGE_HOME", str(tmp_path / "state"))

    with pytest.raises(ValueError, match="include_all"):
        watch_new_videos()


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

    def fake_hydrate_watchlist_transcripts(video_ids: list[str]):
        assert video_ids == ["video-openai"]
        return ([{"video_id": "video-openai", "source": "database", "language": "stored"}], [])

    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.fetch_channel_metadata",
        fake_fetch_channel_metadata,
    )
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.hydrate_watchlist_transcripts",
        fake_hydrate_watchlist_transcripts,
    )

    result = sync_watchlist(options=ChannelOptions(last=3))

    assert result == {
        "watched_channels": ["@OpenAI"],
        "new_video_ids": ["video-openai"],
        "hydrated_transcripts": [
            {
                "video_id": "video-openai",
                "source": "database",
                "language": "stored",
            }
        ],
        "failed_transcripts": [],
    }


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
                "view_count": 1000,
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
                "view_count": 900,
            }
        ],
    }
