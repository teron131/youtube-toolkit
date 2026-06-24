"""Channel URL, list membership, and option validation tests."""

import pytest

from youtube_toolkit.cli.parser import build_parser
from youtube_toolkit.schemas import ChannelIdentity, parse_channel_identity_input
from youtube_toolkit.storage import (
    get_connection,
    list_blacklist_channel_identities,
    list_watchlist_channel_inputs,
    upsert_blacklist_channel,
    upsert_watchlist_channel,
)
from youtube_toolkit.ytdlp.channels import (
    ChannelOptions,
    normalize_channel_url,
    remove_blacklist_channel,
    remove_watchlist_channel,
    validate_channel_options,
)


def test_normalize_channel_url_from_common_inputs() -> None:
    """Normalize handles, ids, and channel URLs into videos-tab URLs."""
    assert normalize_channel_url("@example") == "https://www.youtube.com/@example/videos"
    assert normalize_channel_url("UCabc") == "https://www.youtube.com/channel/UCabc/videos"
    assert normalize_channel_url("https://www.youtube.com/@example/videos") == "https://www.youtube.com/@example/videos"


def test_parse_channel_identity_input_normalizes_removal_inputs() -> None:
    """Normalize handles and channel URLs without fetching remote metadata."""
    assert parse_channel_identity_input("@OpenAI") == ChannelIdentity(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )
    assert parse_channel_identity_input("https://www.youtube.com/@OpenAI") == ChannelIdentity(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )
    assert parse_channel_identity_input("https://www.youtube.com/channel/UC123") == ChannelIdentity(
        youtube_channel_id="UC123",
        channel_handle=None,
    )


def test_parser_accepts_watchlist_and_blacklist_remove() -> None:
    """Parse remove commands for both channel membership lists."""
    watchlist_args = build_parser().parse_args(["watchlist", "remove", "@OpenAI"])
    blacklist_args = build_parser().parse_args(["blacklist", "remove", "@OpenAI"])

    assert watchlist_args.watchlist_command == "remove"
    assert watchlist_args.channel == "@OpenAI"
    assert blacklist_args.blacklist_command == "remove"
    assert blacklist_args.channel == "@OpenAI"


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

    assert remove_watchlist_channel("@OpenAI") == "Removed @OpenAI from watchlist"

    with get_connection() as connection:
        channel_row = connection.execute(
            """
            SELECT id
            FROM channels
            WHERE id = ?
            """,
            (channel_id,),
        ).fetchone()
        watched_channels = list_watchlist_channel_inputs(connection)

    assert channel_row is not None
    assert watched_channels == []


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

    def fake_fetch_channel_videos(
        channel_input: str,
        *,
        options: ChannelOptions,
    ) -> dict:
        assert channel_input == "https://www.youtube.com/c/OpenAI"
        assert options.last == 1
        return {
            "channel_id": "UC123",
            "channel_handle": "@OpenAI",
        }

    monkeypatch.setattr(
        "youtube_toolkit.ytdlp.channels.fetch_channel_videos",
        fake_fetch_channel_videos,
    )

    assert remove_watchlist_channel("https://www.youtube.com/c/OpenAI") == "Removed @OpenAI from watchlist"


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

    assert remove_blacklist_channel("https://www.youtube.com/channel/UC123") == "Removed UC123 from blacklist"

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

    assert remove_watchlist_channel("@Missing") == "@Missing was not in watchlist"


def test_validate_channel_options_rejects_non_positive_values() -> None:
    """Reject non-positive channel fetch window values."""
    with pytest.raises(ValueError, match="--last"):
        validate_channel_options(ChannelOptions(last=0))

    with pytest.raises(ValueError, match="--days"):
        validate_channel_options(ChannelOptions(days=0))
