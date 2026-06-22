import pytest

from youtube_toolkit.schemas import ChannelIdentity
from youtube_toolkit.search import SearchOptions, filter_entries, parse_duration


def test_parse_duration_accepts_seconds_and_colon_values() -> None:
    assert parse_duration("90") == 90
    assert parse_duration("1:30") == 90
    assert parse_duration("1:02:03") == 3723


def test_parse_duration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid duration"):
        parse_duration("1::30")


def test_filter_entries_applies_metadata_filters() -> None:
    entries = [
        {
            "id": "aaaaaaaaaaa",
            "ie_key": "Youtube",
            "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "title": "LangGraph tutorial",
            "channel": "LangChain",
            "view_count": 25_000,
            "duration": 600,
        },
        {
            "id": "bbbbbbbbbbb",
            "ie_key": "Youtube",
            "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
            "title": "Short",
            "channel": "Other",
            "view_count": 500,
            "duration": 30,
        },
    ]
    assert filter_entries(
        entries,
        SearchOptions(
            limit=5,
            include_channel=("lang",),
        ),
    ) == [entries[0]]


def test_filter_entries_excludes_blacklisted_channels() -> None:
    entries = [
        {
            "id": "aaaaaaaaaaa",
            "ie_key": "Youtube",
            "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "title": "LangGraph tutorial",
            "channel": "LangChain",
            "channel_id": "UC123",
            "view_count": 25_000,
            "duration": 600,
        },
    ]
    assert (
        filter_entries(
            entries,
            SearchOptions(limit=5),
            blacklisted_channels=[
                ChannelIdentity(
                    youtube_channel_id="UC123",
                    channel_handle=None,
                )
            ],
        )
        == []
    )
