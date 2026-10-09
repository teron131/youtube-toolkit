"""Channel reference normalization and bounded listing retrieval tests."""

from datetime import date

import pytest

from youtube_toolkit.identity import (
    ChannelIdentity,
    normalize_channel_url,
    parse_channel_ref,
)
from youtube_toolkit.youtube.channels import (
    ChannelOptions,
    fetch_channel_videos,
    validate_channel_options,
)


def test_fetch_channel_videos_returns_provider_metadata() -> None:
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

    assert (
        fetch_channel_videos(
            "@OpenAI", options=ChannelOptions(last=1), client=FakeClient()
        )
        == openai_listing()
    )


def test_normalize_channel_url_from_common_inputs() -> None:
    """Normalize handles, ids, and channel URLs into videos-tab URLs."""
    assert (
        normalize_channel_url("@example") == "https://www.youtube.com/@example/videos"
    )
    assert (
        normalize_channel_url("UCabc") == "https://www.youtube.com/channel/UCabc/videos"
    )
    assert (
        normalize_channel_url("https://www.youtube.com/@example/videos")
        == "https://www.youtube.com/@example/videos"
    )


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
    assert parse_channel_ref(
        "https://www.youtube.com/channel/UC123"
    ) == ChannelIdentity(
        youtube_channel_id="UC123",
        channel_handle=None,
    )


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

    metadata = fetch_channel_videos(
        "https://www.youtube.com/@OpenAI/videos",
        options=ChannelOptions(last=None, days=3),
        client=FakeClient(),
    )

    assert calls == [("https://www.youtube.com/@OpenAI/videos", 10, False)]
    assert [entry["id"] for entry in metadata["entries"]] == ["recent"]


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
