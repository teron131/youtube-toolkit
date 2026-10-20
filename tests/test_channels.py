"""Channel reference normalization and bounded listing retrieval tests."""

from datetime import date, timedelta

import pytest

from youtube_toolkit.identity import (
    ChannelIdentity,
    normalize_channel_url,
    parse_channel_ref,
)
from youtube_toolkit.youtube import channels
from youtube_toolkit.youtube.channels import (
    ChannelOptions,
    fetch_channel_videos,
    validate_channel_options,
)


def test_fetch_channel_videos_returns_provider_metadata() -> None:
    class FakeClient:
        def channel_videos(self, channel_url: str, *, last: int | None) -> dict:
            assert channel_url == "https://www.youtube.com/@OpenAI/videos"
            assert last == 1
            return openai_listing()

        def metadata(self, video_id: str) -> dict:
            pytest.fail("count listings need no per-video metadata")

    assert (
        fetch_channel_videos(
            "@OpenAI", options=ChannelOptions(last=1), client=FakeClient()
        )
        == openai_listing()
    )


def test_normalize_channel_url_from_common_inputs() -> None:
    assert (
        normalize_channel_url("@example") == "https://www.youtube.com/@example/videos"
    )
    assert (
        normalize_channel_url("UCabcdefghijklmnopqrstuv")
        == "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv/videos"
    )
    assert (
        normalize_channel_url("https://www.youtube.com/@example/videos")
        == "https://www.youtube.com/@example/videos"
    )


def test_parse_channel_ref_normalizes_public_channel_refs() -> None:
    assert parse_channel_ref("@OpenAI") == ChannelIdentity(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )
    assert parse_channel_ref("https://www.youtube.com/@OpenAI") == ChannelIdentity(
        youtube_channel_id=None,
        channel_handle="@OpenAI",
    )
    assert parse_channel_ref(
        "https://www.youtube.com/channel/UC1234567890123456789012"
    ) == ChannelIdentity(
        youtube_channel_id="UC1234567890123456789012",
        channel_handle=None,
    )


def test_unlimited_days_window_uses_bounded_first_fetch(monkeypatch) -> None:
    calls = []
    metadata_calls = []
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.cutoff_date_for_days",
        lambda days: date(2026, 6, 29),
    )
    upload_dates = {"recent": "20260630", "old": "20260601"}

    class FakeClient:
        def channel_videos(self, channel_url: str, *, last: int | None) -> dict:
            calls.append((channel_url, last))
            return {
                "entries": [
                    {"id": "recent", "title": "Recent"},
                    {"id": "old", "title": "Old"},
                ]
            }

        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            return {"id": video_id, "upload_date": upload_dates[video_id]}

    listing = fetch_channel_videos(
        "https://www.youtube.com/@OpenAI/videos",
        options=ChannelOptions(last=None, days=3),
        client=FakeClient(),
    )

    assert calls == [("https://www.youtube.com/@OpenAI/videos", 10)]
    assert sorted(metadata_calls) == ["old", "recent"]
    assert listing["entries"] == [
        {"id": "recent", "title": "Recent", "upload_date": "20260630"}
    ]


def test_days_window_grows_listing_and_dates_each_video_once(monkeypatch) -> None:
    today = date(2026, 7, 31)
    monkeypatch.setattr(
        "youtube_toolkit.youtube.channels.cutoff_date_for_days",
        lambda days: today - timedelta(days=days),
    )
    video_ids = [f"video{index:03d}" for index in range(60)]
    # Three uploads per day, newest first.
    upload_dates = {
        video_id: (today - timedelta(days=index // 3)).strftime("%Y%m%d")
        for index, video_id in enumerate(video_ids)
    }
    listing_sizes = []
    metadata_calls = []

    class FakeClient:
        def channel_videos(self, channel_url: str, *, last: int | None) -> dict:
            listing_sizes.append(last)
            return {"entries": [{"id": video_id} for video_id in video_ids[:last]]}

        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            if video_id == "video005":
                raise RuntimeError("members-only video")
            return {"id": video_id, "upload_date": upload_dates[video_id]}

    listing = fetch_channel_videos(
        "@OpenAI", options=ChannelOptions(days=7), client=FakeClient()
    )

    batch_size = channels.MAX_CONCURRENT_METADATA_FETCHES
    assert listing_sizes == [10, 20, 40]
    assert sorted(metadata_calls) == video_ids[: 4 * batch_size]
    assert len(metadata_calls) == len(set(metadata_calls))
    assert [entry["id"] for entry in listing["entries"]] == [
        video_id for video_id in video_ids[:24] if video_id != "video005"
    ]
    assert listing["entries"][-1]["upload_date"] == "20260724"


def test_days_window_stops_when_no_listed_video_can_be_dated() -> None:
    listing_sizes = []
    metadata_calls = []

    class FakeClient:
        def channel_videos(self, channel_url: str, *, last: int | None) -> dict:
            listing_sizes.append(last)
            return {
                "entries": [{"id": f"video{index:03d}"} for index in range(last or 0)]
            }

        def metadata(self, video_id: str) -> dict:
            metadata_calls.append(video_id)
            raise RuntimeError("members-only video")

    listing = fetch_channel_videos(
        "@OpenAI", options=ChannelOptions(days=7), client=FakeClient()
    )

    assert listing_sizes == [10]
    assert len(metadata_calls) == channels.MAX_CONCURRENT_METADATA_FETCHES
    assert listing["entries"] == []


def test_validate_channel_options_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--last"):
        validate_channel_options(ChannelOptions(last=0))

    with pytest.raises(ValueError, match="--days"):
        validate_channel_options(ChannelOptions(days=0))


def openai_listing() -> dict:
    return {
        "channel_id": "UC1234567890123456789012",
        "channel_handle": "@OpenAI",
        "entries": [
            {
                "id": "video-openai",
                "title": "OpenAI update",
                "upload_date": "20260630",
            }
        ],
    }
