import pytest

from youtube_toolkit.channels import ChannelOptions, normalize_channel_url, validate_channel_options


def test_normalize_channel_url_from_common_inputs() -> None:
    assert normalize_channel_url("@example") == "https://www.youtube.com/@example/videos"
    assert normalize_channel_url("UCabc") == "https://www.youtube.com/channel/UCabc/videos"
    assert normalize_channel_url("https://www.youtube.com/@example/videos") == "https://www.youtube.com/@example/videos"


def test_validate_channel_options_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="--last"):
        validate_channel_options(ChannelOptions(last=0))

    with pytest.raises(ValueError, match="--days"):
        validate_channel_options(ChannelOptions(days=0))
