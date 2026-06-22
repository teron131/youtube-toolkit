from youtube_toolkit.output import format_timestamp


def test_format_timestamp_uses_hour_when_needed() -> None:
    assert format_timestamp(3723) == "1:02:03"


def test_format_timestamp_uses_minute_without_hour() -> None:
    assert format_timestamp(123) == "2:03"
