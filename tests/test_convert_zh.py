"""Taiwan Traditional script without Taiwan phrasing: the expected text keeps mainland wording on purpose."""

from youtube_toolkit.convert_zh import convert_zh


def test_convert_zh_uses_taiwan_traditional_script() -> None:
    assert convert_zh("数据库里") == "數據庫裡"
