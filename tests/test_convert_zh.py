from youtube_toolkit.convert_zh import convert_zh


def test_convert_zh_uses_hong_kong_traditional_script() -> None:
    assert convert_zh("数据库") == "數據庫"
