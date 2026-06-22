from youtube_toolkit.convert_zh import convert_zh, normalize_opencc_config


def test_normalize_opencc_config_accepts_short_name() -> None:
    assert normalize_opencc_config("s2hk") == "s2hk.json"
    assert normalize_opencc_config("s2hk.json") == "s2hk.json"


def test_convert_zh_uses_opencc_config() -> None:
    assert convert_zh("数据库", "s2hk") == "數據庫"
