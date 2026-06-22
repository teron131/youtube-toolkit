"""Chinese script conversion helpers."""

from __future__ import annotations


def convert_zh(
    text: str,
    opencc_config: str,
) -> str:
    import opencc

    converter = opencc.OpenCC(normalize_opencc_config(opencc_config))
    return converter.convert(text)


def normalize_opencc_config(opencc_config: str) -> str:
    stripped_config = opencc_config.strip()
    if not stripped_config:
        raise ValueError("--opencc must not be empty")
    if stripped_config.endswith(".json"):
        return stripped_config
    return f"{stripped_config}.json"
