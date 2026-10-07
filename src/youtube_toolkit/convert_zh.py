"""Chinese text conversion owns the toolkit's Hong Kong Traditional script policy and its shared OpenCC converter."""

from __future__ import annotations

from functools import cache

import opencc


def convert_zh(text: str) -> str:
    """Convert Simplified Chinese script to Hong Kong Traditional; other text passes through unchanged."""
    return _hong_kong_converter().convert(text)


@cache
def _hong_kong_converter() -> opencc.OpenCC:
    # Construction loads OpenCC dictionaries (about 24 ms); conversion only reads them.
    return opencc.OpenCC("s2hk.json")
