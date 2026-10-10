"""Chinese script policy: all text the toolkit returns is converted to Taiwan Traditional through one shared OpenCC converter."""

from __future__ import annotations

from functools import cache

import opencc


def convert_zh(text: str) -> str:
    """Convert Simplified Chinese characters to Taiwan Traditional, keeping the original wording.

    It runs on every text without a language check: non-Chinese text passes through, except Japanese kanji that share a Simplified form.
    """
    return _taiwan_converter().convert(text)


@cache
def _taiwan_converter() -> opencc.OpenCC:
    # Construction loads OpenCC dictionaries (about 24 ms); conversion only reads them.
    return opencc.OpenCC("s2tw.json")
