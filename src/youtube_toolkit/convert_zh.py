"""Chinese script policy: titles and transcript text are converted to Taiwan Traditional through one shared OpenCC converter."""

from __future__ import annotations

from functools import cache

import opencc


def convert_zh(text: str) -> str:
    """Characters change but wording does not: `数据库里` becomes `數據庫裡`, not Taiwan's `資料庫裡`.

    It runs on every text without a language check: non-Chinese text passes through, except Japanese kanji that share a Simplified form.
    """
    return _taiwan_converter().convert(text)


@cache
def _taiwan_converter() -> opencc.OpenCC:
    # Construction loads OpenCC dictionaries (about 24 ms); conversion only reads them.
    return opencc.OpenCC("s2tw.json")
