"""Chinese script policy: titles and transcript text are converted to Taiwan Traditional through one shared OpenCC converter, except Japanese text, whose kanji would change with it."""

from __future__ import annotations

from functools import cache
import re

import opencc

# Hiragana and katakana letters appear only in Japanese, and all but the shortest Japanese text uses them; the katakana middle dot is left out because Chinese text also writes it between parts of a foreign name.
KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa]")


def convert_zh(text: str) -> str:
    """Characters change but wording does not: `数据库里` becomes `數據庫裡`, not Taiwan's `資料庫裡`.

    Text with kana is Japanese and comes back unchanged, since conversion would replace kanji that share a Simplified form, such as 画 with 畫.
    """
    if is_japanese(text):
        return text
    return _taiwan_converter().convert(text)


def is_japanese(text: str) -> bool:
    """Judge a transcript as a whole rather than line by line, since a short Japanese line can be all kanji."""
    return KANA.search(text) is not None


@cache
def _taiwan_converter() -> opencc.OpenCC:
    # Construction loads OpenCC dictionaries (about 24 ms); conversion only reads them.
    return opencc.OpenCC("s2tw.json")
