"""Name-shaped words in a video's text, which speech-to-text and rewrite requests are given so they spell names right."""

from __future__ import annotations

from collections import Counter
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import VideoMetadata

# NAME judges only a word's shape, so it also takes ordinary capitalized words and Japanese loanwords.
# It matches katakana runs of three or more, words with a capital or digit after their first letter, capitalized words in mid-sentence, and capitalized words right after Chinese or Japanese text, which capitalizes nothing but names.
# Latin words end at the first non-ASCII letter, since Chinese and Japanese put no space between a name and the next word, as in GPTは.
NAME = re.compile(
    r"[\u30a0-\u30ff]{3,}"
    r"|(?<![A-Za-z0-9])[A-Z][A-Za-z0-9.-]*[A-Z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?![A-Za-z0-9])"
    r"|(?<=[^\s.!?] )[A-Z][a-z]+(?![A-Za-z0-9])"
    r"|(?<=[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef])[A-Z][a-z]+(?![A-Za-z0-9])"
)
# Links in a description carry video and channel IDs shaped like names, such as VRFfAeW30qE.
LINK = re.compile(r"https?://\S+")


def metadata_names(metadata: VideoMetadata) -> list[str]:
    """Most frequent first, including ordinary words from a title written in title case."""
    written = "\n".join(
        [
            metadata.title or "",
            metadata.description or "",
            *(chapter.title for chapter in metadata.chapters),
        ]
    )
    counts = Counter(NAME.findall(LINK.sub(" ", written)))
    return [name for name, _ in counts.most_common()]


def repeated_names(text: str) -> list[str]:
    """Most frequent first; mishearings the captions repeat stay in, since only one-off words are dropped."""
    counts = Counter(NAME.findall(text))
    return [name for name, count in counts.most_common() if count > 1]
