"""What every embedding client hands back: the error any of them raises, and the aligned vectors and usage that both vision clients return."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class EmbeddingError(RuntimeError):
    """Covers invalid embedding configuration as well as failed, empty, or malformed responses."""


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    """None means the provider did not report complete usage for that field; local models report none."""

    image_tokens: int | None
    text_tokens: int | None
    total_tokens: int | None

    def __add__(self, other: EmbeddingUsage) -> EmbeddingUsage:
        return EmbeddingUsage(
            sum_reported([self.image_tokens, other.image_tokens]),
            sum_reported([self.text_tokens, other.text_tokens]),
            sum_reported([self.total_tokens, other.total_tokens]),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    images: np.ndarray
    texts: np.ndarray
    usage: EmbeddingUsage


def sum_reported(counts: list[int | None]) -> int | None:
    """One unreported count makes the sum unreported, since a partial total would understate usage."""
    total = 0
    for count in counts:
        if count is None:
            return None
        total += count
    return total
