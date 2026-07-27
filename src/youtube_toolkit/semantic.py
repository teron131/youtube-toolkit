"""Storage-independent transcript chunking and vector similarity."""

from __future__ import annotations

from typing import Any
import math

TRANSCRIPT_CHUNK_CHARS = 1200
TRANSCRIPT_CHUNK_OVERLAP_CHARS = 200


def split_transcript_chunks(
    transcript: str,
    *,
    chunk_chars: int = TRANSCRIPT_CHUNK_CHARS,
    overlap_chars: int = TRANSCRIPT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split normalized transcript text into overlapping chunks for embedding search."""
    clean_transcript = " ".join(transcript.split())
    if not clean_transcript:
        return []
    if chunk_chars <= 0:
        raise ValueError("transcript chunk size must be greater than 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("transcript chunk overlap must be non-negative and smaller than chunk size")

    chunks: list[str] = []
    start = 0
    while start < len(clean_transcript):
        end = min(len(clean_transcript), start + chunk_chars)
        chunk = clean_transcript[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(clean_transcript):
            break
        start = end - overlap_chars
    return chunks


def cosine_similarity(
    left: list[float],
    right: Any,
) -> float:
    """Compute cosine similarity for same-length numeric vectors, treating invalid rows as zero."""
    if not isinstance(right, list) or len(left) != len(right):
        return 0.0
    if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in right):
        return 0.0
    right_values = [float(value) for value in right]
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right_values, strict=True))
    left_norm = math.sqrt(sum(left_value * left_value for left_value in left))
    right_norm = math.sqrt(sum(right_value * right_value for right_value in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
