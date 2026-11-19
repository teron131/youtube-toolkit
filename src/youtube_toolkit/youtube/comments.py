"""Comment search over one bounded sample, blending Top and like-count rankings."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import cycle
from typing import Any

from ..artifacts import write_record
from ..identity import VideoRef, normalize_youtube_url, require_video_id
from .client import YtDlpClient

DEFAULT_COMMENT_LIMIT = 100
# Keep comment pagination bounded even when a caller overrides the normal sample size.
MAX_COMMENT_LIMIT = 1000


def search_comments(
    video_ref: VideoRef, query: str = "", *, limit: int = DEFAULT_COMMENT_LIMIT
) -> dict[str, Any]:
    """An omitted or blank query returns the entire fetched sample; `sampled` counts the comments fetched.

    The limit applies before filtering; sparse matches never trigger additional retrievals.
    The record keeps the unfiltered sample in Top order with the same lean fields as the matches.
    """
    if not 1 <= limit <= MAX_COMMENT_LIMIT:
        raise ValueError(f"--comment-limit must be between 1 and {MAX_COMMENT_LIMIT}")
    payload = YtDlpClient().comments(video_ref, limit=limit)
    comments = payload.get("comments") or []
    document = {
        "title": payload.get("title"),
        "url": normalize_youtube_url(payload.get("id") or ""),
        "comments": _project(comments),
    }
    write_record("comments", require_video_id(video_ref), None, document)
    needle = query.strip().casefold()
    matches = [
        comment
        for comment in comments
        if needle in (comment.get("text") or "").casefold()
    ]
    return {
        "title": document["title"],
        "url": document["url"],
        "query": query,
        "sampled": len(comments),
        "comments": _project(_interleave(matches)),
    }


def _project(comments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"text": comment.get("text") or "", "like_count": comment.get("like_count")}
        for comment in comments
    ]


def _interleave(comments: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Like-count ties preserve Top order, and unknown counts sort last.

    Both rankings contain the same sample, so exhausting either means every comment has been emitted.
    """
    by_likes = sorted(
        comments,
        key=lambda comment: (
            comment.get("like_count") is not None,
            comment.get("like_count") or 0,
        ),
        reverse=True,
    )
    seen: set[str] = set()
    for ranking in cycle((iter(comments), iter(by_likes))):
        comment = next((item for item in ranking if item["id"] not in seen), None)
        if comment is None:
            return
        seen.add(comment["id"])
        yield comment
