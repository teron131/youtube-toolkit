"""Search one bounded comment sample and blend Top and like-count rankings into lean, source-linked results."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import cycle
from typing import Any

from ..artifacts import write_artifact
from ..identity import VideoRef, normalize_youtube_url, require_video_id
from .client import YtDlpClient

DEFAULT_COMMENT_LIMIT = 100


def search_comments(
    video_ref: VideoRef, query: str = "", *, limit: int = DEFAULT_COMMENT_LIMIT
) -> dict[str, Any]:
    """An omitted or blank query returns the entire fetched sample.

    The limit applies before filtering; sparse matches never trigger additional retrievals.
    The archive retains the unfiltered sample in Top order using the same lean fields as the output.
    """
    payload = YtDlpClient().comments(video_ref, limit=limit)
    comments = payload.get("comments") or []
    document = {
        "title": payload.get("title"),
        "url": normalize_youtube_url(payload.get("id") or ""),
        "comments": _project(comments),
    }
    write_artifact(
        "comments", require_video_id(video_ref), request=None, payload=document
    )
    query = query.strip().casefold()
    matches = [
        comment
        for comment in comments
        if query in (comment.get("text") or "").casefold()
    ]
    return {**document, "comments": _project(_interleave(matches))}


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
