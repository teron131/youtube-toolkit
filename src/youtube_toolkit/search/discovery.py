"""Acquire ordered YouTube and channel candidate lists, resolving channel names before metadata filtering."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from ..identity import ChannelIdentity, extract_channel_identity
from ..youtube.channels import ChannelOptions, fetch_channel_videos
from ..youtube.client import YtDlpClient
from .preprocessing import dedupe_candidates, interlace_relevance
from .request import ChannelTarget, ResolvedRequest, SearchRequest

CHANNEL_NAME_SEARCH_LIMIT = 10
MAX_CONCURRENT_DISCOVERY_FETCHES = 4


def youtube_candidates(
    resolved: ResolvedRequest,
    *,
    fetch_limit: int,
) -> list[dict[str, Any]]:
    """Run one YouTube search per query and merge the result lists for the resolved ranking policy.

    The fetch budget is split across the queries, since each supplies only its share of the merged results.
    """
    youtube_sort = "popularity" if resolved.sort == "popularity" else "relevance"
    queries = tuple(dict.fromkeys(resolved.request.queries))
    per_query_limit = -(-fetch_limit // len(queries))
    # executor.map preserves query order even when later requests finish first.
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DISCOVERY_FETCHES) as executor:
        candidate_lists = list(
            executor.map(
                partial(youtube_search, fetch_limit=per_query_limit, sort=youtube_sort),
                queries,
            )
        )
    if resolved.sort == "relevance":
        candidates = interlace_relevance(candidate_lists)
    else:
        candidates = dedupe_candidates(
            [candidate for results in candidate_lists for candidate in results]
        )
    return [{**candidate, "source": "youtube"} for candidate in candidates]


def youtube_search(
    query: str,
    *,
    fetch_limit: int,
    sort: str,
) -> list[dict[str, Any]]:
    """Call the yt-dlp search adapter and keep only object entries."""
    client = YtDlpClient()
    payload = client.search(
        query,
        fetch=fetch_limit,
        sort=sort,
    )
    return [entry for entry in payload.get("entries") or [] if isinstance(entry, dict)]


def channel_listings(
    request: SearchRequest,
    *,
    limit: int,
) -> list[list[dict[str, Any]]]:
    """Fetch channels concurrently while retaining separate newest-first lists in the requested channel order."""
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DISCOVERY_FETCHES) as executor:
        return list(
            executor.map(partial(_channel_listing, limit=limit), request.channels)
        )


def _channel_listing(selection: ChannelTarget, *, limit: int) -> list[dict[str, Any]]:
    """Resolve and fetch one channel without mixing its chronology with other channels."""
    channel_ref, channel_identity = resolve_channel_listing(selection)
    metadata = fetch_channel_videos(channel_ref, options=ChannelOptions(last=limit))
    return [
        {**candidate, "source": "channel"}
        for candidate in dedupe_candidates(
            channel_entries(metadata, known_channel=channel_identity)
        )
    ]


def resolve_channel_listing(selection: ChannelTarget) -> tuple[str, ChannelIdentity]:
    """Resolve a channel target to a concrete channel ref plus identity for listing fetches."""
    if selection.channel_name is None:
        return (
            selection.channel_ref(),
            ChannelIdentity(
                youtube_channel_id=selection.youtube_channel_id,
                channel_handle=selection.channel_handle,
            ),
        )
    wanted_name = normalized_channel_name(selection.channel_name)
    for query in (
        selection.channel_name,
        f"{selection.channel_name} official YouTube channel",
    ):
        for match in youtube_search(
            query,
            fetch_limit=CHANNEL_NAME_SEARCH_LIMIT,
            sort="relevance",
        ):
            if not any(
                normalized_channel_name(value) == wanted_name
                for value in (match.get("channel"), match.get("uploader"))
                if isinstance(value, str)
            ):
                continue
            identity = extract_channel_identity(match)
            channel_ref = identity.youtube_channel_id or identity.channel_handle
            if channel_ref is not None:
                return (
                    channel_ref,
                    identity,
                )
    raise ValueError(
        f"Could not resolve YouTube channel name: {selection.channel_name}"
    )


def normalized_channel_name(value: str) -> str:
    """Normalize channel names for fuzzy YouTube-search confirmation."""
    return " ".join(value.casefold().split())


def channel_entries(
    metadata: dict[str, Any],
    *,
    known_channel: ChannelIdentity,
) -> list[dict[str, Any]]:
    """Attach known channel identity fields to listing entries before candidate filtering."""
    channel_identity = extract_channel_identity(
        metadata,
        fallback_channel_id=known_channel.youtube_channel_id,
        fallback_channel_handle=known_channel.channel_handle,
    )
    return [
        {
            **entry,
            "channel_id": entry.get("channel_id")
            or channel_identity.youtube_channel_id,
            "channel_handle": entry.get("channel_handle")
            or channel_identity.channel_handle,
        }
        for entry in metadata.get("entries") or []
        if isinstance(entry, dict)
    ]
