"""Cache complete transcript documents locally while preserving language, timing, and source policy."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from .cache import read_cache, write_cache
from .identity import VideoRef, require_video_id
from .schemas import TranscriptChunk, TranscriptDocument, VideoChapter, VideoMetadata
from .youtube.caption import TranscriptClient, TranscriptOptions, fetch_live_transcript
from .youtube.client import YtDlpClient


def fetch_transcript(
    video_ref: VideoRef,
    *,
    options: TranscriptOptions | None = None,
    client: TranscriptClient | None = None,
) -> TranscriptDocument:
    """Reuse a complete task-local transcript for the same video and caption policy, otherwise fetch captions or STT."""
    options = options or TranscriptOptions()
    key = {
        "video_id": require_video_id(video_ref),
        "languages": list(options.languages),
        "source": options.source,
    }
    cached = read_cache("transcripts", key)
    if cached is not None:
        try:
            cached_metadata = cached["metadata"]
            metadata_values: dict[str, Any] = {**cached_metadata}
            metadata_values["chapters"] = [
                VideoChapter(**chapter)
                for chapter in cached_metadata.get("chapters", [])
            ]
            document = TranscriptDocument(
                metadata=VideoMetadata(**metadata_values),
                language=cached["language"],
                source=cached["source"],
                chunks=[TranscriptChunk(**chunk) for chunk in cached["chunks"]],
            )
        except (KeyError, TypeError, ValueError):
            pass  # A disposable cache entry can be rebuilt from the source.
        else:
            if document.metadata.video_id == key["video_id"] and document.text:
                if any(
                    name not in cached_metadata
                    for name in ("description", "thumbnail", "chapters")
                ):
                    # Upgrade metadata without downloading captions or repeating paid STT.
                    metadata = VideoMetadata.from_yt_dlp(
                        (client or YtDlpClient()).metadata(video_ref)
                    )
                    if metadata.video_id != document.metadata.video_id:
                        raise ValueError(
                            "Fetched metadata does not match cached transcript video"
                        )
                    document = replace(document, metadata=metadata)
                    write_cache("transcripts", key, asdict(document))
                return document
    document = fetch_live_transcript(video_ref, options=options, client=client)
    write_cache("transcripts", key, asdict(document))
    return document
