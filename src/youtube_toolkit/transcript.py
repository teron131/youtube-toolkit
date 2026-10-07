"""Cache complete transcript documents locally while preserving language, timing, and source policy."""

from __future__ import annotations

from dataclasses import asdict

from .cache import read_cache, write_cache
from .identity import VideoRef, require_video_id
from .schemas import (
    TranscriptChunk,
    TranscriptDocument,
    TranscriptSection,
    VideoMetadata,
)
from .youtube.caption import (
    CaptionMetadataClient,
    TranscriptOptions,
    fetch_live_transcript,
)


def fetch_transcript(
    video_ref: VideoRef,
    *,
    options: TranscriptOptions | None = None,
    client: CaptionMetadataClient | None = None,
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
            document = TranscriptDocument(
                metadata=VideoMetadata(**cached["metadata"]),
                language=cached["language"],
                source=cached["source"],
                chunks=[TranscriptChunk(**chunk) for chunk in cached["chunks"]],
                sections=[
                    TranscriptSection(**section) for section in cached["sections"]
                ],
            )
            if document.metadata.video_id == key["video_id"] and document.text:
                return document
        except (KeyError, TypeError, ValueError):
            pass  # A disposable cache entry can be rebuilt from the source.
    document = fetch_live_transcript(video_ref, options=options, client=client)
    write_cache("transcripts", key, asdict(document))
    return document
