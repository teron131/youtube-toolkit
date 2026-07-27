"""Transcript workflow adds optional storage to live caption and STT acquisition."""

from __future__ import annotations

from typing import Any, cast

from .capabilities import capability_available
from .identity import VideoRef
from .schemas import TranscriptDocument, VideoMetadata
from .youtube.caption import (
    CaptionMetadataClient,
    SttAudioClient,
    TranscriptOptions,
    fetch_live_transcript,
    fetch_transcript_from_metadata,
    transcript_document_from_text,
)
from .youtube.client import YtDlpClient


def fetch_transcript(
    video_ref: VideoRef,
    *,
    options: TranscriptOptions | None = None,
    client: CaptionMetadataClient | None = None,
    auto_index: bool = True,
) -> TranscriptDocument:
    """Use storage when installed and otherwise return a live caption or STT document."""
    options = options or TranscriptOptions()
    if not capability_available("storage"):
        return fetch_live_transcript(
            video_ref,
            options=options,
            client=client,
        )

    if options == TranscriptOptions():
        stored_document = fetch_stored_transcript(video_ref)
        if stored_document is not None:
            if auto_index:
                index_stored_transcript(stored_document.metadata.video_id)
            return stored_document

    client = client or YtDlpClient()
    metadata_payload = client.metadata(video_ref)
    document = fetch_transcript_from_metadata(
        metadata_payload,
        options=options,
        client=cast(SttAudioClient, client),
    )

    from .storage import get_connection, upsert_video

    with get_connection() as connection:
        upsert_video(
            connection,
            metadata_payload,
            transcript=document.text,
        )
    if auto_index:
        index_stored_transcript(document.metadata.video_id)
    return document


def index_stored_transcript(youtube_video_id: str) -> None:
    """Maintain the video-level embedding for a stored transcript."""
    from .storage import auto_index_video_embedding, get_connection

    with get_connection() as connection:
        auto_index_video_embedding(connection, youtube_video_id)


def fetch_stored_transcript(video_ref: VideoRef) -> TranscriptDocument | None:
    """Use the direct stored-video lookup path for transcript cache hits."""
    from .storage import get_connection, get_stored_video

    with get_connection() as connection:
        stored_video = get_stored_video(
            connection,
            video_ref,
            require_transcript=True,
        )
    if stored_video is None:
        return None
    return stored_transcript_document(stored_video)


def stored_transcript_document(
    stored_video: dict[str, Any],
    *,
    language: str = "stored",
    source: str = "database",
) -> TranscriptDocument:
    """Rewrap stored transcript text in the document contract shared with live acquisition."""
    transcript = stored_video["transcript"]
    metadata = VideoMetadata(
        video_id=stored_video["video_id"],
        url=f"https://www.youtube.com/watch?v={stored_video['video_id']}",
        title=stored_video["title"],
        channel=None,
        channel_id=stored_video["channel_id"],
        channel_handle=stored_video["channel_handle"],
        upload_date=stored_video["upload_date"],
        duration_seconds=stored_video["duration"],
        view_count=stored_video["view_count"],
        like_count=stored_video["like_count"],
    )
    return transcript_document_from_text(
        metadata,
        transcript,
        language=language,
        source=source,
    )
