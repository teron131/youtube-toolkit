from youtube_toolkit.schemas import TranscriptChunk
from youtube_toolkit.ytdlp.caption import remove_sponsorblock_chunks


def test_remove_sponsorblock_chunks_drops_overlapping_chunks() -> None:
    chunks = [
        TranscriptChunk(start_seconds=0, end_seconds=5, text="main"),
        TranscriptChunk(start_seconds=5, end_seconds=10, text="sponsor"),
        TranscriptChunk(start_seconds=10, end_seconds=15, text="more"),
    ]

    assert remove_sponsorblock_chunks(chunks, [(5, 10)]) == [
        TranscriptChunk(start_seconds=0, end_seconds=5, text="main"),
        TranscriptChunk(start_seconds=10, end_seconds=15, text="more"),
    ]
