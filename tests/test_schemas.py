from youtube_toolkit.schemas import VideoMetadata


def test_video_metadata_converts_title_to_hong_kong_traditional() -> None:
    metadata = VideoMetadata.from_yt_dlp(
        {
            "id": "JpSHyEIZ_bo",
            "title": "AI商业发展暗藏机会",
        }
    )

    assert metadata.title == "AI商業發展暗藏機會"
