"""YouTube reference identity parsing tests."""

from youtube_toolkit.identity import YouTubeRefs, extract_channel_id, extract_video_id, is_channel_id, is_video_id, normalize_youtube_url


def test_extract_video_id_from_common_url_shapes() -> None:
    assert extract_video_id("JpSHyEIZ_bo") == "JpSHyEIZ_bo"
    assert extract_video_id("https://youtu.be/JpSHyEIZ_bo?t=10") == "JpSHyEIZ_bo"
    assert extract_video_id("https://www.youtube.com/watch?v=JpSHyEIZ_bo") == "JpSHyEIZ_bo"
    assert extract_video_id("https://www.youtube.com/shorts/JpSHyEIZ_bo") == "JpSHyEIZ_bo"


def test_video_and_channel_id_detectors() -> None:
    assert is_video_id("JpSHyEIZ_bo")
    assert not is_video_id("https://youtu.be/JpSHyEIZ_bo")
    assert is_channel_id("UC123")
    assert not is_channel_id("@OpenAI")
    assert extract_channel_id("UC123") == "UC123"
    assert extract_channel_id("@OpenAI") is None


def test_normalize_youtube_url() -> None:
    expected_url = "https://www.youtube.com/watch?v=JpSHyEIZ_bo"
    noisy_urls = [
        "JpSHyEIZ_bo",
        "https://youtu.be/JpSHyEIZ_bo?t=10",
        "https://youtu.be/JpSHyEIZ_bo?si=sharetoken&t=10",
        "https://www.youtube.com/watch?v=JpSHyEIZ_bo&t=120s&list=PL123&index=4",
        "https://www.youtube.com/watch?list=PL123&index=4&v=JpSHyEIZ_bo&feature=shared",
        "https://m.youtube.com/watch?v=JpSHyEIZ_bo&pp=tags&t=120",
        "https://music.youtube.com/watch?v=JpSHyEIZ_bo&si=sharetoken",
        "https://www.youtube.com/embed/JpSHyEIZ_bo?start=120",
        "https://www.youtube.com/live/JpSHyEIZ_bo?feature=shared",
        "https://www.youtube.com/shorts/JpSHyEIZ_bo?feature=share",
        "https://www.youtube.com/watch?v=JpSHyEIZ_bo#t=120",
    ]

    for noisy_url in noisy_urls:
        assert normalize_youtube_url(noisy_url) == expected_url


def test_youtube_refs_group_urls_and_ids() -> None:
    refs = YouTubeRefs.parse(["JpSHyEIZ_bo", "https://youtu.be/JpSHyEIZ_bo?t=10"])

    assert list(YouTubeRefs.parse("JpSHyEIZ_bo")) == ["JpSHyEIZ_bo"]
    assert list(YouTubeRefs.parse(None)) == []
    assert refs.video_ids() == ["JpSHyEIZ_bo", "JpSHyEIZ_bo"]
    assert refs.video_urls() == [
        "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
        "https://www.youtube.com/watch?v=JpSHyEIZ_bo",
    ]
