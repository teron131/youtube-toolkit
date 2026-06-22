from youtube_toolkit.url import extract_video_id, normalize_youtube_url


def test_extract_video_id_from_common_url_shapes() -> None:
    assert extract_video_id("JpSHyEIZ_bo") == "JpSHyEIZ_bo"
    assert extract_video_id("https://youtu.be/JpSHyEIZ_bo?t=10") == "JpSHyEIZ_bo"
    assert extract_video_id("https://www.youtube.com/watch?v=JpSHyEIZ_bo") == "JpSHyEIZ_bo"
    assert extract_video_id("https://www.youtube.com/shorts/JpSHyEIZ_bo") == "JpSHyEIZ_bo"


def test_normalize_youtube_url() -> None:
    assert normalize_youtube_url("https://youtu.be/JpSHyEIZ_bo?t=10") == "https://www.youtube.com/watch?v=JpSHyEIZ_bo"
