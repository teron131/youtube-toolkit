from youtube_toolkit.ytdlp import build_ffmpeg_audio_args


def test_build_ffmpeg_audio_args_hardcodes_bitrate() -> None:
    assert (
        build_ffmpeg_audio_args(
            sample_rate=16_000,
            channels=1,
            bitrate="32k",
        )
        == "ffmpeg:-ac 1 -ar 16000 -b:a 32k"
    )
