from pathlib import Path

import pytest

from youtube_toolkit.audio import (
    is_default_audio_cache_path,
    normalize_audio_output_path,
    remove_cached_audio_file,
    resolve_audio_output_path,
)
from youtube_toolkit.ytdlp_client import build_ffmpeg_audio_args


def test_resolve_audio_output_path_defaults_to_cache_mp3() -> None:
    assert resolve_audio_output_path("JpSHyEIZ_bo", None) == Path(".cache/audio/JpSHyEIZ_bo.mp3")


def test_normalize_audio_output_path_adds_mp3_suffix() -> None:
    assert normalize_audio_output_path(Path("audio/sample")) == Path("audio/sample.mp3")


def test_normalize_audio_output_path_rejects_mismatched_suffix() -> None:
    with pytest.raises(ValueError, match=r"\.mp3"):
        normalize_audio_output_path(Path("audio/sample.wav"))


def test_build_ffmpeg_audio_args_hardcodes_bitrate() -> None:
    assert (
        build_ffmpeg_audio_args(
            sample_rate=16_000,
            channels=1,
            bitrate="32k",
        )
        == "ffmpeg:-ac 1 -ar 16000 -b:a 32k"
    )


def test_remove_cached_audio_file_only_removes_default_audio_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cached_audio = Path(".cache/audio/sample.mp3")
    custom_audio = Path("audio/sample.mp3")
    cached_audio.parent.mkdir(parents=True)
    custom_audio.parent.mkdir(parents=True)
    cached_audio.write_bytes(b"cached")
    custom_audio.write_bytes(b"custom")

    assert is_default_audio_cache_path(cached_audio)
    assert not is_default_audio_cache_path(custom_audio)
    assert remove_cached_audio_file(cached_audio)
    assert not cached_audio.exists()
    assert not remove_cached_audio_file(custom_audio)
    assert custom_audio.exists()
