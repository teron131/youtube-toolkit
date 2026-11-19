"""Downloads to the working directory of a video, its sound, or its frames, whole or one time range, or single frames at given times, named for the video's title and channel and handed back as they are when already there."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path
import shutil
from typing import Any, Literal, TypedDict

from .artifacts import temporary_directory
from .ffmpeg_utils import decode_frame
from .filename_utils import filename_stem
from .schemas import VideoMetadata
from .youtube import YtDlpClient

type MediaKind = Literal["video", "audio"]
type FrameKind = Literal["frames", "keyframes"]
type DownloadQuality = Literal["360p", "480p", "720p", "1080p"]

# YouTube's own 16:9 stream sizes; 4K is left out because its files run to gigabytes for little an agent can use. A frame of another shape is centred in this box between black bars.
DOWNLOAD_SIZES: dict[DownloadQuality, tuple[int, int]] = {
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
DEFAULT_DOWNLOAD_QUALITY: DownloadQuality = "1080p"
# A clip opens on the first frame at or after its start, and one second holds a frame at any frame rate.
FRAME_CLIP_SECONDS = 1
# A download counts as stuck after twice its span's playing time, and never sooner than vision's ten minutes: at the rate vision's download is sized for, a two-hour 720p video in ten minutes, even 1080p with sound arrives in well under half its playing time.
MIN_DOWNLOAD_TIMEOUT_SECONDS = 10 * 60


class MediaDownload(TypedDict):
    path: str
    bytes: int
    reused: bool


class FrameFolderDownload(TypedDict):
    path: str
    frames: int
    reused: bool


class FrameDownload(TypedDict):
    path: str
    reused: bool


def download_media(
    payload: dict[str, Any],
    kind: MediaKind,
    *,
    quality: DownloadQuality,
    start: float | None,
    end: float | None,
    fresh: bool,
    client: YtDlpClient,
) -> MediaDownload:
    """`start` and `end` in seconds keep only that span, which opens on the first frame at or after `start`; an `end` past the video stops at its end.

    The file name holds the kind, the video's quality, and the range, so a file already there is that same download and comes back without downloading, with `reused` set, unless `fresh` downloads it again.
    The quality names the requested cap, so a video smaller than it is downloaded again under each larger cap.
    A failed download can leave yt-dlp's .part files beside the target, which a rerun resumes, or for a range overwrites.
    """
    metadata = VideoMetadata.from_yt_dlp(payload)
    first, last, section = _download_range(metadata, start, end)
    labels = ([] if kind == "audio" else [quality]) + _range_labels(section)
    extension = "m4a" if kind == "audio" else "mp4"
    path = Path.cwd() / f"{filename_stem(metadata, '_'.join(labels))}.{extension}"
    if path.exists() and not fresh:
        return {"path": str(path), "bytes": path.stat().st_size, "reused": True}
    _require_ffmpeg()
    client.download_media(
        payload,
        path,
        height=None if kind == "audio" else DOWNLOAD_SIZES[quality][1],
        section=section,
        overwrite=fresh,
        timeout_seconds=_download_timeout(first, last),
    )
    return {"path": str(path), "bytes": path.stat().st_size, "reused": False}


def download_frames(
    payload: dict[str, Any],
    kind: FrameKind,
    *,
    quality: DownloadQuality,
    start: float | None,
    end: float | None,
    fresh: bool,
    client: YtDlpClient,
) -> FrameFolderDownload:
    """Save the video's keyframes from `start` to `end` in seconds, each a `TIME.jpg` at the quality's size named by its time in the video (`00h01m02.812s.jpg`), in a folder named like a video download plus the kind.

    keyframes keeps every keyframe; frames drops near-blank frames, near-duplicates of the last few seconds, and cross-fades, leaving one frame per visible change.
    A folder already there is that same download and comes back as it is unless `fresh`; frames gather in a `.part` folder that a rerun replaces, so a folder under the final name is complete.
    """
    metadata = VideoMetadata.from_yt_dlp(payload)
    first, last, section = _download_range(metadata, start, end)
    directory = Path.cwd() / filename_stem(
        metadata, "_".join([quality, kind, *_range_labels(section)])
    )
    if directory.exists() and not fresh:
        return {
            "path": str(directory),
            "frames": len(list(directory.glob("*.jpg"))),
            "reused": True,
        }
    ffmpeg_path = _require_ffmpeg()
    from .vision import collect_candidates

    width, height = DOWNLOAD_SIZES[quality]
    partial = directory.with_name(f"{directory.name}.part")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        with temporary_directory("clips-") as clips:
            video_path = clips / "video.mp4"
            client.download_silent_video(
                payload,
                video_path,
                height=height,
                section=section,
                max_bytes=None,
                timeout_seconds=_download_timeout(first, last),
            )
            frames = collect_candidates(
                video_path,
                ffmpeg_path,
                start=first,
                duration=last - first,
                frame_size=(width, height),
                directory=partial,
                excluded_intervals=[],
                distinct=kind == "frames",
            )
        for frame in frames:
            frame.image_path.rename(
                partial / f"{_time_label(frame.timestamp_seconds)}.jpg"
            )
        shutil.rmtree(directory, ignore_errors=True)
        partial.rename(directory)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return {"path": str(directory), "frames": len(frames), "reused": False}


def download_frames_at(
    payload: dict[str, Any],
    times: list[float],
    *,
    quality: DownloadQuality,
    fresh: bool,
    client: YtDlpClient,
) -> list[FrameDownload]:
    """Save the first frame at or after each time in seconds, keyframe or not, as a JPEG at the quality's size named like a video download with the time for its range (`TITLE_CHANNEL_720p_00h01m15.000s.jpg`), one result per distinct time in the order given.

    A frame already there comes back as it is unless `fresh`; the rest download together as one-second clips.
    """
    metadata = VideoMetadata.from_yt_dlp(payload)
    duration = _require_duration(metadata)
    outside = next((time for time in times if not 0 <= time < duration), None)
    if outside is not None:
        raise ValueError(
            f"Frame time {outside:g} seconds is outside the video, which is {duration:g} seconds long"
        )
    paths = {
        time: Path.cwd()
        / f"{filename_stem(metadata, f'{quality}_{_time_label(time)}')}.jpg"
        for time in times
    }
    missing = [time for time, path in paths.items() if fresh or not path.exists()]
    if missing:
        ffmpeg_path = _require_ffmpeg()
        from .vision import save_frame

        width, height = DOWNLOAD_SIZES[quality]
        with temporary_directory("clips-") as directory:
            clips = client.download_silent_clips(
                payload,
                directory,
                height=height,
                sections=[
                    (time, min(time + FRAME_CLIP_SECONDS, duration)) for time in missing
                ],
                timeout_seconds=MIN_DOWNLOAD_TIMEOUT_SECONDS,
            )
            for time, clip in zip(missing, clips, strict=True):
                partial = paths[time].with_name(f"{paths[time].name}.part")
                save_frame(
                    decode_frame(clip, 0, ffmpeg_path, width=width, height=height),
                    partial,
                )
                partial.replace(paths[time])
    return [
        {"path": str(path), "reused": time not in missing}
        for time, path in paths.items()
    ]


def _download_range(
    metadata: VideoMetadata, start: float | None, end: float | None
) -> tuple[float, float, tuple[float, float] | None]:
    """The first and last second to keep, and the section to cut, which is None when neither end is given."""
    duration = _require_duration(metadata)
    first = 0.0 if start is None else start
    last = duration if end is None else min(end, duration)
    if not 0 <= first < last:
        raise ValueError(
            "A download range must start before it ends and before the video ends"
        )
    return first, last, None if start is None and end is None else (first, last)


def _require_duration(metadata: VideoMetadata) -> float:
    if metadata.duration_seconds is None:
        raise ValueError(
            f"Missing duration for {metadata.video_id}; a live stream has none until it ends"
        )
    return metadata.duration_seconds


def _require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "Downloads need ffmpeg, which joins the picture with its sound, cuts time ranges, and decodes frames"
        )
    return ffmpeg_path


def _download_timeout(first: float, last: float) -> int:
    return max(MIN_DOWNLOAD_TIMEOUT_SECONDS, ceil(2 * (last - first)))


def _range_labels(section: tuple[float, float] | None) -> list[str]:
    if section is None:
        return []
    first, last = section
    return [f"{_time_label(first)}_{_time_label(last)}"]


def _time_label(seconds: float) -> str:
    """Hours, minutes, and seconds to the millisecond, every part at a fixed width so names list in time order: 62.81275 → `00h01m02.812s`.

    Milliseconds round down, since a time finds the first frame at or after it and a name past its frame would find the next one; rounding to the microsecond first keeps float noise from turning 1.005 into `00h00m01.004s`.
    """
    milliseconds = floor(round(seconds * 1000, 3))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole, fraction = divmod(remainder, 1000)
    return f"{hours:02d}h{minutes:02d}m{whole:02d}.{fraction:03d}s"
