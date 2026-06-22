"""Rendered output helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from youtube_toolkit.schemas import TranscriptDocument


def write_or_print(
    output: str,
    output_path: Path | None,
) -> None:
    if output_path is None:
        print(output, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(str(output_path))


def normalize_text_output_path(output_path: Path) -> Path:
    if output_path.suffix == "":
        return output_path.with_suffix(".txt")
    if output_path.suffix.casefold() != ".txt":
        raise ValueError("text artifact output must be a .txt path")
    return output_path


def render_plain_text(text: str) -> str:
    return f"{text.strip()}\n"


def render_transcript_markdown(document: TranscriptDocument) -> str:
    metadata = document.metadata
    lines = [
        f"# {metadata.title or metadata.video_id}",
        "",
        f"- URL: {metadata.url}",
        f"- Video ID: {metadata.video_id}",
    ]
    if metadata.channel:
        lines.append(f"- Channel: {metadata.channel}")
    if metadata.upload_date:
        lines.append(f"- Upload date: {metadata.upload_date}")
    if metadata.view_count is not None:
        lines.append(f"- Views: {metadata.view_count:,}")
    if metadata.like_count is not None:
        lines.append(f"- Likes: {metadata.like_count:,}")
    lines.extend(
        [
            f"- Transcript language: {document.language}",
            f"- Transcript source: {document.source}",
            "",
        ]
    )

    for section in document.sections:
        heading = section.title
        if section.start_seconds is not None:
            heading = f"{format_timestamp(section.start_seconds)} {heading}"
        lines.extend(
            [
                f"## {heading}",
                "",
                section.text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_transcript_json(document: TranscriptDocument) -> str:
    return json.dumps(
        asdict(document),
        indent=2,
        ensure_ascii=False,
    )


def format_timestamp(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
