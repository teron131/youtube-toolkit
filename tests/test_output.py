from pathlib import Path

import pytest

from youtube_toolkit.output import normalize_text_output_path, render_plain_text


def test_normalize_text_output_path_adds_txt_suffix() -> None:
    assert normalize_text_output_path(Path("transcripts/sample")) == Path("transcripts/sample.txt")


def test_normalize_text_output_path_rejects_non_txt_suffix() -> None:
    with pytest.raises(ValueError, match=r"\.txt"):
        normalize_text_output_path(Path("transcripts/sample.json"))


def test_render_plain_text_returns_only_text() -> None:
    assert render_plain_text("  This is the transcript.  ") == "This is the transcript.\n"
