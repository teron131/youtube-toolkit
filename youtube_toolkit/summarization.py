"""Transcript-to-summary structured LLM calls."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from .model_config import ModelConfig

SUMMARY_SYSTEM_PROMPT = """Create a grounded, chronological summary of the supplied text.

Rules:
- Write the summary in English.
- Ground every claim in the source text; do not invent unsupported details.
- Chapters must be chronological and non-overlapping.
- Prefer concrete facts, names, numbers, claims, and steps when present.
- Avoid meta-language such as 'this video', 'the speaker says', or 'the transcript discusses'.
- Exclude sponsors, ads, promos, and calls to action entirely.
- Use null for start_time and end_time when timestamps are not present or unclear."""


class Chapter(BaseModel):
    """A chronological chapter in a text summary."""

    title: str = Field(description="Concise chapter heading.")
    description: str = Field(description="Grounded chapter description with key facts when present.")
    start_time: str | None = Field(default=None, description="Optional chapter start timestamp, such as MM:SS.")
    end_time: str | None = Field(default=None, description="Optional chapter end timestamp, such as MM:SS.")


class TranscriptSummary(BaseModel):
    """Structured transcript summary returned by an LLM or agent."""

    overview: str = Field(description="End-to-end summary of the full source text.")
    chapters: list[Chapter] = Field(
        description="Chronological, non-overlapping chapters covering the core content.",
    )

    def to_text(self) -> str:
        lines = ["# Summary", "", self.overview, "", "# Chapters"]
        for chapter in self.chapters:
            heading = chapter.title
            if chapter.start_time or chapter.end_time:
                heading = f"{chapter.start_time or '?'}-{chapter.end_time or '?'} {heading}"
            lines.extend(["", f"## {heading}", "", chapter.description])
        return "\n".join(lines).rstrip() + "\n"


def summarize_transcript(
    text: str,
    config: ModelConfig,
    *,
    title: str | None = None,
    description: str | None = None,
) -> TranscriptSummary:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Summary text cannot be empty")

    system_prompt = SUMMARY_SYSTEM_PROMPT
    context_lines = []
    if title:
        context_lines.append(f"Title: {title}")
    if description:
        context_lines.append(f"Description: {description}")
    if context_lines:
        system_prompt += "\n\nContext:\n" + "\n".join(context_lines)

    llm = ChatOpenAI(
        model=config.model,
        api_key=SecretStr(config.api_key),
        base_url=config.base_url,
    )
    result = llm.with_structured_output(TranscriptSummary).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Text to summarize:\n{clean_text}"),
        ]
    )
    return cast(TranscriptSummary, result)
