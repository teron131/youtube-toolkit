"""Video summaries as a one-pass LangGraph workflow: a gist of the transcript, then an outline that picks the sound bites carrying it by their sentence references, copied word for word, then one step that writes the summary from the transcript, guided by both."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .artifacts import write_artifact
from .config import LLMConfig
from .llm import ChatModel
from .prompts import (
    OWN_LANGUAGE,
    SUMMARY,
    SUMMARY_GIST,
    SUMMARY_OUTLINE,
    SUMMARY_OUTLINE_REQUEST,
    SUMMARY_TRANSCRIPT_REQUEST,
    SUMMARY_VIDEO,
    SUMMARY_WRITE_REQUEST,
    TARGET_LANGUAGE,
)
from .rewrite.sentences import split_sentences
from .schemas import TranscriptDocument

HASH_LENGTH = 6
# A sentence's reference is its number and a hash of its text, so a miscounted number does not copy the wrong sentence.
REFERENCE_PATTERN = rf"^\d+#[0-9a-f]{{{HASH_LENGTH}}}$"


class SoundBite(BaseModel):
    """Consecutive transcript sentences, first through last."""

    first: str = Field(
        description="The reference that the sound bite's first sentence opens with, such as 12#a3f9c1, copied exactly",
        pattern=REFERENCE_PATTERN,
    )
    last: str = Field(
        description="The reference that the sound bite's last sentence opens with, copied exactly; the same as first for a single sentence",
        pattern=REFERENCE_PATTERN,
    )


class OutlinePart(BaseModel):
    """One part of the video and its sound bites."""

    title: str = Field(
        description="What the part establishes and any change it makes from an earlier part, and, when the video covers several stages, versions, experiments, or things, which one it is about, since the speaker's words may leave that to context"
    )
    sound_bites: list[SoundBite] = Field(
        description="The sound bites taken from this part, in order; empty when nothing in it is worth keeping"
    )


class Outline(BaseModel):
    """A video cut down to the sound bites that carry its core."""

    parts: list[OutlinePart] = Field(
        description="The video's parts in the order it takes them up, leaving out sponsor messages and calls to subscribe"
    )


@dataclass
class _State:
    video: str
    transcript: str
    sentences: list[str]
    language: str
    gist: str = ""
    outline: str = ""
    summary: str = ""


@dataclass(frozen=True)
class _Context:
    llm: ChatModel


def summarize_transcript(
    document: TranscriptDocument, config: LLMConfig, *, language: str | None
) -> str:
    """Written in the transcript's own language when `language` is None; raises RuntimeError when a request fails or returns no text."""
    metadata = document.metadata
    video = SUMMARY_VIDEO.format(
        title=metadata.title or "",
        channel=metadata.channel or "",
        uploaded=metadata.upload_date or "",
        description=metadata.description or "",
        chapters="\n".join(chapter.title for chapter in metadata.chapters),
    )
    llm = ChatModel(config, reasoning_effort="medium")
    state = asyncio.run(
        _GRAPH.ainvoke(
            _State(
                video=video,
                transcript=document.text,
                sentences=[
                    sentence.text for sentence in split_sentences(document.segments)
                ],
                language=TARGET_LANGUAGE.format(language=language)
                if language
                else OWN_LANGUAGE,
            ),
            context=_Context(llm),
        )
    )
    request = {
        "video_id": metadata.video_id,
        "language": language,
        "model": config.model,
    }
    write_artifact(
        "summaries",
        metadata.video_id,
        request,
        {name: state[name] for name in ("gist", "outline", "summary")},
    )
    return state["summary"]


async def _gist(state: _State, runtime: Runtime[_Context]) -> dict[str, str]:
    user = SUMMARY_TRANSCRIPT_REQUEST.format(
        video=state.video, transcript=state.transcript
    )
    reply = await runtime.context.llm.ainvoke(SUMMARY_GIST, user)
    return {"gist": _checked(reply)}


async def _outline(state: _State, runtime: Runtime[_Context]) -> dict[str, str]:
    references = _references(state.sentences)
    user = SUMMARY_OUTLINE_REQUEST.format(
        video=state.video,
        gist=state.gist,
        lines="\n".join(
            f"[{reference}] {sentence}"
            for reference, sentence in zip(references, state.sentences, strict=True)
        ),
    )
    outline = await runtime.context.llm.ainvoke(SUMMARY_OUTLINE, user, schema=Outline)
    return {"outline": _checked(_copied(outline, state.sentences, references))}


async def _write(state: _State, runtime: Runtime[_Context]) -> dict[str, str]:
    user = SUMMARY_WRITE_REQUEST.format(
        video=state.video,
        transcript=state.transcript,
        gist=state.gist,
        outline=state.outline,
    )
    reply = await runtime.context.llm.ainvoke(
        SUMMARY.format(language=state.language), user
    )
    return {"summary": _checked(reply)}


def _references(sentences: list[str]) -> list[str]:
    return [
        f"{number}#{hashlib.sha256(sentence.encode()).hexdigest()[:HASH_LENGTH]}"
        for number, sentence in enumerate(sentences, 1)
    ]


def _copied(outline: Outline, sentences: list[str], references: list[str]) -> str:
    """Each part's title over its sound bites, a paragraph per run of consecutive sentences; a sound bite whose references find no sentence, or run backward, is dropped."""
    blocks = []
    for part in outline.parts:
        indexes: set[int] = set()
        for bite in part.sound_bites:
            first = _located(bite.first, references)
            last = _located(bite.last, references)
            if first is not None and last is not None and first <= last:
                indexes.update(range(first, last + 1))
        paragraphs: list[list[str]] = []
        for index in sorted(indexes):
            if index - 1 not in indexes:
                paragraphs.append([])
            paragraphs[-1].append(sentences[index])
        blocks.append(
            "\n".join([f"## {part.title}", *(" ".join(run) for run in paragraphs)])
        )
    return "\n\n".join(blocks)


def _located(reference: str, references: list[str]) -> int | None:
    """A reference whose number and hash disagree goes to the one sentence with that hash, as a miscounted number; None when no single sentence has it."""
    if reference in references:
        return references.index(reference)
    digest = reference.partition("#")[2]
    matches = [
        index
        for index, candidate in enumerate(references)
        if candidate.partition("#")[2] == digest
    ]
    return matches[0] if len(matches) == 1 else None


def _checked(reply: str) -> str:
    if not reply.strip():
        raise RuntimeError("The LLM returned an empty reply to a summary request")
    return reply.strip()


def _build_graph():
    graph = StateGraph(_State, context_schema=_Context)
    graph.add_node("gist", _gist)
    graph.add_node("outline", _outline)
    graph.add_node("write", _write)
    graph.add_edge(START, "gist")
    graph.add_edge("gist", "outline")
    graph.add_edge("outline", "write")
    graph.add_edge("write", END)
    return graph.compile()


_GRAPH = _build_graph()
