"""The article route: sentences grouped into paragraphs at the model's turning points and chapter openings, then rewritten whole under the video's chapter titles."""

from __future__ import annotations

from bisect import bisect_right
from itertools import pairwise
import re

from ..llm import ChatModel
from ..prompts import (
    CHAPTER_MARKERS,
    CORRECT_ARTICLE,
    TRANSLATE_ARTICLE,
    TRANSLATE_TITLES,
    TURNING_POINTS,
)
from ..schemas import TranscriptDocument, TranscriptSegment, VideoChapter
from .requests import (
    ATTEMPTS,
    RewriteError,
    fixed_batches,
    numbered,
    replies,
    request,
    run_batches,
)
from .sentences import split_sentences

SENTENCE_ID = re.compile(r"\[LINE(\d+)\]")
CHAPTER_ID = re.compile(r"CHAPTER(\d+)\s*\[LINE(\d+)\]")
SENTENCES_PER_WINDOW = 128
# A sentence at a window's edge still sees the talk on both sides.
WINDOW_CONTEXT = 10
# Whole paragraphs are packed into requests of up to this many sentences to keep each request short; a longer paragraph goes alone.
SENTENCES_PER_BATCH = 40
CONTEXT_SENTENCES = 2
# A same-language cleanup keeps close to its source's length, so a batch outside these bounds was summarized or padded.
MIN_ARTICLE_RATIO = 0.5
MAX_ARTICLE_RATIO = 1.3


def rewrite_article(
    llm: ChatModel, header: str, document: TranscriptDocument, language: str | None
) -> TranscriptSegment:
    sentences = split_sentences(document.segments)
    texts = [sentence.text for sentence in sentences]
    chapters = _chapter_starts(
        sentences, document.metadata.chapters if document.timed else []
    )
    turning_points, openings = _find_turning_points(llm, header, texts, chapters)
    titles = _placed_chapters(chapters, openings, len(texts))
    starts = sorted({0, *titles, *turning_points})
    paragraphs = _rewrite_paragraphs(llm, header, texts, starts, language)
    if language and any(titles.values()):
        titles = _translate_titles(llm, header, titles, language)
    blocks: list[str] = []
    for start, paragraph in zip(starts, paragraphs, strict=True):
        if title := titles.get(start):
            blocks.append(f"## {title}")
        blocks.append(paragraph)
    return TranscriptSegment(
        sentences[0].start_seconds, sentences[-1].end_seconds, "\n\n".join(blocks)
    )


def _chapter_starts(
    sentences: list[TranscriptSegment], chapters: list[VideoChapter]
) -> dict[int, str | None]:
    """The first sentence of each chapter's run, by the chapter the sentence starts in, with that chapter's title; sentences before the first chapter start an untitled run."""
    ordered = sorted(chapters, key=lambda chapter: chapter.start_seconds)
    starts = [chapter.start_seconds for chapter in ordered]
    runs: dict[int, str | None] = {}
    current: int | None = None
    for index, sentence in enumerate(sentences):
        chapter = bisect_right(starts, sentence.start_seconds) - 1
        if chapter != current:
            runs[index] = ordered[chapter].title if chapter >= 0 else None
            current = chapter
    return runs


def _find_turning_points(
    llm: ChatModel, header: str, sentences: list[str], chapters: dict[int, str | None]
) -> tuple[set[int], dict[int, int]]:
    """The sentences the model calls turning points, and by chapter number the sentence it names as truly opening that chapter."""
    numbers = {
        index: number
        for number, (index, title) in enumerate(chapters.items(), 1)
        if index > 0 and title
    }

    def judge(start: int, end: int) -> tuple[set[int], dict[int, int]]:
        marked = {
            index: number for index, number in numbers.items() if start <= index < end
        }
        lines: list[str] = []
        for index in range(
            max(0, start - WINDOW_CONTEXT), min(len(sentences), end + WINDOW_CONTEXT)
        ):
            if index in marked:
                lines.append(
                    f'<chapter id="CHAPTER{marked[index]}">{chapters[index]}</chapter>'
                )
            lines.append(f"[LINE{index + 1}] {sentences[index]}")
        system = TURNING_POINTS.format(first=f"[LINE{start + 1}]", last=f"[LINE{end}]")
        if marked:
            system += f"\n{CHAPTER_MARKERS}"
        user = f"{header}\n\n<lines>\n" + "\n".join(lines) + "\n</lines>"
        for _ in range(ATTEMPTS):
            reply = llm.invoke(system, user)
            openings = {
                int(number): int(line) - 1
                for number, line in CHAPTER_ID.findall(reply)
                if int(number) in marked.values()
            }
            named = {
                int(number) - 1
                for number in SENTENCE_ID.findall(CHAPTER_ID.sub("", reply))
            }
            if named or openings or "NONE" in reply:
                return {index for index in named if start <= index < end}, openings
        raise RewriteError("The model named neither turning points nor NONE")

    windows = run_batches(
        fixed_batches(len(sentences), SENTENCES_PER_WINDOW),
        judge,
        task="Finding paragraphs",
        unit="sentences",
        total=len(sentences),
    )
    turning_points = {index for named, _ in windows for index in named}
    openings = {
        number: line for _, answers in windows for number, line in answers.items()
    }
    return turning_points, openings


def _placed_chapters(
    chapters: dict[int, str | None], openings: dict[int, int], total: int
) -> dict[int, str | None]:
    """`openings` maps a chapter's 1-based position to the sentence named to open it.

    A title moves there only when it lands between the previous chapter's start and the next chapter's timestamp, so chapters stay in order.
    """
    marks = list(chapters)
    placed: dict[int, str | None] = {}
    previous = -1
    for number, index in enumerate(marks, 1):
        following = marks[number] if number < len(marks) else total
        named = openings.get(number, index)
        start = named if previous < named < following else index
        placed[start] = chapters[index]
        previous = start
    return placed


def _rewrite_paragraphs(
    llm: ChatModel,
    header: str,
    sentences: list[str],
    starts: list[int],
    language: str | None,
) -> list[str]:
    """A cleaned-up batch that fails its checks keeps its source text, while a translation raises RewriteError."""
    system = (
        TRANSLATE_ARTICLE.format(language=language) if language else CORRECT_ARTICLE
    )

    def rewrite(start: int, end: int) -> list[str]:
        bounds = [index for index in starts if start <= index < end]
        sources = [" ".join(sentences[a:b]) for a, b in pairwise([*bounds, end])]
        user = request(
            header, sentences, (start, end), sources, context=CONTEXT_SENTENCES
        )
        for _ in range(ATTEMPTS):
            texts = replies(llm.invoke(system, user), len(sources))
            if len(texts) < len(sources):
                continue
            rewritten = [texts[number] for number in range(1, len(sources) + 1)]
            ratio = sum(map(len, rewritten)) / sum(map(len, sources))
            if language or MIN_ARTICLE_RATIO <= ratio <= MAX_ARTICLE_RATIO:
                return rewritten
        if language:
            raise RewriteError("The model did not translate every paragraph")
        return sources

    batches = run_batches(
        _paragraph_batches(starts, len(sentences)),
        rewrite,
        task="Rewriting",
        unit="sentences",
        total=len(sentences),
    )
    return [paragraph for batch in batches for paragraph in batch]


def _paragraph_batches(starts: list[int], total: int) -> list[tuple[int, int]]:
    batches: list[tuple[int, int]] = []
    for start, end in pairwise([*starts, total]):
        if batches and end - batches[-1][0] <= SENTENCES_PER_BATCH:
            batches[-1] = (batches[-1][0], end)
        else:
            batches.append((start, end))
    return batches


def _translate_titles(
    llm: ChatModel, header: str, titles: dict[int, str | None], language: str
) -> dict[int, str | None]:
    """A chapter title the reply leaves out keeps its original wording."""
    named = {start: title for start, title in titles.items() if title}
    texts = replies(
        llm.invoke(
            TRANSLATE_TITLES.format(language=language),
            f"{header}\n\n{numbered(list(named.values()))}",
        ),
        len(named),
    )
    translated = {
        start: texts.get(number, title)
        for number, (start, title) in enumerate(named.items(), 1)
    }
    return {start: translated.get(start) for start in titles}
