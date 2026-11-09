"""Transcript rewrites through the LLM, a translation or a cleaned-up copy: the entry point picks one of three situations, each a chain of the package's steps, and archives the result.

- timestamped captions: caption lines → numbered batches → replies aligned onto each line's own timing
- untimed speech: sentences → numbered batches → one untimed chunk
- article: sentences → paragraph starts at turning points and at the sentence that truly opens each chapter → numbered batches of whole paragraphs → paragraphs under the chapter titles
"""

from __future__ import annotations

from dataclasses import asdict, replace

from ..artifacts import write_artifact
from ..config import LLMConfig
from ..llm import ChatModel
from ..schemas import TranscriptChunk, TranscriptDocument, TranscriptRewrite
from .article import rewrite_article
from .requests import request_header
from .sentences import split_sentences
from .timestamped import rewrite_lines


def rewrite_transcript(
    document: TranscriptDocument,
    config: LLMConfig,
    *,
    language: str | None,
    refine: bool,
    article: bool,
) -> TranscriptDocument:
    """`language` is the target language, None for the transcript's own, and a transcript already in it comes back unchanged unless `refine` asks for a cleaned-up copy.

    Timed output keeps every caption line with its own timestamps, and an untimed transcript is rewritten sentence by sentence and comes back as one untimed chunk; an article is one chunk of paragraphs spanning the transcript.
    A translation fails as a whole when a request keeps returning no text, so untranslated text never passes as translated, while a cleaned-up copy keeps the original wherever a reply fails its checks.
    `language` reaches the model as written, so a code, a name, or a misspelling all work, and the output keeps that language's script: Taiwan Traditional conversion would corrupt Japanese kanji.
    """
    translated = language is not None and (
        _base_language(language) != _base_language(document.language)
    )
    if not translated and not refine:
        return document
    target = language if translated else None
    header = request_header(document)
    # The default reasoning effort tripled a gpt-6-luna translation batch's time with the same wording.
    llm = ChatModel(config, reasoning_effort="low")
    if article:
        chunks = [rewrite_article(llm, header, document, target)]
    elif document.timed:
        chunks = rewrite_lines(llm, header, document.chunks, target)
    else:
        chunks = [_rewrite_untimed_speech(llm, header, document, target)]
    result = replace(
        document,
        language=target or document.language,
        chunks=chunks,
        rewrite=TranscriptRewrite(config.model, translated),
    )
    request = {
        "video_id": document.metadata.video_id,
        "language": language,
        "refine": refine,
        "article": article,
        "model": config.model,
    }
    write_artifact("rewrites", document.metadata.video_id, request, asdict(result))
    return result


def _rewrite_untimed_speech(
    llm: ChatModel, header: str, document: TranscriptDocument, language: str | None
) -> TranscriptChunk:
    sentences = rewrite_lines(llm, header, split_sentences(document.chunks), language)
    return replace(
        document.chunks[0], text=" ".join(sentence.text for sentence in sentences)
    )


def _base_language(language: str) -> str:
    """The primary subtag, so en-US and en-orig both read as en and zh-Hant matches zh-TW; a name such as Japanese never matches a code and is always translated."""
    return language.split("-")[0].lower()
