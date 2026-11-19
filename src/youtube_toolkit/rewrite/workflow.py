"""Entry point for transcript rewrites, which picks one of three routes and saves the result beside the original transcript for later runs to reuse.

- timestamped captions: caption lines → numbered batches → replies aligned onto each line's own timing
- untimed speech: sentences → numbered batches → one untimed segment
- article: sentences → paragraph starts at turning points and at the sentence that truly opens each chapter → numbered batches of whole paragraphs → paragraphs under the chapter titles
"""

from __future__ import annotations

from dataclasses import asdict, replace

from ..artifacts import Cached, reuse
from ..config import LLMConfig
from ..llm import ChatModel
from ..schemas import TranscriptDocument, TranscriptRewrite, TranscriptSegment
from ..sentences import split_sentences
from .article import rewrite_article
from .requests import request_header
from .timestamped import rewrite_lines


def rewrite_transcript(
    original: Cached[TranscriptDocument],
    config: LLMConfig,
    *,
    language: str | None,
    refine: bool,
    article: bool,
    fresh: bool,
) -> Cached[TranscriptDocument]:
    """`language` is the target language, None for the transcript's own, and the original already in it comes back as it is unless `refine` asks for a cleaned-up copy.

    `language` reaches the model as written, so a name such as Japanese works but always counts as a translation.
    A translation raises RewriteError rather than pass untranslated text, while a cleaned-up copy keeps the original wherever a reply fails its checks.
    A rewrite saved earlier from the same original with the same settings and model comes back unless `fresh`.
    """
    document = original.value
    translated = language is not None and (
        _base_language(language) != _base_language(document.language)
    )
    if not translated and not refine:
        return original
    target = language if translated else None
    entry = reuse(
        original.path.parent,
        "_".join(
            [
                target or document.language,
                *(["refined"] if refine else []),
                "article" if article else "lines",
            ]
        ),
        {
            "original": original.path.stem,
            "language": language,
            "refine": refine,
            "article": article,
            "model": config.model,
        },
        fresh=fresh,
        compute=lambda: asdict(
            _rewrite(
                document, config, target=target, translated=translated, article=article
            )
        ),
    )
    return Cached(TranscriptDocument.from_dict(entry.value), entry.reused, entry.path)


def _rewrite(
    document: TranscriptDocument,
    config: LLMConfig,
    *,
    target: str | None,
    translated: bool,
    article: bool,
) -> TranscriptDocument:
    header = request_header(document)
    llm = ChatModel(config, reasoning_effort="low")
    if article:
        segments = [rewrite_article(llm, header, document, target)]
    elif document.timed:
        segments = rewrite_lines(llm, header, document.segments, target)
    else:
        segments = [_rewrite_untimed_speech(llm, header, document, target)]
    return replace(
        document,
        language=target or document.language,
        segments=segments,
        rewrite=TranscriptRewrite(config.model, translated),
    )


def _rewrite_untimed_speech(
    llm: ChatModel, header: str, document: TranscriptDocument, language: str | None
) -> TranscriptSegment:
    sentences = rewrite_lines(llm, header, split_sentences(document.segments), language)
    return replace(
        document.segments[0], text=" ".join(sentence.text for sentence in sentences)
    )


def _base_language(language: str) -> str:
    """The primary subtag, so en-US and en-orig both read as en and zh-Hant matches zh-TW."""
    return language.split("-")[0].lower()
