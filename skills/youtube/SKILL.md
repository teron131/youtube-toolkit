---
name: youtube
description: Use the YouTube toolkit for directory-cached search, transcripts, metadata, and visual context.
---

# YouTube

Use the installed `youtube` CLI from the current task directory. Retrieval is cached under `.cache/youtube-toolkit/` there. Use `uv run youtube` when intentionally testing this checkout.

## Retrieval

```bash
youtube video metadata <url-or-id>
youtube video transcript <url-or-id>
youtube video transcript <url-or-id> --timestamps
youtube video transcript <url-or-id> --export
youtube video vision <url-or-id>
youtube channel videos @OpenAI --last 5
youtube channel videos @OpenAI --days 7
youtube doctor
```

Metadata returns normalized video facts, including the description, thumbnail URL, and chapter titles with `start_seconds` and optional `end_seconds`. Chapters are an empty list when unavailable. Transcript retrieval uses cached normalized documents, then captions, then speech-to-text fallback. It returns one text string unless `--timestamps` requests timed chunks. `--export` writes metadata and transcript to `<normalized-title>_<normalized-channel>.txt` in the working directory; add `--force` to replace an existing export.

Use vision when visual cues materially help. It requires the vision extra and `ffmpeg`, samples a bounded video-only download, and returns timestamps and absolute collage paths under `.cache/youtube-toolkit/vision/<video-id>/`. Sampled frames can miss important moments; read the transcript alongside them.

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --source channel --channels @OpenAI --mode semantic --return video
youtube search "open source agents" --extra-query LangGraph --exclude-titles reaction
```

YouTube is the default source. CLI defaults require at least 10,000 views and a duration from 3 minutes to 8 hours; override these with the existing filter flags when the task calls for different limits. Repeat `--source youtube|channel` to mix candidate sources. Channel search requires repeated `--channels` values, accepting channel URLs, handles, or IDs. `--extra-query` adds YouTube relevance queries to the same candidate pool. `--limit` caps results; `--min-views`, `--min-duration`, `--max-duration`, `--uploaded-after`, and `--uploaded-within-days` constrain candidates. Repeat `--include-channels`, `--exclude-channels`, `--include-titles`, and `--exclude-titles` for multiple filters. Native YouTube search does not support recency sorting. Channel search fetches per-video metadata because channel listings omit view counts and upload dates, so it takes several seconds longer.

Semantic search uses environment-configured embeddings to rank titles and transcript passages during the invocation. Set `EMBEDDING_API_KEY`; optional configuration is `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSIONS`. Results are chunks by default, or ranked videos with `--return video`. No persistent embedding index is maintained.

## Cache and installation

Metadata, search, and channel listing caches expire after 15 minutes. Transcript documents are cached by video and retrieval options until removed. Delete the disposable cache to refresh retrieval. Run `youtube doctor` to see the effective cache path and optional capabilities without creating files.

Install with `uv run python scripts/install.py`; add `--vision` for visual extraction dependencies. There is no database, collection administration, watchlist, scheduled sync, or persistent model setup. Existing user data and old schedules remain untouched. The calling agent owns synthesis; summarize when the user requests analysis or a summary.
