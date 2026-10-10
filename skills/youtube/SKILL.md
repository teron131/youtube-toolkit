---
name: youtube
description: Use the YouTube toolkit for search with directory-local request archives, transcripts, metadata, and visual context.
---

# YouTube

Use the installed `youtube` CLI from the current task directory. Every retrieval fetches fresh data and saves a timestamped copy under `.cache/youtube-toolkit/` there, preserving earlier copies. Use `uv run youtube` when intentionally testing this checkout.

## Retrieval

```bash
youtube metadata <url-or-id>
youtube transcript <url-or-id>
youtube transcript <url-or-id> --timestamps
youtube transcript <url-or-id> --export
youtube vision <url-or-id> [--max-frames 6]
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
youtube doctor
```

Metadata returns normalized video facts, including the description, thumbnail URL, and chapter titles with `start_seconds` and optional `end_seconds`. Chapters are an empty list when unavailable. Transcript returns one text string, or timed chunks with `--timestamps`. The text is English or Chinese when the video has those subtitles, otherwise the video's own language; videos without captions are transcribed by speech-to-text. `--export` writes metadata and transcript to `<normalized-title>_<normalized-channel>.txt` in the working directory; add `--force` to replace an existing export. Channel lists the latest videos with `--last N`, or those uploaded in the last N days with `--days N`, where each entry includes `upload_date`.

Use vision when pictures add useful context. It requires the vision extra and `ffmpeg`, and returns frame timestamps and absolute collage paths under `.cache/youtube-toolkit/vision/<datetime>-<video-id>/`. `--max-frames` accepts 1-96 and defaults to six; repetitive content can yield fewer.

Each collage holds up to six screenshots in chronological order, two columns by three rows. Each item in `frames` has its timestamp, the `transcript` passage it was chosen for, and raw `visual_scores`. Without usable captions, `selection_mode` is `visual`, `transcript` is null, and `transcript_warning` says why. The scores are measurements, not importance; read the collage alongside the transcript.

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --source channel --channels @OpenAI --mode semantic --return video
youtube search "open source agents" "LangGraph agents" --exclude-titles reaction
```

YouTube is the default source. CLI defaults require at least 10,000 views and a duration from 3 minutes to 8 hours; override these with the existing filter flags when the task calls for different limits. Repeat `--source youtube|channel` to mix candidate sources. Channel search requires repeated `--channels` values, accepting channel URLs, handles, or IDs. Give several queries to get one list that alternates between them, each in YouTube's own relevance order. `--limit` caps results; `--min-views`, `--min-duration`, `--max-duration`, `--uploaded-after`, and `--uploaded-within-days` constrain candidates. Repeat `--include-channels`, `--exclude-channels`, `--include-titles`, and `--exclude-titles` for multiple filters. Native YouTube search does not support recency sorting. Channel search fetches per-video metadata because channel listings omit view counts and upload dates, so it takes several seconds longer.

Semantic search uses environment-configured embeddings to rank titles and transcript passages during the invocation. Set `EMBEDDING_API_KEY`; optional configuration is `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSIONS`. Results are chunks by default, or ranked videos with `--return video`. It reads captions only, so videos without captions are skipped and named in `warnings`. No persistent embedding index is maintained.

## Artifacts and installation

Metadata, search results, channel listings, and transcripts are saved as separate JSON artifacts for every fresh retrieval. Filenames contain a UTC datetime and video ID, query slug, or channel identifier; the JSON records request details, `fetched_at`, and the response payload. Saved copies are never automatically reused or expired. Delete artifacts when they are no longer needed. Run `youtube doctor` to see the effective artifact path and optional capabilities without creating files.

Install with `uv run python scripts/install.py`; add `--vision` for visual extraction dependencies. There is no database, collection administration, watchlist, scheduled sync, or persistent model setup. Existing user data and old schedules remain untouched. The calling agent owns synthesis; summarize when the user requests analysis or a summary.
