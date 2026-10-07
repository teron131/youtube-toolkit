# YouTube Toolkit

A directory-cached YouTube CLI for agents: search, read transcripts, inspect metadata, and extract visual context through yt-dlp.

Run it from the project or task directory where you want reusable retrieval files. The agent owns analysis and summaries; the toolkit owns fetching and normalized output.

## Install

Requires Python 3.13+ and uv.

```bash
uv run python scripts/install.py
```

For visual extraction, install its optional dependencies and install `ffmpeg` separately:

```bash
uv run python scripts/install.py --vision
```

The installer replaces the installed `youtube` tool. To run the checkout directly, use `uv run youtube`.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube channel @OpenAI --last 5
youtube metadata <url-or-id>
youtube transcript <url-or-id>
youtube transcript <url-or-id> --timestamps
youtube transcript <url-or-id> --export
youtube vision <url-or-id>
youtube doctor
```

Commands return JSON, except `doctor`, which prints a short readiness report. `metadata` includes the description, thumbnail URL, and chapters with titles, `start_seconds`, and optional `end_seconds`; videos without chapter metadata return an empty chapter list. Transcript output is a text string by default, or timed chunks with `--timestamps`. `--export` writes metadata and transcript to `<normalized-title>_<normalized-channel>.txt` in the working directory; use `--force` to replace an existing export.

CLI search defaults to at least 10,000 views and a duration between 3 minutes and 8 hours, with 10 results for YouTube or 30 for channel-only searches. These defaults live in `config.py`; the existing search flags override them per invocation.

Search sources are selected with repeated `--source youtube|channel` flags. YouTube is the default. Channel search requires `--channels`, accepting handles, URLs, or channel IDs. Repeat `--extra-query` to expand YouTube discovery. Repeat `--include-channels`, `--exclude-channels`, `--include-titles`, or `--exclude-titles` to filter candidates. View, duration, and upload-date filters apply before ranking. Channel listings omit view counts and upload dates, so channel search fetches per-video metadata for those filters and for recency sorting; expect it to take several seconds longer than YouTube search.

## Directory cache

Retrieval files live under `.cache/youtube-toolkit/` in the current working directory. Metadata, search, and channel listings expire after 15 minutes. Normalized transcripts are reused by video and retrieval options until you delete their cache files. Transcript caches also retain this metadata; older caches refresh missing metadata without repeating caption retrieval or speech-to-text. Visual collages live under `.cache/youtube-toolkit/vision/<video-id>/`.

The cache is disposable: deleting it causes the next retrieval to fetch again. There is no database, persistent catalog, watchlist, blacklist, scheduled sync, or stored model configuration. Existing databases, configuration files, and installed schedules from earlier versions are left untouched; an existing schedule should be removed separately if you previously installed one.

## Semantic search

Native search needs no model configuration. Optional semantic search ranks candidate titles and transcript passages using an embedding API during the current invocation:

```bash
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --source channel --channels @OpenAI --mode semantic --return video
```

Set `EMBEDDING_API_KEY` in the environment. Optional settings are `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSIONS`. Semantic results are chunks by default; `--return video` returns ranked videos. No durable embedding index is maintained.

## Agent skill

The repository skill is [skills/youtube/SKILL.md](skills/youtube/SKILL.md).
