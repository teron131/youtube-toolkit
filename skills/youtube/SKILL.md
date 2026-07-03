---
name: youtube
description: Use this repo-local YouTube toolkit for transcript-first research workflows: fetch normalized transcripts, search YouTube videos, fetch metadata, list recent channel videos, manage watchlists/blacklists, and persist state in the local database.
---

# YouTube

Use this skill for YouTube transcript retrieval, metadata inspection, YouTube search via `yt-dlp` with local filters, recent channel videos, watchlist/blacklist state, and durable local database state.

This repo is the toolkit source of truth. Prefer the package CLI over copying scripts from older skill installs.

## CLI

Prefer the installed command for normal use:

```bash
youtube <command>
```

Use `uv run` only when intentionally testing the checkout without reinstalling:

```bash
uv run youtube <command>
```

Commands:

- `doctor`: print a readable status report with the local `yt-dlp` version, active database path, database schema health, schedule health, and model API config health
- `setup`: write app-local model API keys, base URLs, and model IDs, then print a readable setup report
- `search "<query>"`: query YouTube through `yt-dlp`, then apply local filters and blacklist rules
- `video transcript <url-or-id>`: fetch and store a normalized transcript, then print JSON
- `video metadata <url-or-id>`: print full `yt-dlp` metadata JSON
- `channel videos <channel>`: list recent videos from one YouTube channel URL, handle, or channel ID
- `watchlist add <channel...>`: persist one or more channels for later sync
- `watchlist remove <channel...>`: stop watching one or more channels without deleting stored catalog data
- `watchlist sync`: refresh watched channels, record discovered videos, and fetch missing transcripts
- `watchlist schedule install|status|uninstall`: manage the macOS LaunchAgent for scheduled sync
- `watchlist recent [channel...] --days N`: return objectively recent watched videos from the local DB without mutating storage
- `blacklist add <channel...>`: exclude one or more channels from future search results
- `blacklist remove <channel...>`: stop excluding one or more channels from future search results

Transcript language selection is hardcoded to English first, then common Chinese caption codes; caption lookup is hardcoded to automatic captions first, then manual subtitles. SponsorBlock-marked transcript chunks are always removed.

Internal STT audio downloads prefer M4A/MP4/AAC containers for xAI and ElevenLabs compatibility without conversion.

Speech-to-text uses `XAI_API_KEY`, hardcodes the English language hint, and normalizes provider output with `s2hk`; the transcript workflow owns SQLite persistence and generated `.cache/audio` cleanup.

Useful search options:

- `--limit N`
- `--sort relevance|views`
- `--min-views N`; default `10000`
- `--min-duration SECONDS_OR_HH:MM:SS`; default `3:00`
- `--max-duration SECONDS_OR_HH:MM:SS`; default `8:00:00`
- `--include-channel TEXT` / `--exclude-channel TEXT`
- `--include-title TEXT` / `--exclude-title TEXT`
Useful channel/watchlist options:

- `--last N`; default `10`
- `--days N`
- `watchlist add/remove` and `blacklist add/remove` accept multiple channel URLs, handles, or channel IDs as space-separated arguments: `@OpenAI @Anthropic UC123`
- `watchlist recent` accepts optional watched-channel filters as space-separated arguments: `watchlist recent @OpenAI @Anthropic`
- `watchlist schedule install --minutes N`; default `720`, with no `--last` cap and `--days 3`

## Storage

Storage lives in this checkout's `.cache/youtube.sqlite3` during source-checkout usage. Installed CLI usage stores the database at `~/Library/Application Support/youtube-toolkit/youtube.sqlite3`. Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

`doctor` prints the active database path. If `uv run youtube doctor` and `youtube doctor` disagree, the checkout and installed command are using different storage roots.

`doctor` inspects database schema health without creating, migrating, or repairing the DB. Stale schema reports include a repair hint for the future `youtube setup database` flow.

The schema source of truth is `youtube_toolkit/storage/schema.sql`; DB creation and doctor inspection both read that SQL file.

Model configuration lives in `~/Library/Application Support/youtube-toolkit/config.env`. Use `youtube setup` to write `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`; it can optionally verify the config by sending `Hello` to both model APIs. Use `--verify` to force the check or `--no-verify` to skip it. Environment variables with those names override the app-local file for one-off runs.

YouTube search via `yt-dlp` automatically applies stored blacklist channels and persists returned video metadata for later workflows. Transcript, channel listing, and watchlist commands also persist normalized channel/video rows.

## Workflow

Treat the CLI as an agent operation surface. It should fetch, persist, and return structured command output rather than managing export formats.

Use summaries only when the user explicitly asks for summary or analysis. LLM transcript summarization is an internal DB workflow option, not the default transcript behavior.
