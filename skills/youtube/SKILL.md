---
name: youtube
description: |
  Use this repo-local YouTube toolkit for transcript-first research workflows: fetch normalized transcripts, search YouTube videos, inspect metadata, list recent channel uploads, manage watchlists/blacklists, and persist state in the local database.
---

# YouTube

Use this skill for YouTube transcript retrieval, metadata inspection, filtered video search, recent channel uploads, watchlist/blacklist state, and durable local database state.

This repo is the toolkit source of truth. Prefer the package CLI over copying scripts from older skill installs.

## CLI

Run commands from the repo root:

```bash
uv run youtube <command>
```

Commands:

- `doctor`: check that the local runtime can import `yt-dlp`
- `transcript <url-or-id>`: fetch and store a normalized transcript, then print JSON
- `audio <url-or-id>`: download provider-compatible audio for STT models without conversion
- `transcribe-audio <audio-path>`: transcribe local audio with xAI STT, store it, and print the stored video JSON payload
- `inspect <url-or-id>`: print full `yt-dlp` metadata JSON
- `search "<query>"`: search YouTube videos with local metadata filters
- `channel-videos <channel>`: list recent videos from a YouTube channel URL, handle, or channel ID
- `watchlist add <channel>`: persist a channel for later sync
- `watchlist sync`: refresh watched channels and fetch missing transcripts
- `blacklist add <channel>`: exclude a channel from future search results

Transcript language selection is hardcoded to English first, then common Chinese caption codes; caption lookup is hardcoded to automatic captions first, then manual subtitles. SponsorBlock-marked transcript chunks are always removed.

Audio downloads prefer M4A/MP4/AAC containers for xAI and ElevenLabs compatibility without conversion.

Speech-to-text uses `XAI_API_KEY`, hardcodes the English language hint, stores transcripts in SQLite, returns the stored video payload as JSON, and removes generated `.cache/audio` input after successful transcription.

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

## Storage

Storage lives in `.cache/youtube.sqlite3` by default, resolved from `YOUTUBE_TOOLKIT_STORAGE_HOME`, then legacy `YOUTUBE_TOOLKIT_STATE_HOME` or `YOUTUBE_MEMORY_HOME`, then the current Git repo root, then the package source root.

Search automatically applies stored blacklist channels. Search, transcript, channel listing, and watchlist commands persist normalized channel/video rows for later workflows.

## Workflow

Treat the CLI as an agent operation surface. It should fetch, persist, and return structured command output rather than managing export formats.

Use summaries only when the user explicitly asks for summary or analysis. LLM transcript summarization is an internal DB workflow option, not the default transcript behavior.
