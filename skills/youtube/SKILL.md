---
name: youtube
description: Use this repo-local YouTube toolkit for transcript-first research workflows: fetch normalized transcripts, search YouTube videos, inspect metadata, list recent channel uploads, manage watchlists/blacklists, and persist state in the local database.
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
- `watchlist remove <channel>`: stop watching a channel without deleting stored catalog data
- `watchlist sync`: refresh watched channels, mark new videos, and fetch missing transcripts
- `watchlist scheduled-sync`: run the lock-aware scheduler job body
- `watchlist schedule install|status|uninstall`: manage the macOS LaunchAgent for scheduled sync
- `watchlist new --channel <channel>` / `watchlist new --all`: return new watched videos as JSON and mark only returned rows seen
- `blacklist add <channel>`: exclude a channel from future search results
- `blacklist remove <channel>`: stop excluding a channel from future search results

Transcript language selection is hardcoded to English first, then common Chinese caption codes; caption lookup is hardcoded to automatic captions first, then manual subtitles. SponsorBlock-marked transcript chunks are always removed.

Audio downloads prefer M4A/MP4/AAC containers for xAI and ElevenLabs compatibility without conversion.

Speech-to-text uses `XAI_API_KEY`, hardcodes the English language hint, and normalizes provider output with `s2hk`; CLI and transcript workflows own SQLite persistence and generated `.cache/audio` cleanup.

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
- `watchlist schedule install --minutes N`; default `30`, with `--last 50 --days 2`

## Storage

Storage lives in this checkout's `.cache/youtube.sqlite3` during source-checkout usage. Installed CLI usage stores the database at `~/Library/Application Support/youtube-toolkit/youtube.sqlite3`. Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

Search automatically applies stored blacklist channels. Search, transcript, channel listing, and watchlist commands persist normalized channel/video rows for later workflows.

## Workflow

Treat the CLI as an agent operation surface. It should fetch, persist, and return structured command output rather than managing export formats.

Use summaries only when the user explicitly asks for summary or analysis. LLM transcript summarization is an internal DB workflow option, not the default transcript behavior.
