---
name: youtube
description: Use this repo-local YouTube toolkit for transcript-first research workflows: fetch normalized transcripts, search YouTube videos, inspect metadata, list recent channel uploads, manage watchlists/blacklists, and persist state in the local database.
---

# YouTube

Use this skill for YouTube transcript retrieval, metadata inspection, filtered video search, recent channel uploads, watchlist/blacklist state, and durable local database state.

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

- `doctor`: return JSON with the local `yt-dlp` version and active database path
- `transcript <url-or-id>`: fetch and store a normalized transcript, then print JSON
- `inspect <url-or-id>`: print full `yt-dlp` metadata JSON
- `search "<query>"`: search YouTube videos with local metadata filters
- `channel-videos <channel>`: list recent videos from one YouTube channel URL, handle, or channel ID
- `watchlist add <channel...>`: persist one or more channels for later sync
- `watchlist remove <channel...>`: stop watching one or more channels without deleting stored catalog data
- `watchlist sync`: refresh watched channels, mark new videos, and fetch missing transcripts
- `watchlist schedule install|status|uninstall`: manage the macOS LaunchAgent for scheduled sync
- `watchlist new --channel <channel>` / `watchlist new --all`: return new watched videos as JSON and mark only returned rows seen; repeat `--channel` for multiple channels
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
- `watchlist new --channel <channel>` can be repeated as `--channel @OpenAI --channel @Anthropic`
- `watchlist schedule install --minutes N`; default `30`, with `--last 50 --days 2`

## Storage

Storage lives in this checkout's `.cache/youtube.sqlite3` during source-checkout usage. Installed CLI usage stores the database at `~/Library/Application Support/youtube-toolkit/youtube.sqlite3`. Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

`doctor` returns the active database path. If `uv run youtube doctor` and `youtube doctor` disagree, the checkout and installed command are using different storage roots.

Search automatically applies stored blacklist channels. Search, transcript, channel listing, and watchlist commands persist normalized channel/video rows for later workflows.

## Workflow

Treat the CLI as an agent operation surface. It should fetch, persist, and return structured command output rather than managing export formats.

Use summaries only when the user explicitly asks for summary or analysis. LLM transcript summarization is an internal DB workflow option, not the default transcript behavior.
