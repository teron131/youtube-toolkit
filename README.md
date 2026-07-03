# YouTube Toolkit

Transcript-first YouTube toolkit for local research, agent workflows, and future provider/MCP/API adapters.

This repo pulls the strongest parts of the existing Codex YouTube skill and the YouTube summarizer repo into a cleaner library-first shape:

- the skill's reliable local `yt-dlp` path for search, metadata, captions, recent channel uploads, watchlists, blacklists, SponsorBlock-aware transcript cleanup, and chaptered output
- the summarizer repo's provider-oriented posture, summary schema discipline, and deployment lessons, but without making MCP, Cloud Run, or LLM routing the core abstraction
- local database state as the first-class workflow, so agents can reuse transcripts, summaries, search metadata, and embeddings

## Positioning

This repo is not only a summarizer and not only a Codex skill. The core object is a normalized video document:

```text
YouTube URL -> metadata + timed transcript chunks + chapters -> durable state -> optional summaries/adapters
```

The toolkit intentionally ships a small working surface:

- `youtube search "<query>"` queries YouTube through `yt-dlp`, then applies local filters and blacklist rules
- `youtube video transcript <url-or-id>` fetches and stores a transcript, then prints JSON
- `youtube video metadata <url-or-id>` prints metadata JSON
- `youtube channel videos <channel>` lists recent videos for one channel
- `youtube watchlist add <channel...>`, `youtube watchlist list`, `youtube watchlist remove <channel...>`, and `youtube watchlist sync` persist channel workflows; sync also auto-indexes transcript embeddings when enabled
- `youtube watchlist recent --days 3` returns recently uploaded videos from watched channels without mutating storage; pass channels after `recent` to narrow the scope
- `youtube index [video...]` embeds stored transcript text for semantic search and RAG prep
- `youtube watchlist schedule install|status|uninstall` manages its macOS LaunchAgent
- `youtube blacklist add <channel...>` and `youtube blacklist remove <channel...>` manage channels excluded from future search results
- `youtube setup` writes app-local model API keys, base URLs, and model IDs, then prints a readable setup report
- `youtube doctor` prints a readable status report with the `yt-dlp` version, active database path, database schema health, watchlist schedule health, and model API config health

Storage lives in this checkout's `.cache/youtube.sqlite3` during source-checkout usage. Installed CLI usage stores the database at `~/Library/Application Support/youtube-toolkit/youtube.sqlite3`. Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

`doctor` inspects the database schema without creating, migrating, or repairing the file. If the schema is stale or unexpected, it reports a warning and points to the future `youtube setup database` repair flow.

The schema source of truth is `youtube_toolkit/storage/schema.sql`; DB creation and doctor inspection both read that SQL file.

Model configuration lives in `~/Library/Application Support/youtube-toolkit/config.env`. Bare `youtube setup` opens an Up/Down selector for embedding, LLM, settings, verification, and save. Flag-based usage stays scriptable: use `youtube setup --verify` to force a `Hello` check, `youtube setup --no-verify` to skip it, and `youtube setup --no-auto-embeddings` to keep transcript sync from indexing embeddings automatically. Environment variables override the file for one-off runs:

```env
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSIONS=768
EMBEDDING_AUTO_INDEX=true
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
```

The default watchlist schedule runs every 12 hours, scans uploads from the last 3 days, and does not cap the number of videos fetched per watched channel.

## Setup

```bash
uv sync
uv tool install .
youtube setup
youtube doctor
```

Use `uv tool install --force .` when refreshing an existing install.
Use `uv run youtube ...` only when intentionally testing the checkout without reinstalling. Normal usage below assumes the installed `youtube` command.

For `<channel...>` arguments, pass separate shell words: `@OpenAI @Anthropic UC123`. Do not use commas or `[a,b,c]` list syntax. For repeated options, write the option each time: `--include-channel OpenAI --include-channel Anthropic`.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube video transcript JpSHyEIZ_bo
youtube video metadata https://www.youtube.com/watch?v=JpSHyEIZ_bo
youtube channel videos @OpenAI --last 5
youtube watchlist add @OpenAI @Anthropic
youtube watchlist list
youtube watchlist schedule install
youtube watchlist recent --days 3
youtube watchlist recent @OpenAI @Anthropic --days 3
youtube watchlist remove @OpenAI @Anthropic
youtube blacklist add @SpamChannel UC123
youtube setup
```

## Codex Skill

The repo also includes a Markdown-only local skill at `skills/youtube/SKILL.md`.

## Next Slots

- Add transcript provider adapters for Scrape Creators and Supadata.
- Add summary provider adapters for Gemini URL-native summaries and OpenRouter transcript summaries.
- Add MCP/FastAPI adapters after the library contracts settle.
