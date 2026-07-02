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

- `youtube transcript <url-or-id>` fetches and stores a transcript, then prints JSON
- `youtube search "<query>"` searches videos with local metadata filters
- `youtube inspect <url-or-id>` prints metadata JSON
- `youtube channel-videos <channel>` lists recent uploads for one channel
- `youtube watchlist add <channel...>`, `youtube watchlist list`, `youtube watchlist remove <channel...>`, and `youtube watchlist sync` persist channel workflows
- `youtube watchlist new --channel <channel>` accepts repeated `--channel` flags; `youtube watchlist new --all` returns all new watched videos
- `youtube watchlist schedule install|status|uninstall` manages its macOS LaunchAgent
- `youtube blacklist add <channel...>` and `youtube blacklist remove <channel...>` manage channels excluded from future search results
- `youtube doctor` returns JSON with the `yt-dlp` version and active database path

Storage lives in this checkout's `.cache/youtube.sqlite3` during source-checkout usage. Installed CLI usage stores the database at `~/Library/Application Support/youtube-toolkit/youtube.sqlite3`. Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

The default watchlist schedule runs every 12 hours, scans uploads from the last 3 days, and does not cap the number of videos fetched per watched channel.

## Setup

```bash
uv sync
uv tool install .
youtube doctor
```

Use `uv tool install --force .` when refreshing an existing install.
Use `uv run youtube ...` only when intentionally testing the checkout without reinstalling. Normal usage below assumes the installed `youtube` command.

For `<channel...>` arguments, pass separate shell words: `@OpenAI @Anthropic UC123`. Do not use commas or `[a,b,c]` list syntax. For repeated options, write the option each time: `--channel @OpenAI --channel @Anthropic`.

## Examples

```bash
youtube transcript JpSHyEIZ_bo
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube inspect https://www.youtube.com/watch?v=JpSHyEIZ_bo
youtube channel-videos @OpenAI --last 5
youtube watchlist add @OpenAI @Anthropic
youtube watchlist list
youtube watchlist schedule install
youtube watchlist new --channel @OpenAI --channel @Anthropic
youtube watchlist new --all
youtube watchlist remove @OpenAI @Anthropic
youtube blacklist add @SpamChannel UC123
```

## Codex Skill

The repo also includes a Markdown-only local skill at `skills/youtube/SKILL.md`.

## Next Slots

- Add transcript provider adapters for Scrape Creators and Supadata.
- Add summary provider adapters for Gemini URL-native summaries and OpenRouter transcript summaries.
- Add MCP/FastAPI adapters after the library contracts settle.
