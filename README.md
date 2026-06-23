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
- `youtube audio <url-or-id>` downloads provider-compatible audio for STT models without conversion
- `youtube transcribe-audio <audio-path>` transcribes local audio with xAI STT, applies `s2hk` Chinese conversion, stores the transcript in SQLite, returns JSON, and removes generated `.cache/audio` input
- `youtube search "<query>"` searches videos with local metadata filters
- `youtube inspect <url-or-id>` prints metadata JSON
- `youtube channel-videos <channel>` lists recent channel uploads
- `youtube watchlist add <channel>` and `youtube watchlist sync` persist channel workflows
- `youtube blacklist add <channel>` excludes channels from future search results
- `youtube doctor` checks the runtime can import `yt-dlp`

Storage lives in `.cache/youtube.sqlite3` by default. Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

## Setup

```bash
uv sync
uv run youtube doctor
```

## Examples

```bash
uv run youtube transcript JpSHyEIZ_bo
audio_path=$(uv run youtube audio JpSHyEIZ_bo)
uv run youtube transcribe-audio "$audio_path"
uv run youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
uv run youtube inspect https://www.youtube.com/watch?v=JpSHyEIZ_bo
uv run youtube channel-videos @OpenAI --last 5
uv run youtube watchlist add @OpenAI
```

## Codex Skill

The repo also includes a Markdown-only local skill at `skills/youtube/SKILL.md`.

## Next Slots

- Add transcript provider adapters for Scrape Creators and Supadata.
- Add summary provider adapters for Gemini URL-native summaries and OpenRouter transcript summaries.
- Add MCP/FastAPI adapters after the library contracts settle.
