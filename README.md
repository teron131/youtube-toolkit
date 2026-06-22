# YouTube Toolkit

Transcript-first YouTube toolkit for local research, agent workflows, and future provider/MCP/API adapters.

This repo pulls the strongest parts of the existing Codex YouTube skill and the YouTube summarizer repo into a cleaner library-first shape:

- the skill's reliable local `yt-dlp` path for search, metadata, captions, recent channel uploads, watchlists, blacklists, SponsorBlock-aware transcript cleanup, and chaptered output
- the summarizer repo's provider-oriented posture, summary schema discipline, and deployment lessons, but without making MCP, Cloud Run, or LLM routing the core abstraction
- local artifact output as a first-class workflow, so agents can create durable Markdown/JSON files instead of only printing summaries into chat

## Positioning

This repo is not only a summarizer and not only a Codex skill. The core object is a normalized video document:

```text
YouTube URL -> metadata + timed transcript chunks + chapters -> durable state/artifacts -> optional summaries/adapters
```

The toolkit intentionally ships a small working surface:

- `youtube transcript <url-or-id>` prints or writes a transcript artifact
- `youtube audio <url-or-id>` downloads compact mono 16 kHz MP3 audio ready for STT models
- `youtube transcribe-audio <audio-path>` transcribes local audio with xAI STT, prints plain transcript text, and removes generated `.cache/audio` input
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
uv run youtube transcript JpSHyEIZ_bo --output artifacts/JpSHyEIZ_bo.md
uv run youtube audio JpSHyEIZ_bo
uv run youtube transcribe-audio .cache/audio/JpSHyEIZ_bo.mp3 --output .cache/audio/JpSHyEIZ_bo.txt
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
