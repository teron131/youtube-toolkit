---
name: youtube
description: Use this repo-local YouTube toolkit for transcript-first YouTube research, search, watchlists, and local database-backed agent workflows.
---

# YouTube

Use the `youtube` CLI as the agent surface for YouTube work. It should fetch, persist, and print structured output; do not copy old one-off scripts.

## CLI

Prefer the installed command:

```bash
youtube <command>
```

Use `uv run` only when intentionally testing the checkout without reinstalling:

```bash
uv run youtube <command>
```

## Recipes

Video retrieval:

```bash
youtube video transcript <url-or-id>
youtube video transcript <url-or-id> --timestamps
youtube video transcript <url-or-id> --export
youtube video metadata <url-or-id>
```

- `<url-or-id>` is a YouTube watch URL or bare video ID.
- Metadata output contains normalized video facts instead of the raw yt-dlp payload.
- Transcript fetch is DB-first, then captions, then STT fallback; it persists the normalized transcript and returns one text string unless `--timestamps` is requested.
- `--export` follows the same retrieval route and writes metadata plus transcript to `<normalized-title>_<normalized-channel>.txt` in the working directory; add `--force` to replace an existing export.

Database:

```bash
youtube db view
youtube db get <url-or-id>
youtube db refresh <url-or-id>
youtube db edit <url-or-id> --summary "Corrected summary"
youtube db edit <url-or-id> --transcript-file transcript.txt
youtube db edit <url-or-id> --transcript-file -
youtube db edit <url-or-id> --clear summary --dry-run
youtube db delete <url-or-id> --dry-run
youtube db delete <url-or-id> --yes
```

- `db get` returns the complete locally stored record, including summary text.
- `db refresh` makes a network request through `yt-dlp` and is the only direct command that changes provider-owned title, upload date, duration, channel identity, view count, or like count. It preserves the stored transcript and summary.
- `db edit` changes only local agent-owned content through `--summary`, `--transcript-file`, or `--clear transcript|summary`. It never changes provider-owned metadata.
- Repeat `--clear FIELD` to clear stored agent content; valid field names are `transcript` and `summary`.
- Use `--transcript-file -` to read replacement transcript text from stdin.
- Edits return compact JSON change details and automatically clear stale summaries, transcript chunks, and embeddings.
- Use `--dry-run` before uncertain edits or deletions. Actual deletion requires `--yes` and preserves the channel record.

Search:

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search "agent memory" --source db --mode semantic --limit 5
youtube search "model evals" --source youtube --source db --mode semantic --return video --limit 5
youtube search "open source agents" --extra-query "LangGraph" --include-channels OpenAI --include-channels Anthropic
youtube search "open source agents" --exclude-channels shorts --exclude-channels clips --include-titles agents --include-titles tools --exclude-titles reaction --exclude-titles drama
```

- `--limit N` caps result count.
- `--source youtube|channel|db` selects candidate sources and can be repeated to mix them.
- `--source channel` searches videos from specified channels; repeat `--channels <channel>` for multiple channel targets.
- `--source db` searches stored local video rows.
- `<channel>` is a channel URL, handle like `@OpenAI`, or channel ID.
- `--mode semantic` uses independent title, transcript, chunk-index, and scoring stages. Fresh YouTube/channel searches prefer the DB option at each storage-capable stage and fall back stage-by-stage to memory. It returns chunks by default, or ranked videos with `--return video`.
- `--extra-query TEXT` adds another YouTube relevance query while keeping one result set.
- `--min-views N`, `--min-duration 3:00`, and `--max-duration 20:00` filter YouTube-style candidates.
- `--uploaded-after YYYY-MM-DD` and `--uploaded-within-days N` filter by upload date.
- `--sort relevance|recency|popularity` changes ranking; native YouTube search cannot use `recency`.
- `--include-channels`, `--exclude-channels`, `--include-titles`, and `--exclude-titles` are repeatable one-value filters; repeat the flag for multiple values.

Channels and watchlist:

```bash
youtube channel videos @OpenAI --last 5
youtube channel videos @OpenAI --days 7
youtube watchlist add @OpenAI @Anthropic
youtube watchlist list
youtube watchlist sync
youtube watchlist recent @OpenAI --days 3
youtube watchlist remove @Anthropic
youtube watchlist schedule install --minutes 720 --days 3
youtube watchlist schedule install --label com.example.youtube --no-load
youtube watchlist schedule status
youtube watchlist schedule uninstall
youtube blacklist add @SpamChannel UC123
youtube blacklist remove @SpamChannel
```

- `--last 5` caps channel listing results.
- `--days N` sets the channel listing, watchlist sync, recent, or schedule lookback window.
- `--minutes 720` sets the LaunchAgent interval for scheduled watchlist sync.
- `--label NAME` selects a non-default LaunchAgent label; `--no-load` writes the plist without loading it.
- `watchlist remove` and `blacklist remove` delete membership only; stored catalog data stays.
- Pass multiple channels as separate shell words: `@OpenAI @Anthropic UC123`.
- `watchlist recent` reads stored recent videos without mutating storage.

Maintenance:

```bash
youtube index
youtube index --include-without-transcript
youtube doctor
youtube setup
youtube setup --no-auto-embeddings --verify
youtube setup --embedding-api-key ... --embedding-base-url ... --embedding-model ... --embedding-dimensions 768
youtube setup --llm-api-key ... --llm-base-url ... --llm-model ... --no-verify
youtube setup codex
youtube setup codex --yes
```

- `index` with no videos indexes all stored videos that already have transcripts.
- `--include-without-transcript` also considers stored videos that do not have transcripts.
- Bare `setup` opens the interactive model config menu; setup flags save scriptable config.
- `--auto-embeddings` and `--no-auto-embeddings` control automatic transcript embedding after storage.
- `--embedding-*` configures embedding API key, base URL, model, and dimensions.
- `--llm-*` configures LLM API key, base URL, and model.
- `--verify` sends a small live check to configured model APIs after saving; `--no-verify` skips it.
- `setup codex` previews the active Codex config and writable root without writing.
- `setup codex --yes` creates that storage root, preserves existing TOML, and adds the path only when an equivalent writable root is not already configured.

## Rules

- Search defaults to YouTube; use repeated `--source youtube|channel|db` flags only when mixing sources is useful.
- `watchlist sync` discovers watched-channel videos, stores transcripts, and may auto-index embeddings.
- Use `watchlist schedule install|status|uninstall`; the hidden `scheduled-sync` subcommand is for launchd.
- Keep browser database inspection read-only; use `db refresh` for provider metadata and `db get|edit|delete` for agent-owned catalog work so storage invariants remain enforced.
- Summarize only when the user asks for analysis or a summary.

## Storage Gotcha

All CLI and source-checkout invocations use the operating system's standard user data directory unless `YOUTUBE_TOOLKIT_STORAGE_HOME` explicitly selects isolated storage. Run `youtube doctor` when storage behavior matters.

If a sandbox denies a write to the installed data directory, fresh YouTube/channel semantic search records the denial once and routes later stages directly to their in-memory options. Results report skipped stored blacklist behavior or semantic-stage degradation. For DB sources, watchlists, blacklists, indexes, and other stateful operations, run `youtube setup codex --yes` from a normal terminal and start a new Codex task. Do not set `YOUTUBE_TOOLKIT_STORAGE_HOME` as a permission workaround because that silently selects a separate database; use the override only when the user intentionally wants isolated storage.

The repository install flow is `uv run python scripts/install.py`; it replaces an existing tool environment rather than creating another install and idempotently configures the Codex writable root when Codex is available.
