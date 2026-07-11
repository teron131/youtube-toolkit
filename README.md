# YouTube Toolkit

Opinionated `yt-dlp` plus LanceDB toolkit for agents that need reusable YouTube search, transcripts, watchlists, and semantic retrieval.

The toolkit turns YouTube into local agent memory:

- `yt-dlp` owns the live YouTube edge: search, metadata, captions, recent channel uploads, watchlists, blacklists, and SponsorBlock-aware transcript cleanup
- local storage owns durable state, so agents can reuse transcripts, summaries, search metadata, channel rows, and embedding indexes across runs
- LanceDB powers semantic title/transcript retrieval for agent workflows that need chunk search, ranked videos, or RAG prep

## Positioning

This repo is not a generic YouTube wrapper and not only a summarizer. The core object is an agent-readable video document:

```text
YouTube URL -> yt-dlp metadata + transcript document -> local catalog + LanceDB indexes -> agent output
```

The toolkit intentionally ships a small working surface:

<table>
  <thead>
    <tr>
      <th>Module</th>
      <th>Submodule</th>
      <th>Args</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><code>video</code></td>
      <td><code>transcript</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Return a stored transcript first, then fetch captions or fall back to STT; persists the normalized transcript.</td>
    </tr>
    <tr>
      <td><code>metadata</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Fetch raw <code>yt-dlp</code> metadata JSON.</td>
    </tr>
    <tr>
      <td><code>search</code></td>
      <td><code>-</code></td>
      <td><code>"&lt;query&gt;"</code><br><code>[--source youtube|channel|db]</code><br><code>[--mode native|semantic]</code></td>
      <td>Search YouTube by default, or mix YouTube/channel/DB sources with local filters and semantic transcript lookup.</td>
    </tr>
    <tr>
      <td><code>channel</code></td>
      <td><code>videos</code></td>
      <td><code>&lt;channel&gt;</code><br><code>[--last N]</code><br><code>[--days N]</code></td>
      <td>List recent videos for one channel and persist channel/video rows.</td>
    </tr>
    <tr>
      <td rowspan="8"><code>watchlist</code></td>
      <td><code>add</code></td>
      <td><code>&lt;channel...&gt;</code></td>
      <td>Add one or more channels to the watched set.</td>
    </tr>
    <tr>
      <td><code>list</code></td>
      <td><code>-</code></td>
      <td>List watched channels.</td>
    </tr>
    <tr>
      <td><code>remove</code></td>
      <td><code>&lt;channel...&gt;</code></td>
      <td>Remove watchlist membership without deleting stored catalog data.</td>
    </tr>
    <tr>
      <td><code>sync</code></td>
      <td><code>[--last N]</code><br><code>[--days N]</code></td>
      <td>Refresh watched-channel videos, store transcripts, and auto-index embeddings when enabled.</td>
    </tr>
    <tr>
      <td><code>recent</code></td>
      <td><code>[channel...]</code><br><code>[--days N]</code></td>
      <td>Read recently uploaded watched videos from storage without mutating state.</td>
    </tr>
    <tr>
      <td><code>schedule install</code></td>
      <td><code>[--minutes N]</code><br><code>[--last N]</code><br><code>[--days N]</code><br><code>[--label NAME]</code><br><code>[--no-load]</code></td>
      <td>Install the macOS LaunchAgent for scheduled watchlist sync.</td>
    </tr>
    <tr>
      <td><code>schedule status</code></td>
      <td><code>[--label NAME]</code></td>
      <td>Show scheduled sync LaunchAgent status.</td>
    </tr>
    <tr>
      <td><code>schedule uninstall</code></td>
      <td><code>[--label NAME]</code></td>
      <td>Unload and remove the scheduled sync LaunchAgent.</td>
    </tr>
    <tr>
      <td rowspan="2"><code>blacklist</code></td>
      <td><code>add</code></td>
      <td><code>&lt;channel...&gt;</code></td>
      <td>Add channels excluded from future search results.</td>
    </tr>
    <tr>
      <td><code>remove</code></td>
      <td><code>&lt;channel...&gt;</code></td>
      <td>Remove blacklist membership without deleting stored catalog data.</td>
    </tr>
    <tr>
      <td><code>index</code></td>
      <td><code>-</code></td>
      <td><code>[video...]</code><br><code>[--include-without-transcript]</code></td>
      <td>Embed stored transcript text for semantic search and RAG prep.</td>
    </tr>
    <tr>
      <td><code>setup</code></td>
      <td><code>-</code></td>
      <td><code>[--embedding-*]</code><br><code>[--llm-*]</code><br><code>[--auto-embeddings|--no-auto-embeddings]</code><br><code>[--verify|--no-verify]</code></td>
      <td>Configure app-local embedding and LLM API settings.</td>
    </tr>
    <tr>
      <td><code>doctor</code></td>
      <td><code>-</code></td>
      <td><code>-</code></td>
      <td>Check <code>yt-dlp</code>, active database path, schema health, schedule health, and model API config.</td>
    </tr>
  </tbody>
</table>

Search sources are selected with repeated `--source youtube|channel|db` flags. Use `--source channel --channels @OpenAI --channels @Anthropic` for channel-scoped discovery, `--source db` for stored rows, and `--mode semantic` when the answer should come from transcript embeddings rather than title/metadata matching. Semantic mode returns chunks by default; add `--return video` to get ranked videos instead.

Storage lives in this checkout's `.cache/youtube.lancedb` during source-checkout usage. Installed CLI usage follows the platform data directory (`~/Library/Application Support/youtube-toolkit` on macOS, `~/.local/share/youtube-toolkit` on Linux, and Local AppData on Windows). Set `YOUTUBE_TOOLKIT_STORAGE_HOME` to move it.

`doctor` inspects the database schema without creating, migrating, or repairing the file. If the schema is stale or unexpected, it reports a warning and points to the future `youtube setup database` repair flow.

The schema source of truth is `youtube_toolkit/storage/schema.sql`; DB creation and doctor inspection both read that SQL file.

Model configuration lives alongside installed CLI data in the platform data directory. Bare `youtube setup` opens an Up/Down selector for embedding, LLM, settings, verification, and save. Flag-based usage stays scriptable: use `youtube setup --verify` to force a `Hello` check, `youtube setup --no-verify` to skip it, and `youtube setup --no-auto-embeddings` to keep transcript sync from indexing embeddings automatically. Environment variables override the file for one-off runs:

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

For `<channel...>` positional arguments, pass separate shell words: `@OpenAI @Anthropic UC123`. Do not use commas or `[a,b,c]` list syntax. For repeated options, write the option each time: `--channels @OpenAI --channels @Anthropic` or `--include-channels OpenAI --include-channels Anthropic`.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search "agent memory" --source db --mode semantic --limit 5
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
youtube index
youtube setup
```

## Agent Skill

The repo also includes a Markdown-only local skill at `skills/youtube/SKILL.md`.
