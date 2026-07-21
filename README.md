# YouTube Toolkit

Opinionated `yt-dlp` plus LanceDB toolkit for agents that need reusable YouTube search, transcripts, visual context, watchlists, and semantic retrieval.

The toolkit turns YouTube into local agent memory:

- `yt-dlp` owns the live YouTube edge: search, metadata, captions, recent channel uploads, watchlists, blacklists, and SponsorBlock-aware transcript cleanup
- local storage owns durable state, so agents can reuse transcripts, summaries, search metadata, channel rows, and embedding indexes across runs
- fresh YouTube/channel semantic search uses DB-first title, transcript, index, and scoring stages with independent in-memory fallbacks

## Positioning

This repo is not a generic YouTube wrapper and not only a summarizer. The core object is an agent-readable video document:

```text
YouTube URL -> yt-dlp metadata -> staged title, transcript, index, and scoring options -> agent output
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
      <td rowspan="3"><code>video</code></td>
      <td><code>transcript</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Return a stored transcript first, then fetch captions or fall back to STT; persists the normalized transcript.</td>
    </tr>
    <tr>
      <td><code>metadata</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Fetch normalized video metadata JSON.</td>
    </tr>
    <tr>
      <td><code>vision</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Generate bounded timestamped 2×3 visual-context collages and return their absolute paths.</td>
    </tr>
    <tr>
      <td><code>search</code></td>
      <td><code>-</code></td>
      <td><code>"&lt;query&gt;"</code><br><code>[--source youtube|channel|db]</code><br><code>[--mode native|semantic]</code></td>
      <td>Search YouTube by default, use DB-first semantic stages with in-memory fallback, or include DB sources for stored retrieval.</td>
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
      <td rowspan="5"><code>db</code></td>
      <td><code>view</code></td>
      <td><code>[--port N]</code><br><code>[--no-open]</code></td>
      <td>Open a live, read-only browser for tables, rows, schemas, indexes, and statistics.</td>
    </tr>
    <tr>
      <td><code>get</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Return one complete locally stored video record as JSON.</td>
    </tr>
    <tr>
      <td><code>refresh</code></td>
      <td><code>&lt;url-or-id&gt;</code></td>
      <td>Refresh provider-owned metadata such as title, duration, upload date, channel identity, and engagement counts from YouTube.</td>
    </tr>
    <tr>
      <td><code>edit</code></td>
      <td><code>&lt;url-or-id&gt;</code><br><code>[--summary TEXT]</code><br><code>[--transcript-file PATH|-]</code><br><code>[--clear transcript|summary]</code><br><code>[--dry-run]</code></td>
      <td>Edit agent-owned transcript or summary content and automatically invalidate stale derived data.</td>
    </tr>
    <tr>
      <td><code>delete</code></td>
      <td><code>&lt;url-or-id&gt;</code><br><code>--yes|--dry-run</code></td>
      <td>Delete one stored video and its watchlist, transcript-chunk, and embedding rows.</td>
    </tr>
    <tr>
      <td rowspan="2"><code>setup</code></td>
      <td><code>-</code></td>
      <td><code>[--embedding-*]</code><br><code>[--llm-*]</code><br><code>[--auto-embeddings|--no-auto-embeddings]</code><br><code>[--verify|--no-verify]</code></td>
      <td>Configure app-local embedding and LLM API settings.</td>
    </tr>
    <tr>
      <td><code>codex</code></td>
      <td><code>--yes</code></td>
      <td>Preview or idempotently grant the active storage directory to Codex workspace-write sandboxes.</td>
    </tr>
    <tr>
      <td><code>doctor</code></td>
      <td><code>-</code></td>
      <td><code>-</code></td>
      <td>Check <code>yt-dlp</code>, active database path, schema health, schedule health, and model API config.</td>
    </tr>
  </tbody>
</table>

Search sources are selected with repeated `--source youtube|channel|db` flags. Use `--source channel --channels @OpenAI --channels @Anthropic` for channel-scoped discovery, `--source db` for stored rows, and `--mode semantic` when the answer should come from transcript embeddings rather than title/metadata matching. Semantic mode returns chunks by default; add `--return video` to get ranked videos instead. Fresh YouTube/channel semantic search advances through independent title, transcript, chunk-index, scoring, and result stages. Each storage-capable stage prefers LanceDB when available and falls back to its in-memory option without switching the whole workflow.

Stateful invocations use the platform data directory (`~/Library/Application Support/youtube-toolkit` on macOS, `~/.local/share/youtube-toolkit` on Linux, and Local AppData on Windows). Set `YOUTUBE_TOOLKIT_STORAGE_HOME` only when intentionally selecting an isolated database.

Sandboxed callers must authorize write access to that active installed-data directory before running commands that persist metadata, transcripts, watchlist state, or indexes. Fresh YouTube/channel semantic search remembers a DB permission failure and sends later stages directly to their in-memory options; results label any skipped stored blacklist or degraded semantic stage. DB sources and other stateful commands still fail clearly. Do not redirect `YOUTUBE_TOOLKIT_STORAGE_HOME` merely to bypass a permission error: that selects a separate database. Run `youtube doctor` to confirm the active path, then grant access and retry the original stateful command.

`youtube setup codex` previews the exact Codex config and writable-root paths. `youtube setup codex --yes` creates the active storage root, appends it to `sandbox_workspace_write.writable_roots` only when an equivalent path is not already present, preserves the rest of the TOML file, and requires a new Codex task before the permission takes effect.

`youtube db view` serves the active store on a loopback-only URL and opens it in the default browser. The viewer reads the live database directly and exposes no mutation endpoints; it does not create, migrate, repair, or optimize storage. Use `--no-open` to leave browser navigation to another process, choose a fixed loopback port with `--port`, and press Ctrl-C to stop the server.

Stored-video administration stays under `youtube db` rather than the retrieval-focused `youtube video` group or the read-only viewer. Field ownership is explicit: `youtube db refresh` fetches provider-owned facts such as title, upload date, duration, channel identity, view count, and like count from YouTube, while `youtube db edit` can change only agent-owned transcript and summary content. Edits return a compact JSON change report without echoing full text, and transcript changes clear stale summaries and transcript-derived embeddings. Use `--dry-run` before an uncertain content edit or deletion; destructive deletion additionally requires `--yes`.

`doctor` inspects the database schema without creating, migrating, or repairing the file. If the schema is stale or unexpected, it reports a warning and points to the future `youtube setup database` repair flow.

The schema source of truth is `youtube_toolkit/storage/connection.py`; database creation and health inspection use those Python schema definitions.

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

Runtime requirements:

- `yt-dlp` is installed with the Python package and owns YouTube metadata, captions, search, and media acquisition.
- `ffmpeg` must be available on `PATH`; video vision uses it for frame analysis and decoding. On macOS, install it with `brew install ffmpeg`.

Install or refresh the system command and configure Codex in one idempotent step:

```bash
uv run python scripts/install.py
```

The installer builds a wheel, replaces the existing `uv tool` environment instead of creating a duplicate, runs `youtube setup codex --yes` when Codex is installed, and finishes with `youtube doctor`.

For checkout-only development:

```bash
uv sync
youtube setup
youtube doctor
```

Use `uv run youtube ...` only when intentionally testing the checkout without reinstalling. Normal usage below assumes the installed `youtube` command.

For `<channel...>` positional arguments, pass separate shell words: `@OpenAI @Anthropic UC123`. Do not use commas or `[a,b,c]` list syntax. For repeated options, write the option each time: `--channels @OpenAI --channels @Anthropic` or `--include-channels OpenAI --include-channels Anthropic`.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search "agent memory" --source db --mode semantic --limit 5
youtube video transcript JpSHyEIZ_bo
youtube video transcript JpSHyEIZ_bo --timestamps
youtube video transcript JpSHyEIZ_bo --export
youtube video metadata https://www.youtube.com/watch?v=JpSHyEIZ_bo
youtube video vision https://www.youtube.com/watch?v=JpSHyEIZ_bo
youtube db get JpSHyEIZ_bo
youtube db refresh JpSHyEIZ_bo
youtube db edit JpSHyEIZ_bo --summary "Concise corrected summary"
youtube db edit JpSHyEIZ_bo --transcript-file transcript.txt --clear summary
youtube db delete JpSHyEIZ_bo --dry-run
youtube db delete JpSHyEIZ_bo --yes
youtube channel videos @OpenAI --last 5
youtube watchlist add @OpenAI @Anthropic
youtube watchlist list
youtube watchlist schedule install
youtube watchlist recent --days 3
youtube watchlist recent @OpenAI @Anthropic --days 3
youtube watchlist remove @OpenAI @Anthropic
youtube blacklist add @SpamChannel UC123
youtube index
youtube db view
youtube setup
```

Transcript exports use `<normalized-title>_<normalized-channel>.txt` in the working directory, omitting the channel suffix when unavailable. Add `--force` to replace an existing export.

Vision extraction requires `ffmpeg`. It downloads one temporary video-only stream at up to 720p, samples at one frame per second, combines scene changes with optional viewer-replay peaks and stationary frames, rejects transient blur, removes perceptual duplicates, and writes at most 96 selected frames as timestamped 2×3 JPEG pages under `.cache/vision/<video-id>/`. Each page contains up to six frames downscaled with Lanczos to 640×360 and uses optimized progressive 4:2:2 JPEG quality 80. The JSON result includes timestamps and absolute collage paths; temporary downloaded media is removed after processing.

## Agent Skill

The repo also includes a Markdown-only local skill at `skills/youtube/SKILL.md`.
