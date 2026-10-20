---
name: youtube
description: Use the YouTube toolkit to search videos and comment text, read transcripts and metadata, and extract visual context with directory-local request archives.
---

# YouTube

Use the installed `youtube` CLI from the current task directory so its request archives stay with the research. Retrievals fetch fresh data and preserve earlier copies under `.cache/youtube-toolkit/` there. Use `uv run youtube` when intentionally testing this checkout. Commands return JSON except `doctor`; command errors go to stderr.

## Retrieval

```bash
youtube metadata <url-or-id>
youtube transcript <url-or-id>
youtube transcript <url-or-id> --timestamps
youtube transcript <url-or-id> --export
youtube vision <url-or-id> --max-frames 6
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
youtube doctor
```

Metadata returns normalized video facts, including the description, thumbnail URL, and chapter titles with `start_seconds` and optional `end_seconds`. Chapters are an empty list when unavailable.

Transcript returns one JSON text string, or timed chunks with `--timestamps`. Manual English or Chinese subtitles are preferred, with the video's language family tried first, then original-language automatic captions. Machine-translated tracks are excluded, Chinese text is converted to Taiwan Traditional, and caption spans overlapping SponsorBlock segments are removed. Missing usable captions trigger speech-to-text when a route is set up; caption download failures remain errors. An STT transcript is untimed, so `--timestamps` returns it as one chunk with zero start and end.

`--export` writes metadata and transcript to `<normalized-title>_<normalized-channel>.txt` in the working directory and returns the path. Add `--timestamps` for a timed export or `--force` to replace an existing export; `--force` requires `--export`. Long text fields in JSON stdout are capped at 400,000 characters, so use the export or archived transcript when complete long-form text is needed.

`channel` lists recent uploads without search filters. `--last N` selects a count (10 by default), while `--days N` uses a UTC date cutoff and adds `upload_date` to each entry. The day window fetches per-video metadata and omits videos it cannot date; `--last` then controls the initial listing size rather than the final result count.

## Search comments

Use `comments` to find a keyword or phrase in a video's comments; pass a query when the task asks about a topic, claim, or audience reaction.

```bash
youtube comments <url-or-id> "pricing"
youtube comments <url-or-id> "training data" --limit 250
youtube comments <url-or-id> "pricing" | jq '.comments[] | select((.like_count // 0) >= 10)'
```

This fetches up to 100 top-level comments in YouTube's Top order, excludes replies, and matches comment text as a case-insensitive literal substring, like `rg -i -F`.
The optional query is one keyword or phrase; regex and OR expressions are not interpreted.
`--limit` accepts 1-1000, defaults to 100, and bounds the sample before filtering; out-of-range values fail before contacting YouTube.
A query never causes more fetching to fill a match quota.
Omit the query or pass `""` only when intentionally requesting the entire fetched sample.
After filtering, results alternate between the next unseen comment in Top order and the next unseen comment ranked by descending likes, deduplicating IDs internally.
Within the like-count ranking, ties preserve Top order and unavailable counts sort last; unavailable counts remain `null` in the output.
Interleaving changes ordering within the sample, not retrieval coverage or network request count.
The output contains only `title`, `url`, and `comments`, with only `text` and `like_count` per comment.
Use `jq` for additional filtering on those fields.
The unfiltered sample is archived in original Top order under `.cache/youtube-toolkit/comments/` using the same lean JSON fields and no metadata wrapper.
An empty `comments` array means no matches or no available comments, not that the topic is absent from every comment on the video.

## Visual context

Use vision when pictures add useful context. It requires the vision extra and `ffmpeg`, accepts videos up to two hours, and limits the temporary video download to 500 MiB. It returns frame timestamps and absolute collage paths under `.cache/youtube-toolkit/vision/<datetime>-<video-id>/`. `--max-frames` accepts 1-96 and defaults to six; repetitive content can yield fewer. The downloaded video is removed after processing.

Each collage holds up to six screenshots in chronological order, two columns by three rows. Each item in `frames` has its timestamp, the `transcript` passage it was chosen for, and raw `visual_scores`. Without usable timed captions, `selection_mode` is `visual`, each frame's `transcript` is null, and `transcript_warning` says why. Vision never uses a model API or STT. The scores are measurements, not importance; read the collage alongside the transcript.

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search --channels @OpenAI --limit 5
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --channels @OpenAI --mode semantic --return video
youtube search "open source agents" "LangGraph agents" --exclude-titles reaction
```

Search covers YouTube unless `--channels` names channels whose uploads should be searched. Repeat it for several channels. Handles, URLs, and IDs are direct references; a channel name triggers a YouTube lookup, so prefer a known exact reference.

- Native YouTube search requires a query and defaults to relevance. Several queries are interleaved in YouTube's relevance order, with duplicate videos removed. `--sort popularity` orders by views; native YouTube search rejects `--sort recency`.
- Native channel search defaults to recency and allows no query. When queries are supplied, a video title must contain any one of them, case-insensitively.
- CLI defaults are 10 results (30 for channel searches), at least 10,000 views, and duration from 3 minutes to 8 hours. Override them when the task calls for different limits.

`--limit` caps results. Duration filters accept seconds, `MM:SS`, or `HH:MM:SS`; date filters accept `--uploaded-after YYYY-MM-DD` and `--uploaded-within-days N`, using the later cutoff when combined. Relative dates use UTC. Repeat `--include-channels`, `--exclude-channels`, `--include-titles`, and `--exclude-titles` for substring filters: all include terms must match, and any exclude term rejects a candidate. Matching is case-insensitive.

Channel listings omit upload dates, so recency sorting and date filters fetch per-video metadata and take longer. Other filters fetch metadata only when a required field is missing. Filters and unavailable metadata can leave fewer results than `--limit`.

Semantic search requires a query, shortlists titles, and ranks transcript passages by their best similarity to any query. Configure `EMBEDDING_API_KEY` with `youtube setup --embeddings` or through the environment; optional settings are `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and `EMBEDDING_CONCURRENCY` (default 16, maximum 50). Results are passages by default, or videos ranked by their best passage with `--return video`. `--return chunk` requires semantic mode. A passage's `chunk_index` is not a timestamp; retrieve `transcript --timestamps` for timing.

Semantic search reads captions only and never invokes STT. It summarizes partial caption or embedding failures in `warnings`; these summaries may name only a subset of skipped items. If no transcript passage can be embedded, the command fails; an empty result list means no passage passed the similarity threshold. Duplicate texts share an embedding request within a stage. Embeddings and rankings are kept only for the current search.

## Speech-to-text and setup

The transcript command uses a cloud provider when its key is set, otherwise a local Hugging Face model when its dependencies and `ffmpeg` are installed. Cloud request failures propagate; they do not select the local route. With neither route available, the no-captions error is returned. `STT_PROVIDER` defaults to `xai` (`XAI_API_KEY`); other choices are `gemini` (`GEMINI_API_KEY`), `vercel` (`AI_GATEWAY_API_KEY`), `openai` (`OPENAI_API_KEY`), and `elevenlabs` (`ELEVENLABS_API_KEY`).

The local model is a Hugging Face ASR model ID for the `transformers` `automatic-speech-recognition` pipeline: `nvidia/parakeet-tdt-0.6b-v3` by default, or the model named by `STT_LOCAL_MODEL`. It downloads on first use. The default model covers 25 European languages only; treat its output for other languages as unreliable. An unknown `STT_PROVIDER`, or a configured local model that cannot be loaded, falls back to the default with a warning on stderr.

Install with `uv run python scripts/install.py`; add `--vision` for visual extraction dependencies, `--stt` for local speech-to-text, or both. Install `ffmpeg` separately for these optional capabilities.

`youtube setup` interactively saves embedding and STT settings for this application; `--embeddings` or `--stt` limits the prompts, and both flags select both capabilities.
Keys use hidden prompts rather than command-line flags, so let the user enter secrets directly and never echo them into commands or chat.
Enter keeps an existing saved value; `-` removes a saved override; cancellation saves nothing.
Setup saves settings in `.env` under `~/Library/Application Support/youtube-toolkit/` on macOS, `${XDG_CONFIG_HOME:-~/.config}/youtube-toolkit/` on Linux, or `%APPDATA%/youtube-toolkit/` on Windows, with access restricted to the current user on POSIX systems.
`XDG_CONFIG_HOME` overrides the configuration directory on Linux and macOS.
This is a local settings file, not OS keychain storage, and setup changes no shell startup files or `PATH` settings.
Nonblank environment values override saved settings, then existing defaults apply.
File values are read literally without expanding `${VARIABLE}` references or exporting them to the process environment.
Prompts go to stderr and final JSON reports the configuration path and saved setting names without values; setup makes no API calls or model downloads.

`youtube doctor` reports readiness and the artifact path without creating files, loading a model, or contacting services. `youtube stt-setup` reports both STT routes and, when local dependencies and `ffmpeg` are available, downloads the local model if needed and runs a short recording through it. `youtube stt-setup --cloud-test` also sends a two-second recording to the cloud provider, a real request the provider bills, so run it only when asked to verify the key.

## Artifacts

Fresh yt-dlp metadata, search listings, channel listings, and normalized transcript documents are saved as separate JSON artifacts. Filenames contain a UTC datetime and video ID, query slug, or channel identifier; the JSON records `request`, `fetched_at`, and `payload`. Comment archives use timestamped filenames but contain only `title`, `url`, and `comments`, with `text` and `like_count` per comment, in unfiltered Top order. Search archives contain the retrieved listings; final rankings and embeddings live only in the current invocation. Saved copies are never automatically reused or expired. Transcript exports and archived documents retain complete text beyond the stdout cap.

Retrieval is directory-local; there is no database, collection administration, watchlist, scheduled sync, or persistent embedding index. The calling agent owns synthesis; summarize when the user requests analysis or a summary.
