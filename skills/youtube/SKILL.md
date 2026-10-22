---
name: youtube
description: Use the YouTube toolkit to search videos, list channel uploads, search comment text, read transcripts and metadata, and extract visual context, with request archives kept in the working directory.
---

# YouTube

Run the installed `youtube` CLI from the task's directory so its request archives stay with the research: every retrieval fetches fresh data and saves a copy under `.cache/youtube-toolkit/` there, beside earlier ones. To test changes to the toolkit itself, run `uv run youtube` in its checkout instead. Commands print JSON to stdout, except `doctor`, and report errors on stderr.

Pass a video as its 11-character ID or as a URL; watch, `youtu.be`, Shorts, embed, and live URLs all work. Always quote a URL: unquoted, zsh rejects its `?` with `no matches found`, and any shell cuts the command off at an `&`.

## Retrieval

```bash
youtube metadata "https://www.youtube.com/watch?v=VIDEO_ID"
youtube transcript VIDEO_ID
youtube transcript VIDEO_ID --timestamps
youtube transcript VIDEO_ID --export
youtube vision VIDEO_ID --max-frames 6
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
youtube doctor
```

`metadata` returns normalized video facts, including the description, the thumbnail URL, and `chapters`, each with a `title`, `start_seconds`, and an `end_seconds` that may be null. `chapters` is an empty list when the video has none.

`transcript` prints the whole text as one JSON string, not an object, so `jq -r .` turns it into plain text. With `--timestamps` it prints a list of chunks, each with `start_seconds`, `end_seconds`, and `text`.

Manual English or Chinese subtitles come first, starting with whichever matches the video's language, then the original language's automatic captions. Machine-translated tracks are never used, Chinese text is converted to Taiwan Traditional, and caption spans overlapping SponsorBlock segments are removed. When a video has no usable captions, speech-to-text runs if a route is set up; a failed caption download stays an error. A speech-to-text transcript has no timing, so `--timestamps` returns it as one chunk that starts and ends at zero.

`--export` writes metadata and transcript to `<normalized-title>_<normalized-channel>.txt` in the working directory and prints its path. Add `--timestamps` for a timed export, or `--force` to replace an existing one; `--force` works only with `--export`. Long text fields in stdout JSON are cut at 400,000 characters with no marker, so when a long video's complete text matters, read the export or the archived transcript.

`channel` lists recent uploads, without search filters. `--last N` sets how many (10 by default). `--days N` keeps uploads after a UTC date cutoff and adds `upload_date` to each entry; it fetches each video's metadata and drops videos it cannot date, and `--last` then sizes the listing checked against the cutoff, not the number returned.

## Comments

Use `comments` to find a keyword or phrase in a video's comments, passing a query when the task asks about a topic, claim, or audience reaction.

```bash
youtube comments VIDEO_ID "pricing"
youtube comments VIDEO_ID "training data" --limit 250
youtube comments VIDEO_ID "pricing" | jq '.comments[] | select((.like_count // 0) >= 10)'
```

It fetches a sample of top-level comments in YouTube's Top order, without replies: 100 by default, or 1-1000 with `--limit`, and an out-of-range limit fails before contacting YouTube. The query filters that sample as one case-insensitive literal substring, like `rg -i -F`, so regex and OR expressions are not interpreted. A query never fetches more comments to find extra matches; raise `--limit` for wider coverage. Omit the query, or pass `""`, only when the whole sample is wanted.

Matches alternate between the next comment in Top order and the next by descending likes, without repeats; equal like counts keep Top order, and unavailable counts sort last and stay `null`. The output holds only `title`, `url`, and `comments`, each comment with only `text` and `like_count`, so filter further with `jq` on those fields. The unfiltered sample is archived in Top order under `.cache/youtube-toolkit/comments/` in the same shape. An empty `comments` list means no match or no comments in the sample, not that no comment on the video mentions the topic.

## Visual context

Use `vision` when pictures add useful context. It needs the vision extra and `ffmpeg`, accepts videos up to two hours, and caps the temporary download at 500 MiB, deleting it after processing. `--max-frames` takes 1-96 and defaults to six; repetitive footage can yield fewer. The output gives frame timestamps and absolute paths to collages under `.cache/youtube-toolkit/vision/<datetime>-<video-id>/`, each holding up to six screenshots in chronological order, two columns by three rows.

Each item in `frames` has its timestamp, the `transcript` passage it was chosen for, and raw `visual_scores`. Without usable timed captions, `selection_mode` is `visual`, each frame's `transcript` is null, and `transcript_warning` says why. Vision never calls a model API or speech-to-text. The scores are measurements, not importance, so read each collage alongside the transcript.

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search --channels @OpenAI --limit 5
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --channels @OpenAI --mode semantic --return video
youtube search "open source agents" "LangGraph agents" --exclude-titles reaction
```

`search` covers all of YouTube unless `--channels` names channels whose uploads to search; repeat it for several. Handles, URLs, and IDs reference a channel directly, while a plain name costs a YouTube lookup, so prefer an exact reference when one is known.

- Native YouTube-wide search requires a query and sorts by relevance. Several queries are interleaved in YouTube's relevance order without duplicate videos. `--sort popularity` orders by views; `--sort recency` is rejected here.
- Native channel search sorts by recency and allows no query. With queries, a video's title must contain at least one of them, case-insensitively.
- Defaults are 10 results (30 for channel searches), at least 10,000 views, and a duration from 3 minutes to 8 hours. Override them when the task calls for different limits.

`--limit` caps results. Duration filters take seconds, `MM:SS`, or `HH:MM:SS`. Date filters take `--uploaded-after YYYY-MM-DD` and `--uploaded-within-days N`, use the later cutoff when both are given, and count relative days in UTC. `--include-channels`, `--exclude-channels`, `--include-titles`, and `--exclude-titles` filter by case-insensitive substring and repeat for several terms: every include term must match, and any exclude term rejects a video.

Channel listings carry no upload dates, so recency sorting and date filters fetch each video's metadata and take longer; other filters fetch metadata only when a field they need is missing. Filters and unavailable metadata can leave fewer results than `--limit`.

`--mode semantic` requires a query: it shortlists videos by title, then ranks transcript passages by their best similarity to any query. It needs `EMBEDDING_API_KEY`, saved with `youtube setup --embeddings` or set in the environment; optional settings are `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and `EMBEDDING_CONCURRENCY` (default 16, maximum 50). It returns passages by default, or videos ranked by their best passage with `--return video`; `--return chunk` works only in semantic mode. A passage's `chunk_index` is not a timestamp; run `transcript --timestamps` for timing.

Semantic search reads captions only and never runs speech-to-text. Partial caption or embedding failures are summarized in `warnings`, which may name only some of the skipped items. If no passage can be embedded, the command fails; an empty result list means no passage passed the similarity threshold. Embeddings and rankings last only for the current search.

## Speech-to-text and setup

`transcript` falls back to speech-to-text through a cloud provider when its key is set, otherwise through a local Hugging Face model when its dependencies and `ffmpeg` are installed. A failed cloud request is an error, not a reason to try the local model, and with neither route available the no-captions error stands. `STT_PROVIDER` defaults to `xai` (`XAI_API_KEY`); the others are `gemini` (`GEMINI_API_KEY`), `vercel` (`AI_GATEWAY_API_KEY`), `openai` (`OPENAI_API_KEY`), and `elevenlabs` (`ELEVENLABS_API_KEY`).

The local model is a Hugging Face model ID for the `transformers` `automatic-speech-recognition` pipeline: the one named by `STT_LOCAL_MODEL`, or `nvidia/parakeet-tdt-0.6b-v3` by default, downloaded on first use. The default covers only 25 European languages, so treat its output for other languages as unreliable. An unknown `STT_PROVIDER`, or a local model that fails to load, falls back to the default with a warning on stderr.

Install from the toolkit's checkout with `uv run python scripts/install.py`, adding `--vision` for visual extraction, `--stt` for local speech-to-text, or both; `ffmpeg` is installed separately for either.

`youtube setup` interactively saves embedding and speech-to-text settings for this tool only; `--embeddings` or `--stt` limits the prompts to one capability, and both flags select both. Keys go through hidden prompts, never command-line flags, so let the user enter secrets directly and never echo them into commands or chat. At a prompt, Enter keeps the saved value, `-` removes a saved override, and cancelling saves nothing. Setup prints its prompts on stderr and ends with JSON naming the settings file and the saved setting names, never their values; it makes no API calls or model downloads.

Settings live in a `.env` file under `~/Library/Application Support/youtube-toolkit/` on macOS, `${XDG_CONFIG_HOME:-~/.config}/youtube-toolkit/` on Linux, or `%APPDATA%/youtube-toolkit/` on Windows, readable only by the current user on POSIX systems; a set `XDG_CONFIG_HOME` moves it on macOS too. It is a plain local file, not the OS keychain, and setup changes no shell startup files or `PATH`. A nonblank environment variable overrides a saved setting, which overrides the default. File values are read literally, without expanding `${VARIABLE}` references, and are not exported to the process environment.

`youtube doctor` reports readiness and the artifact path without creating files, loading a model, or contacting any service. `youtube stt-setup` reports both speech-to-text routes and, when local dependencies and `ffmpeg` are present, downloads the local model if needed and runs a short recording through it. `youtube stt-setup --cloud-test` also sends a two-second recording to the cloud provider, a real request the provider bills, so run it only when asked to verify the key.

## Artifacts

Fresh yt-dlp metadata, search listings, channel listings, and normalized transcript documents are each saved as a JSON file named with a UTC datetime and the video ID, query slug, or channel identifier, holding `request`, `fetched_at`, and `payload`. Comment archives also have timestamped names but hold only the bare sample described under Comments. Search archives hold the retrieved listings, not the final rankings or embeddings. Saved copies are never reused or expired automatically, and transcript exports and archived documents keep the complete text past the stdout cap.

Retrieval is directory-local: there is no database, collection management, watchlist, scheduled sync, or persistent embedding index. The calling agent owns synthesis; summarize when the user asks for analysis or a summary.
