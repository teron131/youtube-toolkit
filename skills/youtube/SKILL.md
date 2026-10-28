---
name: youtube
description: Use the YouTube toolkit to search videos, list channel uploads, search comment text, read transcripts and metadata, and extract visual context, with request archives kept in the working directory.
---

# YouTube

Run the installed `youtube` CLI from the task's directory: every retrieval fetches fresh data and archives a copy under `.cache/youtube-toolkit/` there. To test changes to the toolkit itself, run `uv run youtube` in its checkout. Commands print JSON to stdout, except `doctor`, and errors to stderr.

Pass a video as its 11-character ID or any YouTube URL: watch, `youtu.be`, Shorts, embed, or live. Always quote URLs; zsh rejects an unquoted `?`, and any shell cuts the command at `&`.

## Retrieval

```bash
youtube metadata "https://www.youtube.com/watch?v=VIDEO_ID"
youtube transcript VIDEO_ID
youtube transcript VIDEO_ID --timestamps
youtube transcript VIDEO_ID --export
youtube vision VIDEO_ID --max-frames 24
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
youtube doctor
```

`metadata` returns video facts, including the description, thumbnail URL, and `chapters`, each with `title`, `start_seconds`, and a nullable `end_seconds`; the list is empty when there are none.

`transcript` prints the text as one JSON string, so `jq -r .` gives plain text; `--timestamps` prints chunks with `start_seconds`, `end_seconds`, and `text`. Manual English or Chinese subtitles come first, starting with the video's language, then original-language automatic captions. Machine translations are never used, Chinese becomes Taiwan Traditional, and SponsorBlock spans are removed. Without usable captions, speech-to-text runs if configured; a failed caption download stays an error.

`--export` writes metadata and text to `<normalized-title>_<normalized-channel>.txt` in the working directory and prints its path; add `--timestamps` for timing or `--force` to overwrite. Stdout text stops at 400,000 characters without a marker, so read the export or archive when a long video's full text matters.

`channel` lists recent uploads: `--last N`, 10 by default, or `--days N`, a UTC cutoff that fetches each video's metadata, adds `upload_date`, and drops videos it cannot date; `--last` then sizes the listing checked.

## Comments

Use `comments` to find a keyword or phrase in a video's comments, passing a query when the task asks about a topic, claim, or audience reaction.

```bash
youtube comments VIDEO_ID "pricing"
youtube comments VIDEO_ID "training data" --limit 250
youtube comments VIDEO_ID "pricing" | jq '.comments[] | select((.like_count // 0) >= 10)'
```

It samples top-level comments in YouTube's Top order, without replies: 100 by default, 1–1000 with `--limit`. The query is one case-insensitive literal substring, like `rg -i -F`, with no regex or OR. It never fetches more to find matches, so raise `--limit` for coverage; omit the query only when the whole sample is wanted. Matches alternate between Top order and most likes. Output holds only `title`, `url`, and `comments` with `text` and `like_count`, `null` when unavailable. An empty list means no match in the sample, not on the whole video. The unfiltered sample is archived under `.cache/youtube-toolkit/comments/`.

## Visual context

Use `vision` when pictures add context. It needs the vision extra and `ffmpeg`, accepts videos up to two hours, and deletes its download, capped at 500 MiB, afterwards. It prints frame timestamps and absolute paths to collages under `.cache/youtube-toolkit/vision/<datetime>-<video-id>/`, six 960×540 frames per page in time order. `--max-frames` takes 1–96, default 96, and never shrinks the candidate pool or embedding work.

With `GEMINI_API_KEY`, or an `EMBEDDING_API_KEY` whose settings point at `gemini-embedding-2` on `generativelanguage.googleapis.com`, and a timed transcript, Gemini Embedding 2 ranks the video's keyframes against its keywords, sentences, and paragraphs. A frame matches only text spoken from five seconds before to two seconds after it, so silent stretches yield no ranked frames. Picks skip frames within `--min-gap`, default 10 seconds, or with image-embedding cosine 0.96 or more to an earlier pick, so the result can be shorter than `--max-frames`. `--min-similarity`, default 0.30, is a raw-cosine floor that is never lowered to fill the count. The transcript is the one `transcript` returns, so a video without captions runs speech-to-text.

Without a key or a timed transcript, or with `--local-only`, vision picks the keyframe nearest the middle of each time slot of at least 12 seconds, skipping picks within the gap or repeating the previous one. `selection_mode`, `semantic` or `visual`, and the warnings say which ran and why. A rejected key fails within seconds, before the download, with Google's reason code.

Each frame has its timestamp, raw `visual_scores`, and, for ranked picks, `relevance` with rank, RRF score, and matched texts with raw and relative cosine. These are measurements, not confidence, so read the collages alongside the transcript. `--no-auto-clean` keeps the video and candidates and reports them as `media_dir`.

Expect about 20 seconds of CPU for a 30-minute video. Network time dominates: the 720p download, about 40–150 MB, and when ranking, the candidate upload, about 25–70 MB, which takes half a minute on a fast connection and several minutes on a slow one.

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search --channels @OpenAI --limit 5
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --channels @OpenAI --mode semantic --return video
youtube search "open source agents" "LangGraph agents" --exclude-titles reaction
```

`search` covers all of YouTube unless `--channels` names channels to search, repeated for several. Prefer handles, URLs, or IDs; a plain name costs a YouTube lookup.

YouTube-wide search needs a query and keeps YouTube's relevance order, interleaving several queries without duplicates; `--sort popularity` orders by views, and `--sort recency` is rejected. Channel search sorts by recency, and with queries a title must contain one of them. Defaults are 10 results, 30 for channels, at least 10,000 views, and a length from 3 minutes to 8 hours; override them when the task needs other limits.

Durations take seconds, `MM:SS`, or `HH:MM:SS`. `--uploaded-after YYYY-MM-DD` and `--uploaded-within-days N` count in UTC, and the later cutoff wins. `--include-channels`, `--exclude-channels`, `--include-titles`, and `--exclude-titles` match case-insensitive substrings and repeat: every include must match, and any exclude rejects. Date filters and recency sorting fetch each video's metadata, so they are slower, and filters can leave fewer results than `--limit`.

`--mode semantic` needs a query and `EMBEDDING_API_KEY`; `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and `EMBEDDING_CONCURRENCY`, at most 50, are optional. It shortlists videos by title and ranks caption passages by their best similarity to any query, returning passages, or videos with `--return video`; `--return chunk` works only in semantic mode. `chunk_index` is not a timestamp. It never runs speech-to-text. Partial failures appear in `warnings`; it fails only when no passage can be embedded, and an empty list means nothing passed the threshold.

## Speech-to-text and setup

`transcript` falls back to speech-to-text through the cloud provider when its key is set, otherwise the local Hugging Face model when its dependencies and `ffmpeg` are installed. A failed cloud request is an error, not a reason to try the local model. `STT_PROVIDER` defaults to `xai` (`XAI_API_KEY`); the others are `elevenlabs` (`ELEVENLABS_API_KEY`), `vercel` (`AI_GATEWAY_API_KEY`), `openai` (`OPENAI_API_KEY`), and `gemini` (`GEMINI_API_KEY`). xAI, ElevenLabs, and the local route return timed lines, the Vercel gateway passes its model's segments through when it has them, and OpenAI and Gemini return one untimed chunk at zero.

The local model is `nvidia/parakeet-tdt-0.6b-v3`, covering 25 European languages only, or any `transformers` speech-recognition model named by `STT_LOCAL_MODEL`, downloaded on first use. An unknown provider or unloadable model falls back to the default with a warning on stderr.

Install from the toolkit's checkout with `uv run python scripts/install.py`, adding `--vision`, `--stt`, or both; `ffmpeg` is installed separately.

`youtube setup` saves embedding and speech-to-text settings for this tool, or one of them with `--embeddings` or `--stt`. Keys go through hidden prompts, so let the user type secrets themselves and never echo them into commands or chat. Enter keeps a saved value, `-` removes it, and cancelling saves nothing; setup makes no API calls. Settings live in a user-only `.env` under `~/Library/Application Support/youtube-toolkit/` on macOS, `${XDG_CONFIG_HOME:-~/.config}/youtube-toolkit/` on Linux, or `%APPDATA%/youtube-toolkit/` on Windows. Environment variables override saved settings, which override defaults.

`youtube doctor` checks readiness without creating files, loading a model, or using the network. `youtube stt-setup` also downloads and runs the local model when it is installed. `--cloud-test` sends a two-second recording the provider bills, so run it only when asked to verify the key.

## Artifacts

Fresh yt-dlp responses and transcripts are saved as JSON named with a UTC datetime and the video ID, query slug, or channel, holding `request`, `fetched_at`, and `payload`; comment archives hold only the sample. Search archives hold listings, not rankings or embeddings. Archives are never reused or expired.

Retrieval is directory-local: there is no database, watchlist, scheduled sync, or persistent embedding index. The calling agent owns synthesis; summarize when the user asks for analysis or a summary.
