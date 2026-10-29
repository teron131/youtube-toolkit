# YouTube Toolkit

A YouTube CLI for agents: simple search and fetching of videos, metadata, transcripts, and comments, plus curated screenshots for visual context.

![How the commands fit together: each command is a line through the shared parts it uses.](docs/architecture.svg)

## Install

Requires Python 3.13+ and uv. The installer builds the checkout and installs `youtube` as a uv tool: an isolated environment with the command in uv's tool bin directory, usually `~/.local/bin`, so it runs from any directory without touching the system Python (`uv tool update-shell` adds that directory to PATH if needed). Rerun it after pulling changes; it replaces the previous install and finishes with `youtube doctor`. `--vision` and `--stt` add optional dependencies and need `ffmpeg` installed separately.

```bash
uv run python scripts/install.py
uv run python scripts/install.py --vision
uv run python scripts/install.py --stt
```

`--stt` installs Hugging Face `transformers` and PyTorch for local speech-to-text. Run the checkout directly with `uv run youtube`.

## Configuration

```bash
youtube setup
youtube setup --embeddings
youtube setup --stt
```

Setup saves embedding and speech-to-text settings, or one of them with a flag. Keys are entered at hidden prompts, never as flags; Enter keeps a saved value and `-` removes it. Setup makes no API calls.

| Platform | Settings file |
| --- | --- |
| macOS | `~/Library/Application Support/youtube-toolkit/.env` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/youtube-toolkit/.env` |
| Windows | `%APPDATA%/youtube-toolkit/.env` |

The file is readable only by the current user on macOS and Linux, and `XDG_CONFIG_HOME` moves it on macOS too. Nonblank environment variables override saved settings, which override the defaults below. Values are read literally, without `${VARIABLE}` expansion.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search --channels @OpenAI --limit 5
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
youtube metadata <url-or-id>
youtube comments <url-or-id> "pricing"
youtube transcript <url-or-id>
youtube transcript <url-or-id> --timestamps
youtube transcript <url-or-id> --export
youtube vision <url-or-id>
youtube stt-setup
youtube doctor
```

Commands print JSON, except `doctor`, which checks local readiness without network calls.

## Search and channel listings

Search covers YouTube, or the uploads of channels named with `--channels`, which repeats.

| Mode | Query | Default order |
| --- | --- | --- |
| YouTube search | Required; several queries interleave without duplicates | Relevance |
| Channel search (`--channels`) | Optional; titles must contain one of the queries | Recency |
| `youtube channel` | None; a plain upload listing | Latest first |

`--sort popularity` orders by views; YouTube-wide search rejects `--sort recency`. Defaults are 10 results (30 for channel searches), at least 10,000 views, and a length from 3:00 to 8:00:00; `youtube search --help` lists every filter. Channel listings carry no upload dates, so date filters and recency sorting fetch each video's metadata.

`youtube channel --last N` lists the latest uploads. `--days N` keeps uploads inside a UTC date window and drops videos it cannot date; `--last` then sizes the listing it checks.

### Semantic search

`--mode semantic` shortlists videos by title, then ranks their caption passages with an embedding API. It returns passages, or videos ranked by their best passage with `--return video`; each passage scores against its closest query.

```bash
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --channels @OpenAI --mode semantic --return video
```

| Variable | Default |
| --- | --- |
| `EMBEDDING_API_KEY` | Required |
| `EMBEDDING_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/`, or any OpenAI-compatible endpoint |
| `EMBEDDING_MODEL` | `gemini-embedding-2` |
| `EMBEDDING_DIMENSIONS` | 768 |
| `EMBEDDING_CONCURRENCY` | 16, clamped to 1–50 |

It reads captions only, never speech-to-text. Partial failures appear in `warnings`, and the command fails only when no passage can be embedded. `chunk_index` is not a timestamp; use `transcript --timestamps` for timing.

## Comment search

```bash
youtube comments <url-or-id> "pricing"
youtube comments <url-or-id> "training data" --limit 250
youtube comments <url-or-id> "pricing" | jq '.comments[] | select((.like_count // 0) >= 10)'
```

It fetches up to 100 top-level comments in YouTube's Top order (`--limit` takes 1–1000) and keeps those containing the query as a case-insensitive literal substring, like `rg -i -F`. `--limit` sizes the sample, not the matches; omit the query for the whole sample. Matches alternate between Top order and most likes, without repeats. No API key is needed.

Output holds only `title`, `url`, and `comments`, each with `text` and `like_count`, which is `null` when unavailable:

```json
{
  "title": "Example video",
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "comments": [
    {"text": "The pricing comparison was helpful.", "like_count": 12}
  ]
}
```

The unfiltered sample is archived in the same shape under `.cache/youtube-toolkit/comments/`.

## Metadata and transcripts

Metadata includes the description, thumbnails, and chapters. Transcripts print as one JSON string, or as timed chunks with `--timestamps`. Manual English or Chinese subtitles come first, starting with the video's language family, then original-language automatic captions. Machine translations are never used, Chinese becomes Taiwan Traditional, and SponsorBlock segments are removed.

Without usable captions, `transcript` falls back to [speech-to-text](#speech-to-text) when it is configured; caption download failures stay errors. `--export` writes metadata and text to `<normalized-title>_<normalized-channel>.txt` in the working directory, with `--timestamps` for a timed export and `--force` to overwrite.

## Visual context

`youtube vision` returns six-frame collages and per-frame JSON. Its candidates are the video's keyframes, which YouTube places every three to seven seconds, minus sponsor segments, dark frames, and near-copies.

![How vision picks frames: caption or speech-to-text text and keyframes are embedded, matched, ranked per scope, fused, and deduplicated into collages, with a dashed time-slot lane when there is no image embedding model or timed transcript.](docs/vision-pipeline.svg)

With an image embedding key and a timed transcript, Gemini Embedding 2 ranks the candidates against the transcript's keywords, sentences, and paragraphs. A frame matches only text spoken from five seconds before to two seconds after it, scored relative to its similarity to the whole transcript, so silent stretches yield no ranked frames. Reciprocal rank fusion merges the three scopes, then frames within the minimum gap or with image-embedding cosine 0.96 or more to a pick are skipped. The cutoff is never lowered to fill the requested count.

The key is `GEMINI_API_KEY`, or `EMBEDDING_API_KEY` when the embedding settings point at Gemini Embedding 2 on Google. The transcript is the one `transcript` returns, so a video without captions runs speech-to-text. Without a key or a timed transcript, or with `--local-only`, vision takes the candidate nearest the middle of each time slot of at least 12 seconds, skipping picks within the minimum gap of the previous one or repeating any earlier one. `selection_mode` and the warnings say which ran and why. A rejected key fails within seconds, before the video downloads, and names Google's reason code.

| Option or limit | Value |
| --- | --- |
| `--max-frames` | 1–96, default 96; the result can be shorter |
| `--min-similarity` | Raw cosine floor, default 0.30 |
| `--min-gap` | Seconds between frames, default 10 |
| Collage page | Up to six 960×540 frames, 2 columns × 3 rows |
| Video limits | 2 hours, 500 MiB download |

Collages go under `.cache/youtube-toolkit/vision/`; the video and candidate images are deleted afterwards unless `--no-auto-clean` keeps them, reported as `media_dir`. JSON includes `candidate_count`, `embedding_usage`, and per frame raw `visual_scores` and, for ranked picks, `relevance`: rank, RRF score, and matched texts with raw and relative cosine. Scores are measurements, not probabilities.

## Speech-to-text

`transcript` uses a cloud provider when its key is set, otherwise a local model when its dependencies and `ffmpeg` are installed; provider errors propagate. Save the provider, key, and local model with `youtube setup --stt`, or set them in the environment.

| `STT_PROVIDER` | Key | Model | Timing |
| --- | --- | --- | --- |
| `xai` (default) | `XAI_API_KEY` | `grok-voice-transcribe-2.0` | Per word |
| `elevenlabs` | `ELEVENLABS_API_KEY` | `scribe_v2` | Per word |
| `vercel` | `AI_GATEWAY_API_KEY` | `spacexai/grok-stt` through Vercel AI Gateway | Model segments, when provided |
| `openai` | `OPENAI_API_KEY` | `gpt-transcribe` | None |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.5-transcribe` | None; word timing would cap audio at 30 minutes |

Word timings become caption-like lines; an untimed transcript is one chunk at zero, which vision cannot rank against. Gemini accepts up to an hour of audio, and OpenAI files up to 25 MB, about 25 minutes.

The local model is `nvidia/parakeet-tdt-0.6b-v3`, or any `transformers` speech-recognition model named in `STT_LOCAL_MODEL`, downloaded on first use and run on an Nvidia GPU, an Apple GPU, or the CPU. It is timed per piece of up to 28 seconds. Parakeet covers 25 European languages only; for others, use a cloud provider or `STT_LOCAL_MODEL=openai/whisper-large-v3-turbo`, which took about 4 minutes per hour of audio on an Apple M5 against Parakeet's 45 seconds. An unknown provider or unloadable model falls back to the default with a warning.

| Command | Checks | Network |
| --- | --- | --- |
| `youtube doctor` | Local readiness and configuration | None |
| `youtube stt-setup` | Both routes, running the local model on a test recording | Downloads the local model if needed |
| `youtube stt-setup --cloud-test` | Also sends a two-second recording to the cloud provider | One billed request |

## Local artifacts

Run commands where their artifacts belong. Fresh YouTube responses and transcripts are archived under `.cache/youtube-toolkit/` with UTC-timestamped names, as `request`, `fetched_at`, and `payload`; comment archives hold just the sample. Archives are never reused or expired. Text fields in stdout stop at 400,000 characters, while exports and archives keep everything.

## Agent skill

The repository skill is [skills/youtube/SKILL.md](skills/youtube/SKILL.md).
