# YouTube Toolkit

A YouTube CLI for agents with directory-local request archives: search, read transcripts, inspect metadata, and extract visual context through yt-dlp.

![Route map of the toolkit: each command is a line through the parts it uses. Every command except doctor reaches YouTube through yt-dlp; semantic search can also call an embedding API, and missing captions fall back to a speech-to-text API.](docs/architecture.svg)

## Install

Requires Python 3.13+ and uv.

```bash
uv run python scripts/install.py
```

For visual extraction, install its optional dependencies and install `ffmpeg` separately:

```bash
uv run python scripts/install.py --vision
```

For local speech-to-text on videos without captions, install its optional dependencies (Hugging Face `transformers` and PyTorch) and install `ffmpeg` separately:

```bash
uv run python scripts/install.py --stt
```

The installer replaces the installed `youtube` tool. To run the checkout directly, use `uv run youtube`.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
youtube metadata <url-or-id>
youtube transcript <url-or-id>
youtube transcript <url-or-id> --timestamps
youtube transcript <url-or-id> --export
youtube vision <url-or-id>
youtube stt-setup
youtube doctor
```

Commands return JSON except `doctor`. Metadata includes descriptions, thumbnails, and chapters. Transcripts return text by default or timed chunks with `--timestamps`; `--export` writes a text file in the working directory, and `--force` allows overwriting it. Transcripts are in English or Chinese when the video has those subtitles, otherwise in the video's own language. Videos without captions fall back to [speech-to-text](#speech-to-text) when it is set up, and otherwise fail with a no-captions error.

Search defaults: at least 10,000 views, duration 3 minutes to 8 hours, and 10 results (30 for channel searches). Use `youtube search --help` for filters and overrides.

Vision returns up to six screenshots in timestamped 2-column × 3-row collages, plus per-frame measurements in JSON. Set `--max-frames 12` to change the limit (1-96). Timed captions choose where to look and local image analysis picks the frames; without timed captions it samples evenly. No model API or speech-to-text is used.

![How vision picks frames: caption passages choose where to look, only those windows are decoded at one frame per second, local image analysis picks one sharp, steady frame per window, and the result is timestamped collages with JSON measurements.](docs/vision-pipeline.svg)

## Semantic search

Semantic search uses an embedding API to rank transcript passages:

```bash
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --channels @OpenAI --mode semantic --return video
```

Set `EMBEDDING_API_KEY` in the environment. Optional settings are `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSIONS`. Semantic results are chunks by default; `--return video` returns ranked videos.

## Speech-to-text

Videos without captions are transcribed by the first route that is set up: a cloud provider when its key is set, otherwise the local model when it is installed with `--stt`.

`STT_PROVIDER` chooses the cloud provider, and each one needs its own key:

| `STT_PROVIDER` | Key | Model |
| --- | --- | --- |
| `elevenlabs` | `ELEVENLABS_API_KEY` | `scribe_v2` |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.5-transcribe` |
| `openai` | `OPENAI_API_KEY` | `gpt-transcribe` |
| `vercel` | `AI_GATEWAY_API_KEY` | `spacexai/grok-stt` through Vercel AI Gateway |
| `xai` (default) | `XAI_API_KEY` | `grok-voice-transcribe-2.0` |

Gemini and the Vercel gateway each transcribed an hour of audio in under a minute. Gemini accepts recordings up to one hour, and OpenAI accepts files up to 25 MB, which is about 25 minutes of YouTube audio.

The local model is `nvidia/parakeet-tdt-0.6b-v3`, or the Hugging Face ASR model ID in `STT_LOCAL_MODEL`. The local route runs any model that works in the `transformers` `automatic-speech-recognition` pipeline, on an Nvidia GPU, an Apple GPU, or the CPU. The model downloads the first time it is needed. An unknown `STT_PROVIDER`, or a local model that cannot be loaded, falls back to the default with a warning on stderr.

Parakeet covers 25 European languages only and returns noise for others such as Chinese. For those, use the cloud route or set `STT_LOCAL_MODEL=openai/whisper-large-v3-turbo`. On an Apple M5, Parakeet transcribed an hour of audio in 45 seconds and Whisper large-v3-turbo took about 4 minutes.

```bash
youtube stt-setup
youtube stt-setup --cloud-test
```

`stt-setup` reports both routes, downloads the local model if needed, and runs it once. `--cloud-test` also sends a two-second recording to the cloud provider, which is a real request the provider bills.

## Agent skill

The repository skill is [skills/youtube/SKILL.md](skills/youtube/SKILL.md).
