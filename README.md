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

The installer replaces the installed `youtube` tool. To run the checkout directly, use `uv run youtube`.

## Examples

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --source channel --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube channel @OpenAI --last 5
youtube metadata <url-or-id>
youtube transcript <url-or-id>
youtube transcript <url-or-id> --timestamps
youtube transcript <url-or-id> --export
youtube vision <url-or-id>
youtube doctor
```

Commands return JSON except `doctor`. Metadata includes descriptions, thumbnails, and chapters. Transcripts return text by default or timed chunks with `--timestamps`; `--export` writes a text file in the working directory, and `--force` allows overwriting it. Videos without captions fall back to xAI speech-to-text, which needs `XAI_API_KEY`.

Search defaults: at least 10,000 views, duration 3 minutes to 8 hours, and 10 results (30 for channel-only searches). Use `youtube search --help` for filters and overrides.

Vision returns up to six screenshots in timestamped 2-column × 3-row collages, plus per-frame measurements in JSON. Set `--max-frames 12` to change the limit (1-96). Timed captions choose where to look and local image analysis picks the frames; without timed captions it samples evenly. No model API or speech-to-text is used.

![How vision picks frames: caption passages choose where to look, only those windows are decoded at one frame per second, local image analysis picks one sharp, steady frame per window, and the result is timestamped collages with JSON measurements.](docs/vision-pipeline.svg)

## Semantic search

Semantic search uses an embedding API to rank transcript passages:

```bash
youtube search "agent memory" --mode semantic --limit 5
youtube search "model evals" --source channel --channels @OpenAI --mode semantic --return video
```

Set `EMBEDDING_API_KEY` in the environment. Optional settings are `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSIONS`. Semantic results are chunks by default; `--return video` returns ranked videos.

## Agent skill

The repository skill is [skills/youtube/SKILL.md](skills/youtube/SKILL.md).
