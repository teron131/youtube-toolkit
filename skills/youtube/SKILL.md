---
name: youtube
description: Use the YouTube toolkit to search videos, list channel uploads, search comment text, read transcripts and metadata, and extract visual context, with request archives kept in the working directory.
---

# YouTube

Run the installed `youtube` CLI from the task's directory; it fetches fresh data and archives each response under `.cache/youtube-toolkit/` there. To test changes to the toolkit itself, run `uv run youtube` in its checkout. Commands print JSON to stdout, except `doctor`, and errors to stderr.

Pass a video as its 11-character ID or any YouTube URL, and always quote URLs: zsh rejects an unquoted `?`, and any shell cuts the command at `&`. An ID starting with `-` reads as an option, so pass it as a URL.

## Videos

```bash
youtube video URL_OR_ID --transcript --vision --comments "QUERY" --metadata
youtube video URL_OR_ID
```

`video` prints the parts its flags name, keyed as `transcript`, `vision`, `comments`, and `metadata`, or the metadata alone without flags; combine flags to get several parts from one call. Options for a part, such as `--timestamps`, require its flag.

### Transcript

```bash
youtube video URL_OR_ID --transcript
youtube video URL_OR_ID --transcript --timestamps
youtube video URL_OR_ID --transcript --export
```

`--transcript` gives the text as one string, so `jq -r .transcript` prints it; `--timestamps` gives chunks with `start_seconds`, `end_seconds`, and `text`. It prefers manual subtitles, then automatic captions, never machine translations, and falls back to speech-to-text, when configured, if there are none. Stdout stops at 400,000 characters, so for long videos use `--export`, which writes a text file to the working directory, and `transcript` holds its path.

### Vision

```bash
youtube video URL_OR_ID --vision --max-frames 24
youtube video URL_OR_ID --transcript --vision --export --timestamps
```

Use `--vision` when pictures add context. It downloads the video, up to two hours, picks screenshots, and gives their timestamps and paths to collages of six frames; read the collages alongside the transcript. `--max-frames` takes 1–96, default 96, and the result can be shorter. A run usually takes a few minutes, mostly the download. With `--transcript`, vision ranks against that same transcript, and `--export` writes it as Markdown beside the collages.

### Comments

```bash
youtube video URL_OR_ID --comments "QUERY"
youtube video URL_OR_ID --comments "QUERY" --comment-limit 250
```

`--comments` samples top-level comments in YouTube's Top order, 100 by default and up to 1000 with `--comment-limit`, and keeps those containing the query as a case-insensitive literal substring, or all of them without a query. An empty list means no match in the sample, not on the whole video, so raise `--comment-limit` for coverage.

### Metadata

Metadata includes the description and `chapters`.

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --channels @OpenAI --channels @Anthropic --uploaded-within-days 30 --sort recency
youtube search "agent memory" --mode semantic --limit 5
```

`search` covers all of YouTube, or the channels named with `--channels`. By default it keeps videos with at least 10,000 views and 3 minutes to 8 hours long, so override those when the task needs other videos; `youtube search --help` lists every filter. Date filters and recency sorting are slower because they fetch each video's metadata. `--mode semantic` ranks caption passages by meaning, and `--return video` returns videos instead of passages.

## Channels

```bash
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
```

`channel` lists recent uploads: the last 10, `--last N`, or `--days N`.

## Setup

Install from the toolkit's checkout with `uv run python scripts/install.py`, adding `--vision`, `--embeddings`, or `--stt` as needed; vision and speech-to-text also need `ffmpeg`.

`youtube setup` saves keys and model settings, or one group with `--embeddings`, `--decisions`, or `--stt`. Keys go through hidden prompts, so let the user type secrets themselves and never echo them into commands or chat. `youtube doctor` checks readiness without the network. `youtube stt-setup --cloud-test` sends a recording the provider bills, so run it only when asked to verify the key.

The calling agent owns synthesis; summarize when the user asks for analysis or a summary.
