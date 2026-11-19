---
name: youtube
description: Use the YouTube toolkit to search videos, list channel uploads, search comment text, read transcripts and metadata, and extract visual context, with request archives kept in the working directory.
---

# YouTube

Run the installed `youtube` CLI from the task's directory; it fetches fresh data and archives each response under `.cache/youtube-toolkit/` there. To test changes to the toolkit itself, run `uv run youtube` in its checkout. Commands print JSON to stdout, except `doctor`, and errors to stderr.

Pass a video as its 11-character ID or any YouTube URL, and always quote URLs: zsh rejects an unquoted `?`, and any shell cuts the command at `&`. An ID starting with `-` reads as an option, so pass it as a URL.

Every command and flag; an option nested inside a flag's brackets needs that flag:

```text
youtube video URL_OR_ID
        [--transcript [--timestamps] [--refine] [--export [--force]] [--stt] [--translate [LANGUAGE]]]
        [--summary [LANGUAGE] [--stt] [--translate [LANGUAGE]]]
        [--vision [--max-frames N] [--rows N] [--columns N] [--resolution 360p|540p|720p] [--min-gap SECONDS] [--min-similarity COSINE] [--local-only] [--no-auto-clean]]
        [--comments [QUERY] [--comment-limit N]]
        [--metadata]
youtube search QUERY ...
        [--semantic chunk|video] [--in-channels CHANNEL ...]
        [--limit N] [--sort relevance|recency|popularity]
        [--days N] [--uploaded-after YYYY-MM-DD]
        [--min-views N] [--min-duration TIME] [--max-duration TIME]
        [--include-channel-names TEXT ...] [--exclude-channel-names TEXT ...]
        [--include-titles TEXT ...] [--exclude-titles TEXT ...]
youtube channel CHANNEL [--last N] [--days N]
youtube setup [--embeddings] [--decisions] [--stt] [--llm]
youtube stt-check [--cloud-test]
youtube doctor
```

## Videos

```bash
youtube video URL_OR_ID --transcript --summary --vision --comments "QUERY" --metadata
youtube video URL_OR_ID
```

| Argument | Required | Default | What it does |
| --- | --- | --- | --- |
| `URL_OR_ID` | Required |  | The video, as its 11-character ID or any YouTube URL. |

`video` prints the parts its flags name, keyed as `transcript`, `summary`, `vision`, `comments`, and `metadata`, or the metadata alone without flags; combine flags to get several parts from one call. Options for a part, such as `--timestamps`, require its flag; `--stt` and `--translate` also work with `--summary`.

### Transcript

```bash
youtube video URL_OR_ID --transcript
youtube video URL_OR_ID --transcript --timestamps
youtube video URL_OR_ID --transcript --export
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--transcript` | Optional | Off | Gives the text as one string, so `jq -r .transcript` prints it. It prefers manual subtitles, then automatic captions, never machine translations, and falls back to speech-to-text, when configured, if there are none. |
| `--timestamps` | Optional | Off: one string | Gives segments with `start_seconds`, `end_seconds`, and `text`. |
| `--stt` | Optional | Off: captions first | Transcribes the audio even when captions exist, for when the captions are poor; it takes longer and fails if speech-to-text is not set up. |
| `--refine` | Optional | Off | Cleans up recognition errors and filler words. |
| `--translate [LANGUAGE]` | Optional | Off; English when LANGUAGE is left out | Returns the transcript in LANGUAGE, written any way the model understands, such as `zh-TW` or `Japanese`. A transcript whose language code matches comes back unchanged unless `--refine` is also given. |
| `--export` | Optional | Off: printed to stdout | Writes a text file to the working directory, and `transcript` holds its path. |
| `--force` | Optional | Off: an existing file is an error | Replaces an export file that already exists. |

`--refine` and `--translate` work line by line on the captions' own timestamps with `--timestamps`, otherwise as paragraphs of complete sentences under the video's chapter titles. Use them when the user wants the full text to read or in another language, since reading a transcript to answer a question needs neither.

Stdout stops at 400,000 characters, so use `--export` for long videos.

### Summary

```bash
youtube video URL_OR_ID --summary
youtube video URL_OR_ID --summary en
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--summary [LANGUAGE]` | Optional | Off; the `--translate` language, else the transcript's own, when LANGUAGE is left out | Gives a Markdown summary of the whole transcript, written by the LLM set up with `youtube setup --llm`, in LANGUAGE written any way the model understands. |
| `--stt` | Optional | Off: captions first | Summarizes the speech-to-text transcript instead of the captions. |
| `--translate [LANGUAGE]` | Optional | Off; English when LANGUAGE is left out | Sets the summary's language when `--summary` names none. |

It fetches the transcript without printing it unless `--transcript` is also given. Use it when the user wants a video summarized, especially a long one; a summary leaves details out, so answer specific questions from the transcript. It only condenses one video, so analysis and comparisons across videos are yours.

### Vision

```bash
youtube video URL_OR_ID --vision --max-frames 24
youtube video URL_OR_ID --transcript --vision --export --timestamps
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--vision` | Optional | Off | Downloads the video, up to two hours, picks screenshots, and gives their timestamps and paths to collages of them. |
| `--max-frames N` | Optional | 96 | Returns at most N screenshots, from 1 to 96; the result can be shorter. |
| `--rows N` | Optional | 3 | Sets the rows of screenshots on each collage page, from 1 to 3. |
| `--columns N` | Optional | 2 | Sets the screenshots in each row, from 1 to 3. |
| `--resolution 360p\|540p\|720p` | Optional | 540p | Sets each screenshot's resolution; 360p saves image tokens but loses small text and digits, and 720p shows more detail for small text. |
| `--min-gap SECONDS` | Optional | 10 | Keeps screenshots at least that far apart. |
| `--min-similarity COSINE` | Optional | Tuned for the ranking model | Sets how closely a frame must match the speech when embeddings rank, from -1 to 1. |
| `--local-only` | Optional | Off | Skips the decision model, the embedding API, and speech-to-text, ranking against captions with the local embedding model or else taking evenly spaced frames. |
| `--no-auto-clean` | Optional | Off: files are deleted | Keeps the downloaded video and candidate frames. |

Use `--vision` when pictures add context, and read the collages alongside the transcript. A run usually takes a few minutes, mostly the download, and the defaults suit most videos. To change only the layout, rerun with new `--rows`, `--columns`, or a resolution no higher than before, and the same other options: the earlier picks are laid out again, which takes seconds and costs nothing. With `--transcript`, vision ranks against that same transcript, and `--export` writes it as Markdown beside the collages.

### Comments

```bash
youtube video URL_OR_ID --comments "QUERY"
youtube video URL_OR_ID --comments "QUERY" --comment-limit 250
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--comments [QUERY]` | Optional | Off; the whole sample when QUERY is left out | Samples top-level comments in YouTube's Top order and keeps those containing QUERY as a case-insensitive literal substring. |
| `--comment-limit N` | Optional | 100 | Samples N comments, up to 1000. |

An empty list means no match in the sample, not on the whole video, so raise `--comment-limit` for coverage.

### Metadata

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--metadata` | Optional | Off, but on when no other part is asked for | Adds the video's facts, including the description and `chapters`. |

## Search

```bash
youtube search "langgraph tutorial" --limit 5 --max-duration 20:00
youtube search "reasoning models" --in-channels @OpenAI @Anthropic --days 30 --sort recency
youtube search "agent memory" --semantic chunk --limit 5
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `QUERY ...` | Required, unless `--in-channels` is given without `--semantic` |  | Searches for each query and interleaves their results. |
| `--semantic RESULT` | Optional | Off: YouTube's own search | Ranks caption passages by meaning with embeddings and returns the passages (`chunk`) or their videos (`video`). |
| `--in-channels CHANNEL ...` | Optional | All of YouTube | Searches only these channels' uploads, each a URL, handle, or ID. |
| `--limit N` | Optional | 10, or 30 inside channels | Returns at most N results. |
| `--sort ORDER` | Optional | `relevance`, or `recency` inside channels without `--semantic` | Orders by `relevance`, `recency`, or `popularity`; a plain search across YouTube cannot sort by recency. |
| `--days N` | Optional | Any age | Keeps videos uploaded within the last N days. |
| `--uploaded-after YYYY-MM-DD` | Optional | Any date | Keeps videos uploaded on or after that date. |
| `--min-views N` | Optional | 10,000 | Keeps videos with at least N views. |
| `--min-duration TIME` | Optional | 3:00 | Keeps videos at least that long, as seconds, `MM:SS`, or `HH:MM:SS`. |
| `--max-duration TIME` | Optional | 8:00:00 | Keeps videos at most that long. |
| `--include-channel-names TEXT ...` | Optional | No filter | Keeps videos whose channel name or handle contains every TEXT. |
| `--exclude-channel-names TEXT ...` | Optional | No filter | Drops videos whose channel name or handle contains any TEXT. |
| `--include-titles TEXT ...` | Optional | No filter | Keeps videos whose title contains every TEXT. |
| `--exclude-titles TEXT ...` | Optional | No filter | Drops videos whose title contains any TEXT. |

Override the view and length defaults when the task needs small or short videos. Date filters and recency sorting are slower because they fetch each video's metadata. Put the query before any option that takes a list, or it is read as one more item of the list.

## Channels

```bash
youtube channel @OpenAI --last 5
youtube channel @OpenAI --days 7
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `CHANNEL` | Required |  | Lists this channel's recent uploads; a URL, handle, or channel ID. |
| `--last N` | Optional | 10 | Returns the latest N uploads. |
| `--days N` | Optional | Off | Returns every upload from the last N days instead, and `--last` no longer caps the result. |

## Setup

Install from the toolkit's checkout with `uv run python scripts/install.py`, adding `--vision`, `--embeddings`, or `--stt` as needed; vision and speech-to-text also need `ffmpeg`.

| Command | What it does |
| --- | --- |
| `youtube setup` | Saves keys and model settings, or one group with `--embeddings`, `--decisions`, `--stt`, or `--llm`. Keys go through hidden prompts, so let the user type secrets themselves and never echo them into commands or chat. |
| `youtube doctor` | Checks readiness without the network. |
| `youtube stt-check` | Reports the speech-to-text setup and, for a local model, downloads it and checks that it runs. |
| `youtube stt-check --cloud-test` | Also sends a recording the provider bills, so run it only when asked to verify the key. |
