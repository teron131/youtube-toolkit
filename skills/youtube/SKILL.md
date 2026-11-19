---
name: youtube
description: Use the YouTube toolkit to search videos, list channel uploads, search comment text, read transcripts and metadata, extract visual context, and download videos, their sound, or their frames, with each video's results saved as files in the working directory.
---

# YouTube

Run the installed `youtube` CLI from the task's directory; it saves what it fetches under `.youtube-toolkit/` there, which Git ignores. To test changes to the toolkit itself, run `uv run youtube` in its checkout. Commands print JSON to stdout, except `doctor`, and errors to stderr.

Pass a video as its 11-character ID or any YouTube URL, and always quote URLs: zsh rejects an unquoted `?`, and any shell cuts the command at `&`. An ID starting with `-` reads as an option, so pass it as a URL.

Every command and flag; an option nested inside a flag's brackets needs that flag:

```text
youtube video URL_OR_ID
        [--transcript [--timestamps] [--refine] [--stt] [--translate [LANGUAGE]]]
        [--summary [LANGUAGE] [--stt] [--translate [LANGUAGE]]]
        [--vision [--max-frames N] [--rows N] [--columns N] [--resolution 360p|540p|720p] [--no-timestamp-annotation] [--min-gap SECONDS] [--min-similarity COSINE] [--local-only] [--no-auto-clean] [--start TIME] [--end TIME]]
        [--download [video|audio|frames|keyframes] [--quality 360p|480p|720p|1080p] [--at TIME ...] [--start TIME] [--end TIME]]
        [--comments [QUERY] [--comment-limit N]]
        [--metadata]
        [--fresh]
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

`video` saves the parts its flags name as files in the video's folder, `.youtube-toolkit/videos/TITLE_ID/`, and prints each file's `path` with a few facts, keyed as `transcript`, `summary`, `vision`, `download`, `comments`, and `metadata`, or the metadata alone without flags; combine flags to get several parts from one call. The output stays short however long the content, so check `characters` and read a long file in pieces with `sed`, `grep`, or `jq` rather than all at once. Transcripts, rewrites, summaries, and screenshot picks made earlier with the same options come back at once and free, with `reused: true`; `--fresh` makes every part of the call afresh. Options for a part, such as `--timestamps`, require its flag; `--stt` and `--translate` also work with `--summary`, and `--start` and `--end` with `--vision`.

### Transcript

```bash
youtube video URL_OR_ID --transcript
youtube video URL_OR_ID --transcript --timestamps
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--transcript` | Optional | Off | Saves the transcript as Markdown, the video's facts over one sentence per line, and gives its `path`, `characters`, `language`, and `source`. It prefers manual subtitles, then automatic captions, never machine translations, and falls back to speech-to-text, when configured, if there are none. |
| `--timestamps` | Optional | Off: one sentence per line | Starts each caption line with its time, such as `[00:01:15]`. |
| `--stt` | Optional | Off: captions first | Transcribes the audio even when captions exist, for when the captions are poor; it takes longer and fails if speech-to-text is not set up. |
| `--refine` | Optional | Off | Cleans up recognition errors and filler words. |
| `--translate [LANGUAGE]` | Optional | Off; English when LANGUAGE is left out | Returns the transcript in LANGUAGE, written any way the model understands, such as `zh-TW` or `Japanese`. A transcript whose language code matches comes back unchanged unless `--refine` is also given. |

`--refine` and `--translate` work line by line on the captions' own timestamps with `--timestamps`, otherwise as paragraphs of complete sentences under the video's chapter titles. Use them when the user wants the full text to read or in another language, since reading a transcript to answer a question needs neither.

### Summary

```bash
youtube video URL_OR_ID --summary
youtube video URL_OR_ID --summary en
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--summary [LANGUAGE]` | Optional | Off; the `--translate` language, else the transcript's own, when LANGUAGE is left out | Saves a Markdown summary of the whole transcript, written by the LLM set up with `youtube setup --llm`, in LANGUAGE written any way the model understands, and gives its `path` and `characters`. |
| `--stt` | Optional | Off: captions first | Summarizes the speech-to-text transcript instead of the captions. |
| `--translate [LANGUAGE]` | Optional | Off; English when LANGUAGE is left out | Sets the summary's language when `--summary` names none. |

It fetches the transcript without giving it unless `--transcript` is also given. Use it when the user wants a video summarized, especially a long one; a summary leaves details out, so answer specific questions from the transcript. It only condenses one video, so analysis and comparisons across videos are yours.

### Vision

```bash
youtube video URL_OR_ID --vision --max-frames 24
youtube video URL_OR_ID --transcript --vision --timestamps
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--vision` | Optional | Off | Downloads up to two hours of the video, picks screenshots, and gives the paths of the `collages` pages, the number of `frames`, and the `path` of `picks.json`, which lists each screenshot's time, its own image as `image_path`, and why it was picked. |
| `--max-frames N` | Optional | 96 | Returns at most N screenshots, from 1 to 96; the result can be shorter. |
| `--rows N` | Optional | 3 | Sets the rows of screenshots on each collage page, from 1 to 3. |
| `--columns N` | Optional | 2 | Sets the screenshots in each row, from 1 to 3. |
| `--resolution 360p\|540p\|720p` | Optional | 540p | Sets each screenshot's resolution; 360p saves image tokens but loses small text and digits, and 720p shows more detail for small text. |
| `--no-timestamp-annotation` | Optional | Off: each screenshot shows its time beneath | Leaves the times out of the collages. |
| `--min-gap SECONDS` | Optional | 10 | Keeps screenshots at least that far apart. |
| `--min-similarity COSINE` | Optional | Tuned for the ranking model | Sets how closely a frame must match the speech when embeddings rank, from -1 to 1. |
| `--local-only` | Optional | Off | Skips the decision model, the embedding API, and speech-to-text, ranking against captions with the local embedding model or else taking evenly spaced frames. |
| `--no-auto-clean` | Optional | Off: files are deleted | Keeps the downloaded video and candidate frames in a `work` folder beside `picks.json`. |
| `--start TIME`, `--end TIME` | Optional | The whole video | Picks screenshots only from that part, given as seconds, `MM:SS`, or `HH:MM:SS`, with a fraction if needed such as `1:15.25`, and downloads only that part. |

Use `--vision` when pictures add context, and read the collages alongside the transcript, or open the screenshots one by one from `picks.json` when a collage is too small to read. A run usually takes a few minutes, mostly the download, and the defaults suit most videos; for one part of a long video, `--start` and `--end` make it faster and cheaper. To change only the layout, rerun with new `--rows`, `--columns`, or a resolution no higher than before, and the same other options: the earlier picks are laid out again, which takes seconds and costs nothing, and `--fresh` picks again instead. With `--transcript`, vision ranks against that same transcript.

### Download

```bash
youtube video URL_OR_ID --download
youtube video URL_OR_ID --download audio --start 1:15 --end 2:30
youtube video URL_OR_ID --download frames --start 10:00 --end 15:00
youtube video URL_OR_ID --download frames --at 1:15 4:02
```

| Option | Required | Default | What it does |
| --- | --- | --- | --- |
| `--download [video\|audio\|frames\|keyframes]` | Optional | Off; `video` when left out | Saves the video with its sound as an mp4, the sound alone as an m4a, a folder of JPEGs with one frame per visible change, or a folder of every keyframe, named for the title, channel, video quality, kind, and range in the working directory; `download` holds the file or folder `path`, its `bytes` or number of `frames`, and `reused` when an earlier download of the same request came back instead. Frames are named by their time in the video, such as `00h01m02.812s.jpg`, so they list in time order. |
| `--quality 360p\|480p\|720p\|1080p` | Optional | 1080p | Caps the video's resolution and sets the frames' size; the sound is the same at every level. |
| `--at TIME ...` | Optional | Off | With `--download frames`, saves the first frame at or after each time, keyframe or not, as its own JPEG named for the time instead of a folder; `download` is then a list with a `path` and `reused` per time. |
| `--start TIME` | Optional | The beginning | Keeps the video from this time, as seconds, `MM:SS`, or `HH:MM:SS`, with a fraction if needed such as `1:15.25`; a video opens on the first frame at or after it. |
| `--end TIME` | Optional | The end | Keeps the video up to this time. |

Use it when the user wants the file itself, or to hand a video or its sound to another tool, such as a model that watches video: 360p or 480p uploads much faster and loses little for a model. With `--start` and `--end`, only that range is fetched, so take the times from the transcript to clip one moment. With `--vision`, a video downloaded at 720p or more over the same range is where the screenshots are picked from, so the video downloads once.

To look through a video yourself, download its frames rather than `--vision`'s curated screenshots: `frames` keeps every visible change at full quality with no model involved, `keyframes` keeps every keyframe, blank and repeated ones included, and `--at` fetches the frames at times the transcript points to in seconds, without downloading the rest. Match frame names to transcript timestamps to see what was on screen as something was said.

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

Install from the toolkit's checkout with `uv run python scripts/install.py`, adding `--vision` (screenshots and frame downloads), `--embeddings`, or `--stt` as needed; vision, speech-to-text, and downloads also need `ffmpeg`.

| Command | What it does |
| --- | --- |
| `youtube setup` | Saves keys and model settings, or one group with `--embeddings`, `--decisions`, `--stt`, or `--llm`. Keys go through hidden prompts, so let the user type secrets themselves and never echo them into commands or chat. |
| `youtube doctor` | Checks readiness without the network. |
| `youtube stt-check` | Reports the speech-to-text setup and, for a local model, downloads it and checks that it runs. |
| `youtube stt-check --cloud-test` | Also sends a recording the provider bills, so run it only when asked to verify the key. |
