"""Transcript-first YouTube research toolkit behind the `youtube` CLI.

Metadata, comments, searches, and channel listings are fetched fresh on every call. Everything the toolkit keeps lives under the working directory's `.youtube-toolkit/`, where a video's transcripts, rewrites, summaries, and screenshot picks are saved and reused by later calls with the same options.
"""
