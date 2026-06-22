"""Embedding-backed semantic indexing and ranking for stored videos."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .connection import find_repo_root

EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
EMBEDDING_PROVIDER = "gemini"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONALITY = 768
EmbedText = Callable[[str], list[float]]


class EmbeddingSearchError(RuntimeError):
    """Raised when semantic search cannot call or parse the embedding provider."""


def index_video_embedding(
    connection: sqlite3.Connection,
    youtube_video_id: str,
    *,
    embed_text: EmbedText | None = None,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT id, youtube_video_id, title, upload_date, transcript
        FROM videos
        WHERE youtube_video_id = ?
        LIMIT 1
        """,
        (youtube_video_id,),
    ).fetchone()
    if row is None:
        return None

    source_text = "\n".join(
        str(part)
        for part in (
            row["youtube_video_id"],
            row["title"],
            row["upload_date"],
            row["transcript"],
        )
        if part not in (None, "")
    )
    content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    existing_embedding_row = connection.execute(
        """
        SELECT content_hash, embedding_json
        FROM video_embeddings
        WHERE video_id = ? AND provider = ? AND model = ?
        """,
        (
            row["id"],
            EMBEDDING_PROVIDER,
            DEFAULT_EMBEDDING_MODEL,
        ),
    ).fetchone()
    if existing_embedding_row is not None and existing_embedding_row["content_hash"] == content_hash:
        return embedding_index_payload(
            row,
            content_hash,
            embedding_from_json(existing_embedding_row["embedding_json"]),
        )

    embedding = (embed_text or embed_text_with_gemini)(source_text)
    if not embedding:
        raise EmbeddingSearchError("Embedding provider returned an empty vector")
    connection.execute(
        """
        INSERT INTO video_embeddings (
            video_id,
            provider,
            model,
            content_hash,
            embedding_json
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(video_id, provider, model)
        DO UPDATE SET
            content_hash = excluded.content_hash,
            embedding_json = excluded.embedding_json,
            embedded_at = CURRENT_TIMESTAMP
        """,
        (
            row["id"],
            EMBEDDING_PROVIDER,
            DEFAULT_EMBEDDING_MODEL,
            content_hash,
            json.dumps(embedding),
        ),
    )
    return embedding_index_payload(row, content_hash, embedding)


def embedding_index_payload(
    row: sqlite3.Row,
    content_hash: str,
    embedding: list[float],
) -> dict[str, Any]:
    return {
        "video_id": row["youtube_video_id"],
        "database_row_id": int(row["id"]),
        "provider": EMBEDDING_PROVIDER,
        "model": DEFAULT_EMBEDDING_MODEL,
        "content_hash": content_hash,
        "dimensions": len(embedding),
    }


def embed_text_with_gemini(text: str) -> list[float]:
    api_key = embedding_api_key()
    if api_key is None:
        raise EmbeddingSearchError("Missing embedding setup: set GEMINI_API_KEY in env or .env.")

    payload = {
        "model": f"models/{DEFAULT_EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
        "output_dimensionality": DEFAULT_EMBEDDING_DIMENSIONALITY,
    }
    response = post_embedding_json(
        f"{EMBEDDING_BASE_URL}/models/{DEFAULT_EMBEDDING_MODEL}:embedContent",
        api_key,
        payload,
    )
    embedding = response.get("embedding")
    values = embedding.get("values") if isinstance(embedding, dict) else response.get("values")
    if not isinstance(values, list):
        raise EmbeddingSearchError("Embedding response did not include values")
    return [float(value) for value in values]


def post_embedding_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not url.startswith(f"{EMBEDDING_BASE_URL}/"):
        raise EmbeddingSearchError("Unsupported embedding URL")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = compact_embedding_error(error.read().decode("utf-8", errors="replace"))
        raise EmbeddingSearchError(message or f"Embedding provider HTTP {error.code}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingSearchError(str(error)) from error


def embedding_api_key() -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    repo_root = find_repo_root(Path.cwd()) or find_repo_root(Path(__file__))
    if repo_root is None:
        return None
    return dotenv_values(repo_root / ".env").get("GEMINI_API_KEY")


def dotenv_values(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        value = raw_value.strip().strip("'\"")
        if key.strip():
            values[key.strip()] = value
    return values


def embedding_from_json(value: str) -> list[float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise EmbeddingSearchError("Stored embedding is not valid JSON") from error
    if not isinstance(parsed, list):
        raise EmbeddingSearchError("Stored embedding is not a vector")
    return [float(item) for item in parsed]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def compact_embedding_error(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return " ".join(text.split())[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return " ".join(str(error.get("message", "")).split())[:500]
    return ""
