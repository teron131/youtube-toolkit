"""Local embeddings through sentence-transformers for the Hugging Face model ID in EMBEDDING_LOCAL_MODEL, used by semantic search and vision when no embedding API key is set.

sentence-transformers reads each model's own pooling, normalization, and query or document prompts, so any model ID it loads plugs in; vision also needs one that embeds images.
Models load with trust_remote_code because multimodal embedding models ship their own code, which then runs on this machine.
Each client loads the model once on the best available device, downloading it the first time it is needed.
The heavy libraries are imported inside the functions that need them, so commands that never embed locally do not pay for loading them.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Self, overload

from ..config import local_embedding_model
from ..huggingface import model_downloaded, quiet_transformers, torch_device
from .results import EmbeddingResult, EmbeddingUsage

IMAGE_BATCH_SIZE = 8


def local_embedding_installed() -> bool:
    """Installing the libraries is what switches the local route on; the model itself has a default."""
    try:
        return all(
            find_spec(module) is not None
            for module in ("sentence_transformers", "torch")
        )
    except (ImportError, ValueError):
        return False


def local_embedding_downloaded(model: str) -> bool:
    return local_embedding_installed() and model_downloaded(model)


class LocalTextEmbeddingClient:
    """Stands in for the OpenAI-compatible client in semantic search, which embeds a single text only for a query and lists only for titles and transcript chunks.

    A single text is therefore embedded as a query and a list as documents, the two sides retrieval models are trained on.
    """

    def __init__(self) -> None:
        self.model = local_embedding_model()
        self._encoder = _load(self.model)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(
        self, text: list[str]
    ) -> tuple[dict[str, list[float]], dict[str, str]]: ...

    def embed(
        self, text: str | list[str]
    ) -> list[float] | tuple[dict[str, list[float]], dict[str, str]]:
        """A list embeds each distinct text once; the failures map stays empty because a local model fails the whole batch or none of it."""
        if isinstance(text, list):
            unique = list(dict.fromkeys(text))
            if not unique:
                return {}, {}
            vectors = self._encoder.encode_document(unique, normalize_embeddings=True)
            return {
                item: vector.tolist()
                for item, vector in zip(unique, vectors, strict=True)
            }, {}
        return self._encoder.encode_query(text, normalize_embeddings=True).tolist()


class LocalImageEmbeddingClient:
    """Stands in for the Gemini client in vision: frames are embedded as documents and transcript texts as queries, and usage is never reported."""

    def __init__(self) -> None:
        self.model = local_embedding_model()
        self._encoder = _load(self.model)

    def embed(self, image_paths: list[Path], texts: list[str]) -> EmbeddingResult:
        import numpy as np
        from PIL import Image

        empty = np.empty((0, self._encoder.get_embedding_dimension()), dtype=np.float32)
        # Opening every candidate at once would hold about a gigabyte of pixels for a long video, so images load one batch at a time.
        image_batches = []
        for start in range(0, len(image_paths), IMAGE_BATCH_SIZE):
            images = []
            for path in image_paths[start : start + IMAGE_BATCH_SIZE]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            image_batches.append(
                self._encoder.encode_document(images, normalize_embeddings=True)
            )
        return EmbeddingResult(
            np.concatenate(image_batches).astype(np.float32)
            if image_batches
            else empty,
            self._encoder.encode_query(texts, normalize_embeddings=True).astype(
                np.float32
            )
            if texts
            else empty,
            EmbeddingUsage(None, None, None),
        )


def _load(model: str) -> Any:
    quiet_transformers(downloaded=local_embedding_downloaded(model))
    from sentence_transformers import SentenceTransformer
    import torch

    device = torch_device()
    # On an Apple GPU half precision ran about four times faster than full precision in under half the memory, and the similarity cutoffs were measured in bfloat16; CPUs need full precision.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    # Cast after loading: models with their own loading code, including the default, ignore a requested dtype.
    return SentenceTransformer(model, device=device, trust_remote_code=True).to(dtype)
