"""Local embedding clients put search queries and transcript texts on the query side, documents and frames on the document side, and load frames one batch at a time."""

import numpy as np
from PIL import Image

from youtube_toolkit.embeddings import local


class Encoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def encode_query(self, text, normalize_embeddings):
        assert normalize_embeddings
        self.calls.append(("query", text))
        return np.ones(2) if isinstance(text, str) else np.ones((len(text), 2))

    def encode_document(self, items, normalize_embeddings):
        assert normalize_embeddings
        self.calls.append(("document", len(items)))
        return np.zeros((len(items), 2))

    def get_embedding_dimension(self) -> int:
        return 2


def test_text_client_embeds_a_single_text_as_query_and_lists_as_documents(
    monkeypatch,
) -> None:
    encoder = Encoder()
    monkeypatch.setattr(local, "_load", lambda _model: encoder)
    with local.LocalTextEmbeddingClient() as client:
        assert client.embed("lens diagram") == [1.0, 1.0]
        assert client.embed(["title", "chunk", "title"]) == (
            {"title": [0.0, 0.0], "chunk": [0.0, 0.0]},
            {},
        )
    assert encoder.calls == [("query", "lens diagram"), ("document", 2)]


def test_image_client_embeds_frames_in_batches_as_documents(
    monkeypatch, tmp_path
) -> None:
    encoder = Encoder()
    monkeypatch.setattr(local, "_load", lambda _model: encoder)
    paths = []
    for index in range(local.IMAGE_BATCH_SIZE + 2):
        path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (16, 9)).save(path)
        paths.append(path)

    result = local.LocalImageEmbeddingClient().embed(paths, [])

    assert encoder.calls == [("document", local.IMAGE_BATCH_SIZE), ("document", 2)]
    assert result.images.shape == (len(paths), 2)
    assert result.texts.shape == (0, 2)
    assert (result.usage.image_tokens, result.usage.text_tokens) == (None, None)
