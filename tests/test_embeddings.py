"""Unit tests for Vault local embedding generator."""

import numpy as np
import pytest

from vault.embeddings import SentenceTransformerEmbedder


class DummyModel:
    """Mock SentenceTransformer model for fast deterministic testing."""

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)
        return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)


def test_embed_text_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify single text embedding calculation and unit length normalization."""
    embedder = SentenceTransformerEmbedder(model_name="dummy")
    embedder._model = DummyModel()

    vector = embedder.embed_text("Self-hosted RAG")
    assert len(vector) == 3
    # Check L2 normalization length == 1.0
    assert pytest.approx(np.linalg.norm(vector), rel=1e-5) == 1.0


def test_embed_documents_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify batch text embedding calculation and normalization."""
    embedder = SentenceTransformerEmbedder(model_name="dummy")
    embedder._model = DummyModel()

    vectors = embedder.embed_documents(["doc 1", "doc 2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert pytest.approx(np.linalg.norm(vectors[0]), rel=1e-5) == 1.0


def test_embed_empty_input() -> None:
    """Verify empty text inputs return empty list."""
    embedder = SentenceTransformerEmbedder(model_name="dummy")
    embedder._model = DummyModel()

    assert embedder.embed_text("") == []
    assert embedder.embed_documents([]) == []
