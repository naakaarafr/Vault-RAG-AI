"""Local embedding generation for Vault.

Uses sentence-transformers locally with zero remote network calls at runtime.
"""

from typing import Protocol

import numpy as np

from vault.config import Settings, get_settings


class BaseEmbedder(Protocol):
    """Protocol interface for embedding models."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string into a vector representation."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings into vector representations."""
        ...


class SentenceTransformerEmbedder:
    """Local SentenceTransformer embedding model implementation."""

    def __init__(
        self,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embed_model
        self._model = None

    @property
    def model(self):
        """Lazy loader for sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load local embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        if not text:
            return []
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Normalize for unit length cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = embeddings / norms
        return normalized_embeddings.tolist()
