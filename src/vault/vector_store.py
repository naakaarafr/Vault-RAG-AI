"""Explicit local vector database engine with cosine similarity search and disk persistence."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Document:
    """Document unit stored within Vault vector store."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchResult:
    """Result returned from vector similarity query."""

    doc_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class VectorStore:
    """In-memory and disk-persisted vector store using NumPy cosine matrix math."""

    def __init__(self, persistence_path: str | Path | None = None) -> None:
        self.documents: dict[str, Document] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None
        if self.persistence_path and self.persistence_path.exists():
            self.load()

    def add_documents(self, docs: list[Document]) -> None:
        """Add or update documents in the vector index."""
        for doc in docs:
            self.documents[doc.doc_id] = doc
        if self.persistence_path:
            self.save()

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a stored document by doc_id."""
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a stored document by doc_id."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if self.persistence_path:
                self.save()
            return True
        return False

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search top-k most similar documents using cosine similarity."""
        if not self.documents or not query_embedding:
            return []

        doc_list: list[Document] = []
        matrix_rows: list[list[float]] = []

        # Filter by metadata if provided
        for doc in self.documents.values():
            if not doc.embedding:
                continue
            if filter_metadata and not self._matches_filter(doc.metadata, filter_metadata):
                continue
            doc_list.append(doc)
            matrix_rows.append(doc.embedding)

        if not doc_list:
            return []

        # Vector calculations using NumPy
        query_vec = np.array(query_embedding, dtype=np.float32)
        doc_matrix = np.array(matrix_rows, dtype=np.float32)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # Normalize matrix rows
        doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1.0
        doc_matrix_norm = doc_matrix / doc_norms

        # Cosine similarity dot product
        scores = np.dot(doc_matrix_norm, query_vec)

        # Sort indices descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            doc = doc_list[idx]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=float(scores[idx]),
                )
            )

        return results

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if document metadata matches filter constraints."""
        return all(metadata.get(key) == expected for key, expected in filters.items())


    def save(self) -> None:
        """Persist document state to disk in JSON format."""
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [asdict(doc) for doc in self.documents.values()]
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def load(self) -> None:
        """Load document state from disk JSON format."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        with open(self.persistence_path, encoding="utf-8") as f:
            data = json.load(f)
        self.documents = {item["doc_id"]: Document(**item) for item in data}
