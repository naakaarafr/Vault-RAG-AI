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
        ignore_dimension_mismatch: bool = False,
    ) -> list[SearchResult]:
        """Search top-k most similar documents using cosine similarity."""
        if not self.documents or not query_embedding:
            return []

        doc_list: list[Document] = []
        matrix_rows: list[list[float]] = []

        target_dim = len(query_embedding)

        # Filter by metadata if provided and validate dimension match
        for doc in self.documents.values():
            if not doc.embedding:
                continue
            doc_dim = len(doc.embedding)
            if doc_dim != target_dim:
                if ignore_dimension_mismatch:
                    continue
                raise ValueError(
                    f"Vector dimension mismatch for doc '{doc.doc_id}': "
                    f"expected {target_dim}, got {doc_dim}."
                )
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


class QdrantVectorStore:
    """Vector database backend wrapping Qdrant server via qdrant-client with RBAC payload filtering."""

    def __init__(
        self,
        url: str | None = None,
        collection_name: str = "vault_chunks",
        vector_size: int = 384,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from vault.config import get_settings

        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        if self.url.startswith("http://") or self.url.startswith("https://"):
            self.client = QdrantClient(url=self.url)
        elif self.url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=self.url)


        # Initialize collection if not exists
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
        except Exception as err:
            raise RuntimeError(f"Failed to connect to Qdrant at '{self.url}': {err}") from err


    @property
    def documents(self) -> dict[str, Document]:
        """Expose stored points as documents dictionary for interface compatibility."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        res: dict[str, Document] = {}
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            for pt in points:
                payload = pt.payload or {}
                emb = pt.vector if isinstance(pt.vector, list) else []
                res[str(pt.id)] = Document(
                    doc_id=str(pt.id),
                    content=payload.get("text", ""),
                    metadata=payload,
                    embedding=emb,
                )
        except Exception:
            pass
        return res

    def add_documents(self, docs: list[Document]) -> None:
        """Upsert Document objects as points with payload into Qdrant."""
        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        for doc in docs:
            payload = dict(doc.metadata)
            payload["text"] = doc.content
            payload["doc_id"] = doc.metadata.get("doc_id", doc.doc_id)
            payload["chunk_id"] = doc.doc_id
            payload["page"] = doc.metadata.get("page", 1)
            payload["allowed_roles"] = doc.metadata.get("allowed_roles", ["public"])

            # Use hash of doc_id if string is non-numeric, or str format
            import uuid
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.doc_id))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=doc.embedding,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve point by document/chunk ID."""
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
        docs = self.documents
        return docs.get(point_id) or docs.get(doc_id)


    def delete_document(self, doc_id: str) -> bool:
        """Delete point from Qdrant by document/chunk ID."""
        import uuid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
            return True
        except Exception:
            return False

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        roles: list[str] | None = None,
        ignore_dimension_mismatch: bool = False,
    ) -> list[SearchResult]:
        """Search Qdrant collection with vector similarity and RBAC payload filter."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        if filter_metadata:
            for k, v in filter_metadata.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))

        # RBAC Payload Filter: Match allowed_roles payload
        if roles:
            role_conditions = [
                FieldCondition(key="allowed_roles", match=MatchValue(value=role))
                for role in roles
            ]
            qdrant_filter = Filter(must=conditions, should=role_conditions)
        else:
            qdrant_filter = Filter(must=conditions) if conditions else None

        if hasattr(self.client, "query_points"):
            query_res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            search_results = query_res.points
        else:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
            )


        results: list[SearchResult] = []
        for hit in search_results:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    doc_id=payload.get("chunk_id", str(hit.id)),
                    content=payload.get("text", ""),
                    metadata=payload,
                    score=float(hit.score),
                )
            )
        return results


def get_vector_store(backend: str | None = None) -> VectorStore | QdrantVectorStore:
    """Factory creating vector store backend based on config settings or backend argument."""
    from vault.config import get_settings
    settings = get_settings()
    active_backend = backend or settings.vector_backend

    if active_backend == "qdrant":
        return QdrantVectorStore()
    
    persistence_file = Path(settings.data_dir) / "vector_store.json"
    return VectorStore(persistence_path=persistence_file)


