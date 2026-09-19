"""Dense vector retriever using VectorStore with RBAC payload filtering."""


from vault.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from vault.retrieve.protocol import Hit
from vault.vector_store import VectorStore


class DenseRetriever:
    """Dense vector retriever enforcing Role-Based Access Control (RBAC)."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or SentenceTransformerEmbedder()

    def search(
        self,
        query: str,
        k: int = 5,
        roles: list[str] | None = None,
    ) -> list[Hit]:
        """Perform dense vector similarity search with RBAC payload filtering."""
        if not query.strip() or not self.vector_store.documents:
            return []

        query_vector = self.embedder.embed_text(query)
        # Retrieve candidate matches from vector store
        raw_results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=len(self.vector_store.documents),  # Fetch candidate set for RBAC filter
        )

        hits: list[Hit] = []
        user_roles = set(roles) if roles else None

        for res in raw_results:
            metadata = res.metadata
            allowed_roles = metadata.get("allowed_roles", ["public"])
            if isinstance(allowed_roles, str):
                allowed_roles = [r.strip() for r in allowed_roles.split(";")]

            # RBAC Filter: Caller roles must intersect with chunk allowed_roles
            if user_roles is not None and not user_roles.intersection(set(allowed_roles)):
                continue

            hits.append(
                Hit(
                    chunk_id=res.doc_id,
                    doc_id=metadata.get("doc_id", res.doc_id),
                    page=metadata.get("page", 1),
                    text=res.content,
                    score=res.score,
                    metadata=metadata,
                )
            )

            if len(hits) >= k:
                break

        return hits
