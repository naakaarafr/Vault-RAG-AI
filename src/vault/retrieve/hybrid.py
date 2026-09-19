"""Hybrid retriever using Reciprocal Rank Fusion (RRF k=60) over dense and BM25 retrievers."""

from vault.retrieve.bm25 import BM25Retriever
from vault.retrieve.dense import DenseRetriever
from vault.retrieve.protocol import Hit


class HybridRetriever:
    """Hybrid retriever combining dense vector and sparse BM25 search using RRF with RBAC."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = 5,
        roles: list[str] | None = None,
        fetch_top_n: int = 30,
    ) -> list[Hit]:
        """Perform hybrid search over top-30 dense and top-30 BM25 candidates via RRF."""
        if not query.strip():
            return []

        # 1. Fetch top-30 candidates with RBAC filtering from both retrievers
        dense_hits = self.dense_retriever.search(query=query, k=fetch_top_n, roles=roles)
        sparse_hits = self.bm25_retriever.search(query=query, k=fetch_top_n, roles=roles)

        # 2. Reciprocal Rank Fusion (RRF) formula calculation
        rrf_scores: dict[str, float] = {}
        hit_map: dict[str, Hit] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            chunk_id = hit.chunk_id
            hit_map[chunk_id] = hit
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, hit in enumerate(sparse_hits, start=1):
            chunk_id = hit.chunk_id
            hit_map[chunk_id] = hit
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        fused_hits: list[Hit] = []
        for chunk_id, rrf_score in sorted_chunks:
            original_hit = hit_map[chunk_id]
            fused_hits.append(
                Hit(
                    chunk_id=original_hit.chunk_id,
                    doc_id=original_hit.doc_id,
                    page=original_hit.page,
                    text=original_hit.text,
                    score=rrf_score,
                    metadata=original_hit.metadata,
                )
            )

        return fused_hits
