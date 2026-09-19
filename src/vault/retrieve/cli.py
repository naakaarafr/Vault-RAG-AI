"""CLI query execution logic for vault.retrieve."""

from typing import Literal

from vault.retrieve.bm25 import BM25Retriever
from vault.retrieve.dense import DenseRetriever
from vault.retrieve.hybrid import HybridRetriever
from vault.retrieve.protocol import Hit


def run_retrieval_query(
    query: str,
    k: int = 5,
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
    roles: list[str] | None = None,
) -> list[Hit]:
    """Execute search query in specified mode with role-based filtering."""
    user_roles = roles or ["public"]

    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )

    if mode == "dense":
        hits = dense_retriever.search(query=query, k=k, roles=user_roles)
    elif mode == "bm25":
        hits = bm25_retriever.search(query=query, k=k, roles=user_roles)
    elif mode == "hybrid":
        hits = hybrid_retriever.search(query=query, k=k, roles=user_roles)
    elif mode == "hybrid_rerank":
        from vault.rerank.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        candidate_hits = hybrid_retriever.search(query=query, k=30, roles=user_roles)
        hits = reranker.rerank(query=query, hits=candidate_hits, top_k=k)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    print(f"\n=== Vault Retrieval Results (Mode: {mode} | Roles: {user_roles}) ===")
    print(f"Query: '{query}'\n")

    if not hits:
        print("No matching hits found.")
    else:
        for idx, h in enumerate(hits, start=1):
            details = f"Chunk: {h.chunk_id}, Page: {h.page}"
            print(f"[{idx}] Score: {h.score:.4f} | Doc ID: {h.doc_id} ({details})")
            print(f"    Text: {h.text[:120]}...\n")

    return hits
