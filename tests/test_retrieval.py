"""Unit tests for BM25 and Hybrid retrieval algorithms."""

from vault.retrieval import BM25Retriever, HybridRetriever
from vault.vector_store import Document, VectorStore


class MockEmbedder:
    """Mock embedder returning deterministic vectors based on text content."""

    def embed_text(self, text: str) -> list[float]:
        if "python" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def test_bm25_retriever_keyword_match() -> None:
    """Verify BM25 sparse keyword scoring."""
    bm25 = BM25Retriever()
    docs = [
        Document(doc_id="d1", content="Vault uses Python 3.11 for core pipeline."),
        Document(doc_id="d2", content="Enterprise database backend setup in Java."),
    ]
    bm25.add_documents(docs)

    results = bm25.search("Python 3.11", top_k=2)
    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_hybrid_retriever_rrf() -> None:
    """Verify hybrid retriever combines vector and BM25 search via RRF."""
    store = VectorStore()
    embedder = MockEmbedder()
    hybrid = HybridRetriever(vector_store=store, embedder=embedder)

    docs = [
        Document(
            doc_id="d1",
            content="Python vector database implementation",
            embedding=[1.0, 0.0, 0.0],
        ),
        Document(
            doc_id="d2",
            content="Specific error code ERR_99482 in system logs",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]
    hybrid.add_documents(docs)

    # Search query matching sparse keyword ERR_99482
    results = hybrid.search(query="What is ERR_99482?", top_k=2)
    assert len(results) >= 1
    # Check that d2 was retrieved via sparse matching
    doc_ids = [r.doc_id for r in results]
    assert "d2" in doc_ids
