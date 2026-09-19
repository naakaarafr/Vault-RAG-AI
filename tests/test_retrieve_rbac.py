"""Unit tests verifying zero RBAC leakage across Dense, BM25, and Hybrid retrievers."""

from vault.retrieve.bm25 import BM25Retriever
from vault.retrieve.dense import DenseRetriever
from vault.retrieve.hybrid import HybridRetriever
from vault.vector_store import Document, VectorStore


class MockEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_rbac_leak_prevention_across_all_retrievers() -> None:
    """Verify that a 'public' caller NEVER receives 'admin' or 'compliance' restricted chunks."""
    store = VectorStore()
    docs = [
        Document(
            doc_id="pub_1",
            content="Public company overview guidelines.",
            metadata={"doc_id": "pub_1", "page": 1, "allowed_roles": ["public"]},
            embedding=[1.0, 0.0],
        ),
        Document(
            doc_id="admin_1",
            content="Top secret admin encryption keys and security bypass codes.",
            metadata={"doc_id": "admin_1", "page": 1, "allowed_roles": ["admin"]},
            embedding=[1.0, 0.0],
        ),
        Document(
            doc_id="compliance_1",
            content="Compliance audit findings and internal investigation logs.",
            metadata={"doc_id": "compliance_1", "page": 1, "allowed_roles": ["compliance"]},
            embedding=[1.0, 0.0],
        ),
    ]
    store.add_documents(docs)

    dense = DenseRetriever(vector_store=store, embedder=MockEmbedder())
    bm25 = BM25Retriever()
    bm25.add_documents(docs)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    # 1. Test DenseRetriever with public role
    dense_hits = dense.search(query="security overview", k=10, roles=["public"])
    assert len(dense_hits) == 1
    assert dense_hits[0].doc_id == "pub_1"
    assert not any(h.doc_id in ("admin_1", "compliance_1") for h in dense_hits)

    # 2. Test BM25Retriever with public role
    bm25_hits = bm25.search(query="security overview", k=10, roles=["public"])
    assert len(bm25_hits) == 1
    assert bm25_hits[0].doc_id == "pub_1"
    assert not any(h.doc_id in ("admin_1", "compliance_1") for h in bm25_hits)

    # 3. Test HybridRetriever with public role
    hybrid_hits = hybrid.search(query="security overview", k=10, roles=["public"])
    assert len(hybrid_hits) == 1
    assert hybrid_hits[0].doc_id == "pub_1"
    assert not any(h.doc_id in ("admin_1", "compliance_1") for h in hybrid_hits)

    # 4. Verify admin caller DOES receive admin chunk
    admin_hits = hybrid.search(query="security overview", k=10, roles=["admin"])
    admin_doc_ids = [h.doc_id for h in admin_hits]
    assert "admin_1" in admin_doc_ids
