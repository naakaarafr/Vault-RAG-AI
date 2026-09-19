"""Unit tests for Reciprocal Rank Fusion (RRF) rank math calculations."""

import pytest

from vault.retrieve.hybrid import HybridRetriever
from vault.retrieve.protocol import Hit


class DummyRetriever:
    """Mock retriever returning canned hit rankings for deterministic RRF math testing."""

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def search(self, query: str, k: int = 30, roles: list[str] | None = None) -> list[Hit]:
        return self.hits[:k]


def test_rrf_math_calculation() -> None:
    """Verify exact RRF score formula: RRF_Score = 1 / (60 + rank_dense) + 1 / (60 + rank_bm25)."""
    dense_hits = [
        Hit(chunk_id="A", doc_id="d1", page=1, text="Chunk A text", score=0.9),
        Hit(chunk_id="B", doc_id="d2", page=1, text="Chunk B text", score=0.8),
        Hit(chunk_id="C", doc_id="d3", page=1, text="Chunk C text", score=0.7),
    ]

    sparse_hits = [
        Hit(chunk_id="B", doc_id="d2", page=1, text="Chunk B text", score=15.0),
        Hit(chunk_id="A", doc_id="d1", page=1, text="Chunk A text", score=10.0),
    ]

    hybrid = HybridRetriever(
        dense_retriever=DummyRetriever(dense_hits),
        bm25_retriever=DummyRetriever(sparse_hits),
        rrf_k=60,
    )

    results = hybrid.search(query="test query", k=3)
    assert len(results) == 3

    expected_score_a = (1.0 / 61.0) + (1.0 / 62.0)
    expected_score_c = 1.0 / 63.0

    hit_map = {h.chunk_id: h.score for h in results}

    assert pytest.approx(hit_map["A"], abs=1e-5) == expected_score_a
    assert pytest.approx(hit_map["B"], abs=1e-5) == expected_score_a
    assert pytest.approx(hit_map["C"], abs=1e-5) == expected_score_c
    assert hit_map["A"] > hit_map["C"]
