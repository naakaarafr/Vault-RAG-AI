"""Unit tests for CrossEncoderReranker score re-ordering."""

from vault.rerank.reranker import CrossEncoderReranker
from vault.retrieve.protocol import Hit


class MockCrossEncoderModel:
    """Mock CrossEncoder model predicting semantic relevance scores."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        # Return high relevance score for chunk mentioning encryption
        scores = []
        for _query, text in pairs:
            if "encryption" in text.lower():
                scores.append(0.95)
            else:
                scores.append(0.12)
        return scores


def test_reranker_ordering_fixture() -> None:
    """Verify reranker re-orders candidate hits based on query semantic relevance."""
    reranker = CrossEncoderReranker()
    reranker._model = MockCrossEncoderModel()

    query = "What encryption algorithm does Vault use?"

    hits = [
        Hit(
            chunk_id="c1",
            doc_id="d1",
            page=1,
            text="Vault uses Python 3.11 for all backend services and module logic.",
            score=0.90,
        ),
        Hit(
            chunk_id="c2",
            doc_id="d2",
            page=1,
            text="All memory buffers and disk stores use AES-256 local encryption.",
            score=0.40,
        ),
    ]

    reranked = reranker.rerank(query=query, hits=hits, top_k=2)

    assert len(reranked) == 2
    # The chunk explicitly discussing encryption should be reranked to top #1
    assert reranked[0].chunk_id == "c2"
    assert "AES-256 local encryption" in reranked[0].text
    assert reranked[0].score > reranked[1].score
