"""Unit tests for Vault explicit RAG pipeline."""

import pytest

from vault.config import Settings
from vault.llm import LLMClient
from vault.rag_pipeline import RAGPipeline
from vault.retrieval import HybridRetriever
from vault.vector_store import Document, VectorStore


class MockEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_rag_pipeline_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end RAG pipeline execution with mocked LLM response."""
    store = VectorStore()
    docs = [
        Document(
            doc_id="d1",
            content="Vault enterprise RAG uses pydantic-settings.",
            embedding=[1.0, 0.0],
        )
    ]
    retriever = HybridRetriever(vector_store=store, embedder=MockEmbedder())
    retriever.add_documents(docs)

    llm = LLMClient(Settings(llm_base_url="http://localhost:11434/v1"))

    def mock_generate(*args, **kwargs):
        return "Vault uses pydantic-settings for configuration."

    monkeypatch.setattr(llm, "generate", mock_generate)

    pipeline = RAGPipeline(retriever=retriever, llm_client=llm)

    result = pipeline.query("How does Vault manage configuration?")
    assert result.answer == "Vault uses pydantic-settings for configuration."
    assert len(result.sources) == 1
    assert result.sources[0].doc_id == "d1"
    assert "Vault enterprise RAG" in result.raw_context
