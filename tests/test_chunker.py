"""Unit tests for Vault text chunker."""

import pytest

from vault.ingestion.chunker import TextChunker


def test_chunk_basic_text() -> None:
    """Verify basic text splitting into overlapping chunks."""
    text = (
        "Vault is a self-hosted enterprise RAG stack. "
        "It features a guardrailed agent operating locally with zero external API calls. "
        "All models run locally via vLLM or Ollama endpoints."
    )
    chunker = TextChunker(chunk_size=60, chunk_overlap=20, min_chunk_size=10)
    chunks = chunker.chunk_text(text, doc_id="test_doc")

    assert len(chunks) > 1
    assert chunks[0].metadata["doc_id"] == "test_doc"
    assert chunks[0].metadata["chunk_index"] == 0
    assert "Vault is a self-hosted" in chunks[0].text


def test_chunk_empty_text() -> None:
    """Verify empty text returns empty list."""
    chunker = TextChunker()
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n ") == []


def test_invalid_overlap_raises() -> None:
    """Verify ValueError is raised if overlap >= chunk_size."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
