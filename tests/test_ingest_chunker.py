"""Unit tests for structure-aware document chunker."""

import pytest

from vault.ingest.chunker import StructureAwareChunker
from vault.ingest.loaders import ExtractedPage


def test_chunker_structure_heading_split() -> None:
    """Verify heading and clause structure boundaries trigger section splits."""
    chunker = StructureAwareChunker(max_tokens=200, token_overlap=20)
    text = (
        "Section 1. General Principles\n"
        "Vault is a self-hosted RAG architecture for enterprise deployments.\n\n"
        "Section 2. Security Protocols\n"
        "All data buffers are encrypted at rest with AES-256 keys."
    )
    pages = [ExtractedPage(page_number=1, text=text)]

    chunks = chunker.chunk_document(
        doc_id="sec_doc",
        title="Security Policy",
        allowed_roles=["admin", "compliance"],
        pages=pages,
    )

    assert len(chunks) == 2
    assert chunks[0].doc_id == "sec_doc"
    assert chunks[0].title == "Security Policy"
    assert chunks[0].page == 1
    assert chunks[0].allowed_roles == ["admin", "compliance"]
    assert "Section 1" in chunks[0].text
    assert "Section 2" in chunks[1].text


def test_chunker_token_budget_and_overlap() -> None:
    """Verify long text sub-splitting obeys max tokens and token overlap bounds."""
    chunker = StructureAwareChunker(max_tokens=30, token_overlap=10)
    long_text = "word " * 100  # 100 words
    pages = [ExtractedPage(page_number=1, text=long_text)]

    chunks = chunker.chunk_document(
        doc_id="long_doc",
        title="Long Text",
        allowed_roles=["public"],
        pages=pages,
    )

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["doc_id"] == "long_doc"
        assert c.metadata["allowed_roles"] == ["public"]


def test_chunker_invalid_overlap_raises() -> None:
    """Verify ValueError is raised if token_overlap >= max_tokens."""
    with pytest.raises(ValueError):
        StructureAwareChunker(max_tokens=100, token_overlap=100)
