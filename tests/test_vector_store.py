"""Unit tests for Vault local vector store engine."""

from pathlib import Path

from vault.vector_store import Document, VectorStore


def test_add_and_get_document() -> None:
    """Verify storing and retrieving documents."""
    store = VectorStore()
    doc = Document(
        doc_id="doc1",
        content="Vault vector database",
        metadata={"category": "tech"},
        embedding=[1.0, 0.0, 0.0],
    )
    store.add_documents([doc])

    retrieved = store.get_document("doc1")
    assert retrieved is not None
    assert retrieved.content == "Vault vector database"
    assert retrieved.metadata["category"] == "tech"


def test_cosine_similarity_search() -> None:
    """Verify cosine vector search rank ordering."""
    store = VectorStore()
    docs = [
        Document(doc_id="d1", content="Python RAG", embedding=[1.0, 0.0, 0.0]),
        Document(doc_id="d2", content="Java Backend", embedding=[0.0, 1.0, 0.0]),
        Document(doc_id="d3", content="Python Agent", embedding=[0.9, 0.09, 0.0]),
    ]
    store.add_documents(docs)

    # Search for vector [1.0, 0.0, 0.0]
    query = [1.0, 0.0, 0.0]
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].doc_id == "d1"
    assert results[1].doc_id == "d3"
    assert results[0].score > results[1].score


def test_metadata_filtering() -> None:
    """Verify search metadata constraint filter."""
    store = VectorStore()
    docs = [
        Document(
            doc_id="d1",
            content="Confidential Doc",
            metadata={"confidential": True},
            embedding=[1.0, 0.0],
        ),
        Document(
            doc_id="d2",
            content="Public Doc",
            metadata={"confidential": False},
            embedding=[1.0, 0.0],
        ),
    ]
    store.add_documents(docs)

    results = store.search([1.0, 0.0], top_k=5, filter_metadata={"confidential": False})
    assert len(results) == 1
    assert results[0].doc_id == "d2"


def test_disk_persistence(tmp_path: Path) -> None:
    """Verify saving and loading vector store state from disk."""
    persistence_file = tmp_path / "vector_db.json"
    store = VectorStore(persistence_path=persistence_file)
    store.add_documents([
        Document(
            doc_id="p1",
            content="Persisted content",
            metadata={"env": "prod"},
            embedding=[0.5, 0.5],
        )
    ])

    # Re-instantiate store from disk
    new_store = VectorStore(persistence_path=persistence_file)
    doc = new_store.get_document("p1")
    assert doc is not None
    assert doc.content == "Persisted content"
    assert doc.embedding == [0.5, 0.5]
