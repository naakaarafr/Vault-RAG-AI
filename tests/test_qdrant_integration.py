import os
import pytest
import requests
from vault.config.settings import get_settings
from vault.vector_store import Document, QdrantVectorStore


def is_qdrant_available() -> bool:
    return True



@pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant service is down or unreachable")
def test_qdrant_vector_store_integration():
    settings = get_settings()
    url = settings.qdrant_url if is_qdrant_available() and settings.qdrant_url.startswith("http") else ":memory:"
    store = QdrantVectorStore(url=url, collection_name="test_integration_collection")

    test_doc = Document(
        doc_id="integration_test_chunk_1",
        content="This is an integration test chunk for Qdrant storage.",
        metadata={
            "doc_id": "test_doc_1",
            "page": 1,
            "allowed_roles": ["admin"],
        },
        embedding=[0.1] * 384,
    )

    store.add_documents([test_doc])

    # Verify retrieval
    retrieved = store.get_document("integration_test_chunk_1")
    assert retrieved is not None
    assert retrieved.metadata.get("doc_id") == "test_doc_1"
    assert retrieved.metadata.get("allowed_roles") == ["admin"]

    # Verify vector search with RBAC filter
    results = store.search(
        query_embedding=[0.1] * 384,
        top_k=5,
        roles=["admin"],
    )
    assert len(results) > 0
    assert results[0].doc_id == "integration_test_chunk_1"

    # Test RBAC filter exclusion
    public_results = store.search(
        query_embedding=[0.1] * 384,
        top_k=5,
        roles=["public"],
    )
    assert len(public_results) == 0

    # Clean up test document
    store.delete_document("integration_test_chunk_1")


