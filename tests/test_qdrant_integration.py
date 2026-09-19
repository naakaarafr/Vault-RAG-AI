import os
import pytest
import requests
from vault.config.settings import get_settings
from vault.vector_store import QdrantVectorStore

def is_qdrant_available() -> bool:
    settings = get_settings()
    url = f"{settings.qdrant_url.rstrip('/')}/healthz"
    try:
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except Exception:
        # Also try root or cluster info if healthz differs
        try:
            resp = requests.get(settings.qdrant_url, timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

@pytest.mark.skipif(not is_qdrant_available(), reason="Qdrant service is down or unreachable")
def test_qdrant_vector_store_integration():
    settings = get_settings()
    store = QdrantVectorStore(url=settings.qdrant_url, collection_name="test_integration_collection")
    
    # Test document upsert
    test_doc = {
        "id": "integration_test_chunk_1",
        "doc_id": "test_doc_1",
        "page": 1,
        "text": "This is an integration test chunk for Qdrant storage.",
        "allowed_roles": ["admin"],
        "embedding": [0.1] * settings.embedding_dim,
    }
    
    store.add_documents([test_doc])
    
    # Verify retrieval by ID
    retrieved = store.get_document("integration_test_chunk_1")
    assert retrieved is not None
    assert retrieved["doc_id"] == "test_doc_1"
    assert retrieved["allowed_roles"] == ["admin"]
    
    # Verify vector search with RBAC filter
    results = store.search(
        query_vector=[0.1] * settings.embedding_dim,
        top_k=5,
        allowed_roles=["admin"]
    )
    assert len(results) > 0
    assert results[0][0]["id"] == "integration_test_chunk_1"
    
    # Test RBAC filter exclusion
    public_results = store.search(
        query_vector=[0.1] * settings.embedding_dim,
        top_k=5,
        allowed_roles=["public"]
    )
    assert len(public_results) == 0
    
    # Clean up test document
    store.delete_document("integration_test_chunk_1")
