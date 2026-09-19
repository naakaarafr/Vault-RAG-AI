"""Empirical benchmarking script for Vault RAG pipeline and vector store performance."""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add src to PYTHONPATH for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vault.ingestion.chunker import TextChunker
from vault.retrieval import HybridRetriever
from vault.vector_store import Document, VectorStore


class FastMockEmbedder:
    """Mock embedder generating deterministic 384-dim vectors for benchmark timing."""

    def embed_text(self, text: str) -> list[float]:
        h = sum(ord(c) for c in text) % 384
        vec = [0.01] * 384
        vec[h] = 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def run_benchmark() -> dict[str, Any]:
    """Execute end-to-end benchmark suite and return real runtime metrics."""
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample Dataset Generation
    sample_docs = [
        (
            "doc_sec_01",
            "Vault Security Architecture: All memory buffers use AES-256 local encryption. "
            "Zero network sockets are exposed outside localhost. "
            "Input guardrails scrub PII including emails, SSNs, credit cards, and API keys."
        ),
        (
            "doc_arch_02",
            "Hybrid Retrieval Engine: Vault combines dense vector search using normalized cosine "
            "similarity dot products with sparse BM25 term frequency-idf ranking via RRF. "
            "RRF parameter k is set to 60 for scale-invariant rank combination."
        ),
        (
            "doc_perf_03",
            "Performance & Scalability: Ingestion pipelines utilize sliding window text chunkers "
            "with sentence boundary protection. Vector indices support JSON serialization."
        ),
    ]

    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    store = VectorStore()
    embedder = FastMockEmbedder()
    retriever = HybridRetriever(vector_store=store, embedder=embedder)

    # Measure Document Chunking & Indexing
    t0 = time.perf_counter()
    total_chunks = 0
    for doc_id, text in sample_docs:
        chunks = chunker.chunk_text(text, doc_id=doc_id)
        total_chunks += len(chunks)
        docs = [
            Document(
                doc_id=c.chunk_id,
                content=c.text,
                metadata=c.metadata,
                embedding=embedder.embed_text(c.text),
            )
            for c in chunks
        ]
        retriever.add_documents(docs)
    ingestion_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Measure Query Performance
    benchmark_queries = [
        "How does Vault protect against PII leakage?",
        "What parameter k is used in Reciprocal Rank Fusion?",
        "What encryption is used in Vault Security Architecture?",
    ]

    latencies_vector_search_ms: list[float] = []
    latencies_hybrid_search_ms: list[float] = []

    for q in benchmark_queries:
        # Measure vector search
        t_start = time.perf_counter()
        q_vec = embedder.embed_text(q)
        store.search(q_vec, top_k=3)
        latencies_vector_search_ms.append((time.perf_counter() - t_start) * 1000.0)

        # Measure hybrid search
        t_start_h = time.perf_counter()
        retriever.search(q, top_k=3)
        latencies_hybrid_search_ms.append((time.perf_counter() - t_start_h) * 1000.0)

    avg_vector_latency = sum(latencies_vector_search_ms) / len(latencies_vector_search_ms)
    avg_hybrid_latency = sum(latencies_hybrid_search_ms) / len(latencies_hybrid_search_ms)

    metrics = {
        "dataset": {
            "num_source_documents": len(sample_docs),
            "total_chunks_created": total_chunks,
        },
        "ingestion": {
            "total_ingestion_time_ms": round(ingestion_time_ms, 3),
            "avg_time_per_chunk_ms": round(ingestion_time_ms / total_chunks, 3),
        },
        "retrieval_latency": {
            "avg_dense_vector_search_latency_ms": round(avg_vector_latency, 3),
            "avg_hybrid_rrf_search_latency_ms": round(avg_hybrid_latency, 3),
            "num_benchmark_queries": len(benchmark_queries),
        },
        "environment": {
            "python_version": "3.11",
            "vector_dimension": 384,
            "offline_mode": True,
        },
    }

    output_path = results_dir / "benchmark_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Benchmark completed successfully. Results written to {output_path}")
    return metrics


if __name__ == "__main__":
    run_benchmark()
