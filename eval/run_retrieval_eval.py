"""Evaluation runner benchmarking dense, hybrid, and hybrid_rerank retrieval configurations."""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vault.eval.metrics import hit_at_k, mrr, percentile
from vault.rerank.reranker import CrossEncoderReranker
from vault.retrieve.bm25 import BM25Retriever
from vault.retrieve.dense import DenseRetriever
from vault.retrieve.hybrid import HybridRetriever


def set_seed(seed: int = 42) -> None:
    """Set random seed for deterministic evaluation runs."""
    random.seed(seed)


def load_verified_dataset(dataset_path: str = "eval/dataset.jsonl") -> list[dict[str, Any]]:
    """Load evaluation questions filtering strictly for verified == True."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    dataset: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("verified") is True:
                dataset.append(item)

    return dataset


def run_evaluation(dataset_path: str = "eval/dataset.jsonl", is_smoke: bool = False) -> dict[str, Any]:
    """Execute end-to-end evaluation across dense, hybrid, and hybrid_rerank configurations."""
    set_seed(42)
    verified_items = load_verified_dataset(dataset_path)
    num_questions = len(verified_items)

    print(f"Loaded {num_questions} verified evaluation questions from '{dataset_path}'.")

    # (2) Fail if verified questions < 50 unless run with --smoke
    if num_questions < 50 and not is_smoke:
        raise RuntimeError(
            f"Evaluation FAILED: Verified questions count ({num_questions}) is less than 50. "
            f"Pass '--smoke' flag to allow evaluating small test sets."
        )

    # Initialize retrievers
    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()

    backend_class = dense_retriever.vector_store.__class__.__name__
    chunks = dense_retriever.vector_store.documents
    chunk_count = len(chunks)
    
    # Calculate unique doc_ids from metadata or documents
    unique_doc_ids = set()
    indexed_doc_ids = set()
    
    if isinstance(chunks, dict):
        chunk_items = list(chunks.values())
    else:
        chunk_items = list(chunks)

    for doc in chunk_items:
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("metadata", {}).get("doc_id")
        else:
            doc_id = getattr(doc, "doc_id", None) or getattr(doc, "metadata", {}).get("doc_id", None)
            if hasattr(doc, "metadata") and doc.metadata:
                doc_id = doc.metadata.get("doc_id", doc_id)

        if doc_id:
            unique_doc_ids.add(str(doc_id))
            indexed_doc_ids.add(str(doc_id))

    doc_count = len(unique_doc_ids)

    print(f"DenseRetriever backend class: {backend_class}")
    # (3) Print chunk count and doc count
    print(f"Vector Store chunk count: {chunk_count}")
    print(f"Vector Store unique document count: {doc_count}")

    if chunk_count == 0:
        raise RuntimeError(
            f"Evaluation FAILED: Dense vector database via {backend_class} is empty (0 points/chunks). "
            f"Please run ingestion or ensure database persistence is initialized before evaluation."
        )

    # (1) Verify every gold_doc_id in eval/dataset.jsonl exists in the index
    missing_gold_docs = set()
    for item in verified_items:
        for gold_id in item.get("gold_doc_ids", []):
            if str(gold_id) not in indexed_doc_ids:
                missing_gold_docs.add(str(gold_id))

    if missing_gold_docs:
        raise RuntimeError(
            f"Evaluation FAILED: The following gold_doc_ids in evaluation dataset do NOT exist in the index: "
            f"{sorted(list(missing_gold_docs))}"
        )


    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )
    reranker = CrossEncoderReranker()

    configs = ["dense", "hybrid", "hybrid_rerank"]
    results_by_config: dict[str, dict[str, Any]] = {}

    for config_name in configs:
        print(f"\nEvaluating configuration: [{config_name}]...")

        # Run 5 throwaway warmup queries per config BEFORE timing
        warmup_query = verified_items[0]["question"] if verified_items else "warmup query"
        for _ in range(5):
            if config_name == "dense":
                _ = dense_retriever.search(query=warmup_query, k=10, roles=["public", "admin"])
            elif config_name == "hybrid":
                _ = hybrid_retriever.search(query=warmup_query, k=10, roles=["public", "admin"])
            elif config_name == "hybrid_rerank":
                cands = hybrid_retriever.search(query=warmup_query, k=30, roles=["public", "admin"])
                _ = reranker.rerank(query=warmup_query, hits=cands, top_k=10)

        hits_at_1: list[float] = []
        hits_at_5: list[float] = []
        hits_at_10: list[float] = []
        mrr_scores: list[float] = []
        latencies_ms: list[float] = []

        # Time ALL evaluation questions
        for item in verified_items:
            question = item["question"]
            gold_doc_ids = item.get("gold_doc_ids", [])
            answerable = item.get("answerable", True)

            t0 = time.perf_counter()
            if config_name == "dense":
                hits = dense_retriever.search(query=question, k=10, roles=["public", "admin"])
            elif config_name == "hybrid":
                hits = hybrid_retriever.search(query=question, k=10, roles=["public", "admin"])
            elif config_name == "hybrid_rerank":
                candidates = hybrid_retriever.search(
                    query=question, k=30, roles=["public", "admin"]
                )
                hits = reranker.rerank(query=question, hits=candidates, top_k=10)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            retrieved_ids = [h.doc_id for h in hits]

            if answerable:
                hits_at_1.append(hit_at_k(retrieved_ids, gold_doc_ids, k=1))
                hits_at_5.append(hit_at_k(retrieved_ids, gold_doc_ids, k=5))
                hits_at_10.append(hit_at_k(retrieved_ids, gold_doc_ids, k=10))
                mrr_scores.append(mrr(retrieved_ids, gold_doc_ids))

        # Aggregate metrics over timed queries
        count_ans = len(hits_at_1) if hits_at_1 else 1
        avg_hit1 = sum(hits_at_1) / count_ans
        avg_hit5 = sum(hits_at_5) / count_ans
        avg_hit10 = sum(hits_at_10) / count_ans
        avg_mrr = sum(mrr_scores) / count_ans

        results_by_config[config_name] = {
            "hit_at_1": round(avg_hit1, 4),
            "hit_at_5": round(avg_hit5, 4),
            "hit_at_10": round(avg_hit10, 4),
            "mrr": round(avg_mrr, 4),
            "p50_latency_ms": round(percentile(latencies_ms, 50), 3),
            "p95_latency_ms": round(percentile(latencies_ms, 95), 3),
        }


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"retrieval_{timestamp}.json"
    markdown_path = results_dir / "latest_retrieval.md"

    output_payload = {
        "timestamp": timestamp,
        "dataset_size": num_questions,
        "configurations": results_by_config,
    }

    # 1. Save timestamped JSON output
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # 2. Save latest_retrieval.md markdown table
    md_content = generate_markdown_report(output_payload)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nEvaluation completed. Results saved to {json_path} and {markdown_path}.")
    print("\n" + md_content)

    return output_payload


def generate_markdown_report(payload: dict[str, Any]) -> str:
    """Generate human-readable Markdown summary table."""
    size = payload["dataset_size"]
    ts = payload["timestamp"]
    configs = payload["configurations"]

    lines = [
        "# Retrieval Evaluation Summary Report",
        "",
        f"* **Timestamp**: {ts}",
        f"* **Verified Dataset Size**: {size} questions",
        "",
        "| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | p50 Latency (ms) | p95 Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cfg_name, metrics in configs.items():
        row = (
            f"| `{cfg_name}` | {metrics['hit_at_1']:.4f} | {metrics['hit_at_5']:.4f} | "
            f"{metrics['hit_at_10']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['p50_latency_ms']:.3f} | {metrics['p95_latency_ms']:.3f} |"
        )
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval Evaluation Benchmark")
    parser.add_argument("dataset_file", nargs="?", default="eval/dataset.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--smoke", action="store_true", help="Allow running evaluation on datasets with < 50 questions")
    args = parser.parse_args()

    run_evaluation(dataset_path=args.dataset_file, is_smoke=args.smoke)

