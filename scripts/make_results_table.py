import glob
import json
import os

RESULTS_DIR = "results"

def find_latest_file(pattern: str) -> str | None:
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def generate_retrieval_table() -> str:
    latest_retrieval = find_latest_file("retrieval_*.json")
    if not latest_retrieval or not os.path.exists(latest_retrieval):
        return "| Config | Hit@1 | Hit@5 | Hit@10 | MRR | p50 Latency (ms) | p95 Latency (ms) |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n| *No results run yet* | - | - | - | - | - | - |"

    with open(latest_retrieval, encoding="utf-8") as f:
        data = json.load(f)

    lines = [
        "| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | p50 Latency (ms) | p95 Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for config, metrics in data.get("configurations", {}).items():
        hit1 = metrics.get("hit_at_1", 0.0)
        hit5 = metrics.get("hit_at_5", 0.0)
        hit10 = metrics.get("hit_at_10", 0.0)
        mrr = metrics.get("mrr", 0.0)
        p50 = metrics.get("p50_latency_ms", 0.0)
        p95 = metrics.get("p95_latency_ms", 0.0)
        lines.append(f"| `{config}` | {hit1:.4f} | {hit5:.4f} | {hit10:.4f} | {mrr:.4f} | {p50:.3f} | {p95:.3f} |")

    return "\n".join(lines)

def generate_agent_table() -> str:
    latest_agent = find_latest_file("agent_eval_*.json")
    if not latest_agent or not os.path.exists(latest_agent):
        return "| Metric | Score |\n| :--- | :---: |\n| *No agent results run yet* | - |"

    with open(latest_agent, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    lines = [
        "| Refusal / Guardrail Metric | Score |",
        "| :--- | :---: |",
        f"| **Refusal Precision** | {metrics.get('refusal_precision', 0.0):.4f} |",
        f"| **Refusal Recall** | {metrics.get('refusal_recall', 0.0):.4f} |",
        f"| **Refusal F1** | {metrics.get('refusal_f1', 0.0):.4f} |",
        f"| **Overall Accuracy** | {metrics.get('overall_accuracy', 0.0):.4f} |"
    ]
    return "\n".join(lines)

def generate_cost_table() -> str:
    return (
        "| Resource Component | Local Hardware Cost | External API Runtime Cost per 1k Queries |\n"
        "| :--- | :---: | :---: |\n"
        "| **LLM Inference** | Local Ollama / vLLM | **$0.00** |\n"
        "| **Dense Vector Embeddings** | `BAAI/bge-small-en-v1.5` | **$0.00** |\n"
        "| **Cross-Encoder Reranking** | `BAAI/bge-reranker-base` | **$0.00** |\n"
        "| **Total Operating Cost** | **Self-Hosted** | **$0.00** |"
    )

def main():
    print("### Retrieval Ablation Results")
    print(generate_retrieval_table())
    print("\n### Agent Guardrail & Refusal Results")
    print(generate_agent_table())
    print("\n### Query Operating Cost Breakdown")
    print(generate_cost_table())

if __name__ == "__main__":
    main()
