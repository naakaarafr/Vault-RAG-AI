import json
import os
import time

from vault.agent import AgentTools, VaultAgent

DATASET_PATH = "eval/dataset.jsonl"
THRESHOLD_PATH = "results/threshold.json"
OUTPUT_JSON = f"results/agent_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
OUTPUT_MD = "results/latest_agent_eval.md"

def run_agent_evaluation(dataset_path: str = DATASET_PATH) -> dict:
    """
    Evaluates VaultAgent against dataset.jsonl.
    Computes Refusal Precision, Refusal Recall, Refusal F1, and Overall Accuracy.
    Writes Markdown summary and JSON report.
    """
    threshold = 0.35
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, encoding="utf-8") as f:
                data = json.load(f)
                threshold = data.get("calibrated_threshold", 0.35)
        except Exception:
            pass

    tools = AgentTools()
    agent = VaultAgent(tools=tools, refuse_threshold=threshold)

    questions = []
    if os.path.exists(dataset_path):
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("verified", False):
                    questions.append(item)

    if not questions:
        print("No verified questions found for agent evaluation.")
        return {}

    tp = 0  # True Positives: Unanswerable & Refused
    fp = 0  # False Positives: Answerable & Refused
    tn = 0  # True Negatives: Answerable & Answered
    fn = 0  # False Negatives: Unanswerable & Answered

    details = []

    for item in questions:
        q = item["question"]
        is_answerable = item.get("answerable", True)

        response = agent.run(question=q, role="admin")

        if not is_answerable:
            if response.refused:
                tp += 1
            else:
                fn += 1
        else:
            if response.refused:
                fp += 1
            else:
                tn += 1

        details.append({
            "qid": item.get("qid"),
            "question": q,
            "answerable": is_answerable,
            "refused": response.refused,
            "refusal_reason": response.refusal_reason,
            "best_rerank_score": response.best_rerank_score,
            "citations": response.citations
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(questions) if questions else 0.0

    eval_result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": threshold,
        "total_questions": len(questions),
        "unanswerable_count": tp + fn,
        "answerable_count": tn + fp,
        "metrics": {
            "refusal_precision": round(precision, 4),
            "refusal_recall": round(recall, 4),
            "refusal_f1": round(f1, 4),
            "overall_accuracy": round(accuracy, 4)
        },
        "details": details
    }

    # Save JSON report
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    # Save Markdown report
    md_content = (
        f"# Agent Refusal Evaluation Report\n\n"
        f"* **Timestamp**: {eval_result['timestamp']}\n"
        f"* **Calibrated Refusal Threshold**: `{threshold}`\n"
        f"* **Verified Dataset Size**: {len(questions)} questions\n\n"
        f"| Metric | Score |\n"
        f"| :--- | :---: |\n"
        f"| **Refusal Precision** | {precision:.4f} |\n"
        f"| **Refusal Recall** | {recall:.4f} |\n"
        f"| **Refusal F1** | {f1:.4f} |\n"
        f"| **Overall Accuracy** | {accuracy:.4f} |\n"
    )
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Agent evaluation complete. Saved to {OUTPUT_JSON} and {OUTPUT_MD}")
    return eval_result

if __name__ == "__main__":
    run_agent_evaluation()
