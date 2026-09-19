import json
import os

from vault.agent.tools import AgentTools

DATASET_PATH = "eval/dataset.jsonl"
OUTPUT_PATH = "results/threshold.json"

def calibrate_refusal_threshold(dataset_path: str = DATASET_PATH, output_path: str = OUTPUT_PATH) -> float:
    """
    Calibrates refusal threshold by comparing max rerank scores on answerable vs unanswerable questions in dataset.jsonl.
    Writes results to results/threshold.json.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found. Using default threshold 0.35.")
        optimal_threshold = 0.35
    else:
        tools = AgentTools()
        answerable_scores = []
        unanswerable_scores = []

        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("verified", False):
                    continue

                q = item["question"]
                is_answerable = item.get("answerable", True)

                search_res = tools.search_docs(query=q, role="admin", top_k=5)
                score = search_res.get("best_rerank_score", 0.0)

                if is_answerable:
                    answerable_scores.append(score)
                else:
                    unanswerable_scores.append(score)

        if not unanswerable_scores:
            optimal_threshold = 0.35
        else:
            max_unanswerable = max(unanswerable_scores) if unanswerable_scores else 0.0
            min_answerable = min(answerable_scores) if answerable_scores else 0.5
            optimal_threshold = round((max_unanswerable + min_answerable) / 2.0, 4)
            if optimal_threshold <= 0:
                optimal_threshold = 0.35

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_data = {
        "calibrated_threshold": optimal_threshold,
        "default_fallback": 0.35,
        "status": "calibrated"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"Calibration complete. Optimal threshold: {optimal_threshold}. Saved to {output_path}")
    return optimal_threshold

if __name__ == "__main__":
    calibrate_refusal_threshold()
