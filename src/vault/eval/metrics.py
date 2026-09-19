"""Pure metric functions for retrieval precision and latency benchmarks."""


def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    """Calculate Hit@k binary metric (1.0 if any gold doc is within top-k hits, 0.0 otherwise)."""
    if not retrieved_doc_ids or not gold_doc_ids or k <= 0:
        return 0.0

    top_k_retrieved = set(retrieved_doc_ids[:k])
    gold_set = set(gold_doc_ids)

    return 1.0 if top_k_retrieved.intersection(gold_set) else 0.0


def mrr(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) (1 / rank of first matching gold doc)."""
    if not retrieved_doc_ids or not gold_doc_ids:
        return 0.0

    gold_set = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank

    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Calculate the p-th percentile (0 <= p <= 100) of a list of float values."""
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(sorted_vals) - 1)
    fraction = rank - lower_idx

    return sorted_vals[lower_idx] + fraction * (sorted_vals[upper_idx] - sorted_vals[lower_idx])
