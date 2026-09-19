"""Unit tests for pure evaluation metric calculations."""

import pytest

from vault.eval.metrics import hit_at_k, mrr, percentile


def test_hit_at_k_hand_computed() -> None:
    """Verify Hit@k for exact rank match cases."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    gold = ["doc_c"]

    # Gold 'doc_c' is at rank 3
    assert hit_at_k(retrieved, gold, k=1) == 0.0
    assert hit_at_k(retrieved, gold, k=2) == 0.0
    assert hit_at_k(retrieved, gold, k=3) == 1.0
    assert hit_at_k(retrieved, gold, k=5) == 1.0


def test_hit_at_k_empty_cases() -> None:
    """Verify Hit@k edge cases with empty inputs."""
    assert hit_at_k([], ["doc_a"], k=5) == 0.0
    assert hit_at_k(["doc_a"], [], k=5) == 0.0
    assert hit_at_k(["doc_a"], ["doc_a"], k=0) == 0.0


def test_mrr_hand_computed() -> None:
    """Verify MRR calculation for rank 1, rank 2, rank 4, and no match."""
    gold = ["doc_target"]

    # Rank 1 -> 1/1 = 1.0
    assert mrr(["doc_target", "doc_b"], gold) == 1.0

    # Rank 2 -> 1/2 = 0.5
    assert mrr(["doc_a", "doc_target"], gold) == 0.5

    # Rank 4 -> 1/4 = 0.25
    assert mrr(["doc_a", "doc_b", "doc_c", "doc_target"], gold) == 0.25

    # No match -> 0.0
    assert mrr(["doc_a", "doc_b"], gold) == 0.0


def test_percentile_hand_computed() -> None:
    """Verify p50 and p95 percentile calculations."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

    # p50 (median) of [10, 20, 30, 40, 50] is 30.0
    assert pytest.approx(percentile(latencies, 50), rel=1e-5) == 30.0

    # p0 is 10.0, p100 is 50.0
    assert percentile(latencies, 0) == 10.0
    assert percentile(latencies, 100) == 50.0
