"""Exact expected top-k metrics under uniform permutations within score ties."""
import itertools
import math


def expected_metrics(records, k=10):
    """Records are (sample_id, finite score, measured relevance grade 0/1/2)."""
    if k < 1 or len(records) < k:
        raise ValueError("Invalid top-k or insufficient candidates")
    if len({r[0] for r in records}) != len(records):
        raise ValueError("Duplicate sample")
    if any(not math.isfinite(float(score)) or grade not in (0, 1, 2)
           for _, score, grade in records):
        raise ValueError("Finite scores and complete measured grades are required")
    ordered = sorted(records, key=lambda r: (-r[1], r[0]))
    discounts = [1 / math.log2(i + 2) for i in range(k)]
    ideal = sorted((r[2] for r in records), reverse=True)[:k]
    idcg = sum((2**g - 1) * w for g, w in zip(ideal, discounts))
    dcg = relevant = strong = 0.0
    start = cutoff_tie_size = 0
    sizes = []
    for _, group in itertools.groupby(ordered, key=lambda r: r[1]):
        block = list(group)
        size = len(block)
        sizes.append(size)
        stop = min(start + size, k)
        take = max(stop - start, 0)
        if start < k <= start + size:
            cutoff_tie_size = size
        if take:
            dcg += sum(discounts[start:stop]) * sum(2**r[2] - 1 for r in block) / size
            relevant += take * sum(r[2] > 0 for r in block) / size
            strong += take * sum(r[2] == 2 for r in block) / size
        start += size
    total_relevant = sum(r[2] > 0 for r in records)
    return {
        "tie_expected_ndcg_at_10": dcg / idcg if idcg else None,
        "tie_expected_precision_at_10": relevant / k,
        "tie_expected_recall_at_10": relevant / total_relevant if total_relevant else None,
        "tie_expected_strong_discovery_at_10": strong / k,
        "unique_scores": len(sizes), "maximum_tie_size": max(sizes),
        "cutoff_tie_size": cutoff_tie_size,
    }
