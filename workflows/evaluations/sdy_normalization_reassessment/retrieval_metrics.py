"""Small deterministic graded-retrieval evaluator with exact score-tie means."""
import itertools
import math


def metrics_for_ranking(records, k=10):
    """records: unique (sample_id, finite score, complete grade in 0/1/2)."""
    if k != 10 or len(records) < k:
        raise ValueError('This protocol requires k=10 and at least ten candidates')
    if len({r[0] for r in records}) != len(records):
        raise ValueError('Duplicate candidate')
    if any(not math.isfinite(float(score)) or grade not in (0, 1, 2) for _, score, grade in records):
        raise ValueError('Finite scores and complete grades 0/1/2 are required')
    ranked = sorted(records, key=lambda r: (-r[1], r[0]))
    discount = [1 / math.log2(i+2) for i in range(k)]
    ideal = sorted([row[2] for row in records], reverse=True)[:k]
    idcg = sum((2**g-1)*w for g, w in zip(ideal, discount))
    positive = sum(row[2] > 0 for row in records)
    if not idcg or not positive:
        raise ValueError('No positive judged candidates')
    top = ranked[:k]
    deterministic = sum((2**row[2]-1)*w for row, w in zip(top, discount)) / idcg
    relevant, strong = sum(r[2] > 0 for r in top), sum(r[2] == 2 for r in top)
    expected_dcg = expected_relevant = expected_strong = 0.0
    offset = cutoff_tie = 0
    sizes = []
    for _, block in itertools.groupby(ranked, key=lambda r: r[1]):
        block = list(block)
        size = len(block)
        sizes.append(size)
        stop = min(offset + size, k)
        take = max(stop - offset, 0)
        if offset < k <= offset + size:
            cutoff_tie = size
        if take:
            expected_dcg += sum(discount[offset:stop]) * sum(2**r[2]-1 for r in block) / size
            expected_relevant += take * sum(r[2] > 0 for r in block) / size
            expected_strong += take * sum(r[2] == 2 for r in block) / size
        offset += size
    return {
        'ndcg_at_k': deterministic, 'precision_at_k': relevant/k,
        'recall_at_10': relevant/positive, 'upper_quartile_discovery_at_10': strong/k,
        'tie_expected_ndcg_at_10': expected_dcg/idcg,
        'tie_expected_precision_at_10': expected_relevant/k,
        'tie_expected_recall_at_10': expected_relevant/positive,
        'tie_expected_strong_discovery_at_10': expected_strong/k,
        'unique_scores': len(sizes), 'maximum_tie_size': max(sizes), 'cutoff_tie_size': cutoff_tie,
        'judged_at_k': k, 'k': k,
    }
