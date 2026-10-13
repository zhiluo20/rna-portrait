"""No-data unit tests for the published graded-retrieval helper."""
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

HERE = Path(__file__).resolve().parent
HELPER = HERE if (HERE / 'retrieval_metrics.py').is_file() else HERE.parent / 'workflows/evaluations/sdy_normalization_reassessment'
sys.path.insert(0, str(HELPER))
from retrieval_metrics import metrics_for_ranking


class RetrievalMetricTests(unittest.TestCase):
    def rows(self):
        return [(f's{i:02d}', float(20-i), 2 if i < 4 else (1 if i < 10 else 0)) for i in range(12)]

    def test_perfect_ranking(self):
        m = metrics_for_ranking(self.rows())
        self.assertAlmostEqual(m['ndcg_at_k'], 1)
        self.assertEqual(m['precision_at_k'], 1)
        self.assertEqual(m['recall_at_10'], 1)
        self.assertEqual(m['upper_quartile_discovery_at_10'], .4)

    def test_all_ties(self):
        rows = [(s, 1.0, g) for s, _, g in self.rows()]
        m = metrics_for_ranking(rows)
        weights = [1/math.log2(i+2) for i in range(10)]
        ideal = sum((2**g-1)*w for g, w in zip(sorted([r[2] for r in rows], reverse=True), weights))
        self.assertAlmostEqual(m['tie_expected_ndcg_at_10'], sum(weights)*sum(2**r[2]-1 for r in rows)/12/ideal)
        self.assertAlmostEqual(m['tie_expected_precision_at_10'], 10/12)
        self.assertAlmostEqual(m['tie_expected_recall_at_10'], 10/12)
        self.assertAlmostEqual(m['tie_expected_strong_discovery_at_10'], 4/12)
        self.assertEqual(m['cutoff_tie_size'], 12)

    def test_boundary_tie_by_exhaustive_permutations(self):
        rows = [(f's{i:02d}', float(20-i) if i < 6 else 1.0, i % 3) for i in range(12)]
        m = metrics_for_ranking(rows)
        ideal = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(sorted([r[2] for r in rows], reverse=True)[:10]))
        dcg, positive, strong = [], [], []
        for permutation in itertools.permutations(rows[6:]):
            top = rows[:6] + list(permutation[:4])
            dcg.append(sum((2**g-1)/math.log2(i+2) for i, (_, _, g) in enumerate(top))/ideal)
            positive.append(sum(g > 0 for _, _, g in top))
            strong.append(sum(g == 2 for _, _, g in top))
        self.assertAlmostEqual(m['tie_expected_ndcg_at_10'], sum(dcg)/len(dcg))
        self.assertAlmostEqual(m['tie_expected_precision_at_10'], sum(positive)/len(positive)/10)
        self.assertAlmostEqual(m['tie_expected_recall_at_10'], sum(positive)/len(positive)/8)
        self.assertAlmostEqual(m['tie_expected_strong_discovery_at_10'], sum(strong)/len(strong)/10)

    def test_input_order_does_not_break_ties(self):
        rows = [(s, 1.0, g) for s, _, g in self.rows()]
        expected = metrics_for_ranking(rows)
        random.Random(123).shuffle(rows)
        self.assertEqual(metrics_for_ranking(rows), expected)

    def test_duplicate_rejected(self):
        rows = self.rows(); rows[-1] = rows[0]
        with self.assertRaises(ValueError): metrics_for_ranking(rows)

    def test_invalid_scores_rejected(self):
        for invalid in [float('nan'), float('inf'), float('-inf')]:
            rows = self.rows(); rows[-1] = ('s11', invalid, 0)
            with self.assertRaises(ValueError): metrics_for_ranking(rows)

    def test_invalid_grades_rejected(self):
        for invalid in [None, -1, 3, .5]:
            rows = self.rows(); rows[-1] = ('s11', 0.0, invalid)
            with self.assertRaises(ValueError): metrics_for_ranking(rows)

    def test_empty_or_incomplete_pool_rejected(self):
        for rows in [[], self.rows()[:9], [(s, score, 0) for s, score, _ in self.rows()]]:
            with self.assertRaises(ValueError): metrics_for_ranking(rows)

    def test_wrong_k_rejected(self):
        with self.assertRaises(ValueError): metrics_for_ranking(self.rows(), k=5)


if __name__ == '__main__':
    unittest.main()
