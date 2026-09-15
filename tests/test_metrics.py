"""
Tests for benchmark.metrics. Where possible, correctness is checked against
an INDEPENDENT reference (a brute-force reimplementation from scratch, or
sklearn where sklearn's own definition is the trusted oracle) rather than
just re-deriving the same formula a second time -- so a bug shared between
the implementation and the test is less likely to slip through.

Alignment with the paper's actual reference MATLAB code (Internal_metrics.mlx)
was verified manually (see conversation) for compactness/Dunn/stability
(exact match) and silhouette (sklearn's default metric had to be overridden
to 'sqeuclidean' to match MATLAB's silhouette() default). That sqeuclidean
choice is re-asserted here as a regression test.
"""

import itertools
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from benchmark.metrics import (
    MetricResult, adjusted_rand_index, compactness, dunn_index, rand_index,
    silhouette, stability,
)


def _brute_force_rand_index(y_true, y_pred):
    """Independent reference: literally count agreeing/disagreeing pairs."""
    n = len(y_true)
    agree = 0
    total = 0
    for i, j in itertools.combinations(range(n), 2):
        total += 1
        same_true = y_true[i] == y_true[j]
        same_pred = y_pred[i] == y_pred[j]
        if same_true == same_pred:
            agree += 1
    return agree / total


def _brute_force_stability(list_of_label_arrays):
    """Independent reference: try every label permutation (only feasible
    for tiny K), taking the best match fraction per pair."""
    R = len(list_of_label_arrays)
    matches = []
    for a, b in itertools.combinations(range(R), 2):
        la, lb = np.asarray(list_of_label_arrays[a]), np.asarray(list_of_label_arrays[b])
        classes = sorted(set(lb.tolist()))
        best = 0.0
        for perm in itertools.permutations(classes):
            mapping = dict(zip(classes, perm))
            remapped = np.array([mapping[v] for v in lb])
            best = max(best, np.mean(la == remapped))
        matches.append(best)
    return float(np.mean(matches))


class TestRandIndex(unittest.TestCase):
    def test_perfect_agreement(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(rand_index(y, y).value, 1.0)
        self.assertEqual(adjusted_rand_index(y, y).value, 1.0)

    def test_matches_brute_force_reference(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=20)
        y_pred = rng.integers(0, 3, size=20)
        expected = _brute_force_rand_index(y_true, y_pred)
        self.assertAlmostEqual(rand_index(y_true, y_pred).value, expected, places=10)

    def test_ari_matches_sklearn_oracle(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 4, size=30)
        y_pred = rng.integers(0, 4, size=30)
        expected = adjusted_rand_score(y_true, y_pred)
        self.assertAlmostEqual(adjusted_rand_index(y_true, y_pred).value, expected, places=10)

    def test_permutation_invariance(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_a = np.array([0, 0, 1, 1])
        y_pred_b = np.array([5, 5, 9, 9])  # same partition, different label values
        self.assertEqual(rand_index(y_true, y_pred_a).value, rand_index(y_true, y_pred_b).value)

    def test_undefined_without_ground_truth(self):
        result = rand_index(None, np.array([0, 1, 0]))
        self.assertEqual(result.status, 'undefined')
        self.assertIsNone(result.value)
        result = adjusted_rand_index(None, np.array([0, 1, 0]))
        self.assertEqual(result.status, 'undefined')


class TestCompactness(unittest.TestCase):
    def test_hand_computed_two_clusters(self):
        # cluster 0: (0,0) and (2,0) -> centroid (1,0), sq dists = 1+1 = 2
        # cluster 1: (0,10) only -> centroid (0,10), sq dist = 0
        X = np.array([[0, 0], [2, 0], [0, 10]])
        labels = np.array([0, 0, 1])
        result = compactness(X, labels)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.value, 2.0, places=10)

    def test_single_cluster_is_defined(self):
        X = np.array([[0, 0], [1, 0], [0, 1]])
        labels = np.zeros(3, dtype=int)
        result = compactness(X, labels)
        self.assertEqual(result.status, 'ok')

    def test_dbscan_noise_excluded_and_noted(self):
        X = np.array([[0, 0], [0.1, 0], [10, 10]])  # last point is "noise"
        labels = np.array([0, 0, -1])
        result = compactness(X, labels)
        self.assertEqual(result.status, 'ok')
        self.assertIn('1 noise point', result.notes)

    def test_all_noise_is_undefined(self):
        X = np.array([[0, 0], [1, 1]])
        labels = np.array([-1, -1])
        result = compactness(X, labels)
        self.assertEqual(result.status, 'undefined')


class TestDunnIndex(unittest.TestCase):
    def test_hand_computed_two_well_separated_clusters(self):
        # cluster 0: (0,0),(1,0) diameter=1 ; cluster 1: (10,0),(11,0) diameter=1
        # min inter-cluster distance: (1,0)-(10,0) = 9
        X = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])
        labels = np.array([0, 0, 1, 1])
        result = dunn_index(X, labels)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.value, 9.0 / 1.0, places=10)

    def test_single_cluster_is_undefined(self):
        X = np.array([[0, 0], [1, 0], [2, 0]])
        labels = np.zeros(3, dtype=int)
        result = dunn_index(X, labels)
        self.assertEqual(result.status, 'undefined')

    def test_all_singleton_clusters_is_undefined_zero_division(self):
        X = np.array([[0, 0], [5, 0]])
        labels = np.array([0, 1])  # each cluster has 1 point -> diameter 0 for both
        result = dunn_index(X, labels)
        self.assertEqual(result.status, 'undefined')
        self.assertIn('zero intra-cluster diameter', result.notes)


class TestSilhouette(unittest.TestCase):
    def test_matches_sklearn_sqeuclidean_not_plain_euclidean(self):
        """
        Regression test for the MATLAB-alignment fix: our silhouette() must
        use sqeuclidean (matching Internal_metrics.mlx's silhouette()
        default), which is numerically different from sklearn's own
        plain-Euclidean default.
        """
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 3))
        labels = rng.integers(0, 3, size=30)
        expected_sq = silhouette_score(X, labels, metric='sqeuclidean')
        expected_plain = silhouette_score(X, labels, metric='euclidean')
        result = silhouette(X, labels)
        self.assertAlmostEqual(result.value, expected_sq, places=10)
        self.assertNotAlmostEqual(result.value, expected_plain, places=6)

    def test_single_cluster_is_undefined(self):
        X = np.array([[0, 0], [1, 0], [2, 0]])
        labels = np.zeros(3, dtype=int)
        result = silhouette(X, labels)
        self.assertEqual(result.status, 'undefined')

    def test_every_point_its_own_cluster_is_undefined(self):
        X = np.array([[0, 0], [1, 0], [2, 0]])
        labels = np.array([0, 1, 2])
        result = silhouette(X, labels)
        self.assertEqual(result.status, 'undefined')


class TestStability(unittest.TestCase):
    def test_identical_runs_give_stability_one(self):
        runs = [np.array([0, 0, 1, 1]) for _ in range(5)]
        result = stability(runs)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.value, 1.0, places=10)

    def test_matches_brute_force_reference(self):
        rng = np.random.default_rng(3)
        runs = [rng.integers(0, 3, size=12) for _ in range(4)]
        expected = _brute_force_stability(runs)
        result = stability(runs)
        self.assertAlmostEqual(result.value, expected, places=10)

    def test_permutation_invariant_relabeling_still_gives_one(self):
        base = np.array([0, 0, 1, 1, 2, 2])
        relabeled = np.array([2, 2, 0, 0, 1, 1])  # same partition, different label names
        result = stability([base, relabeled])
        self.assertAlmostEqual(result.value, 1.0, places=10)

    def test_needs_at_least_two_runs(self):
        result = stability([np.array([0, 1, 0])])
        self.assertEqual(result.status, 'undefined')

    def test_handles_dbscan_noise_label_without_crashing(self):
        runs = [np.array([0, 0, -1, 1]), np.array([1, 1, -1, 0])]  # same partition incl. noise, relabeled
        result = stability(runs)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.value, 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
