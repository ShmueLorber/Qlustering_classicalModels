"""
Tests for benchmark.algorithms. These are thin sklearn wrappers (Step 3), so
the tests focus on: the common ClusteringResult contract, that each wrapper
recovers a trivially-separable ground truth, and the determinism split
(spectral/gmm stochastic vs. dbscan/agglomerative deterministic) that Steps
6/7's repeated-run protocol depends on.
"""

import unittest

import numpy as np

from benchmark.algorithms import (
    ClusteringResult, DETERMINISTIC_ALGORITHMS, STOCHASTIC_ALGORITHMS,
    run_agglomerative, run_dbscan, run_gmm, run_spectral,
)
from benchmark.datasets import generate_overlap3d_dataset
from benchmark.metrics import rand_index

CENTERS_4 = np.array([
    [0.99, 0.11, 0.11], [0.11, 0.99, 0.11], [0.11, 0.11, 0.99], [1, 1, 1],
])


def _well_separated_data():
    inputs, _, labels = generate_overlap3d_dataset(CENTERS_4, 60, width=0.05, seed=0)
    return np.array(inputs), labels


class TestClusteringResultContract(unittest.TestCase):
    def test_result_fields_and_determinism_flag(self):
        X, labels = _well_separated_data()
        cases = [
            run_spectral(X, n_clusters=4, seed=0),
            run_gmm(X, n_clusters=4, seed=0),
            run_dbscan(X, eps=0.3, min_samples=3),
            run_agglomerative(X, n_clusters=4),
        ]
        for res in cases:
            with self.subTest(algorithm=res.algorithm):
                self.assertIsInstance(res, ClusteringResult)
                self.assertEqual(len(res.labels), len(X))
                self.assertGreaterEqual(res.runtime_seconds, 0.0)
                expected_deterministic = res.algorithm in DETERMINISTIC_ALGORITHMS
                self.assertEqual(res.is_deterministic, expected_deterministic)

    def test_algorithm_name_sets_are_disjoint_and_complete(self):
        self.assertEqual(STOCHASTIC_ALGORITHMS | DETERMINISTIC_ALGORITHMS,
                          {'spectral', 'gmm', 'dbscan', 'agglomerative'})
        self.assertEqual(STOCHASTIC_ALGORITHMS & DETERMINISTIC_ALGORITHMS, set())


class TestRecoversWellSeparatedClusters(unittest.TestCase):
    """On trivially-separable synthetic data, every method should recover
    the ground-truth partition (RI ~= 1), sanity-checking each wrapper end
    to end against a known-good answer rather than just "it didn't crash"."""

    def test_spectral(self):
        X, labels = _well_separated_data()
        res = run_spectral(X, n_clusters=4, seed=0)
        self.assertGreaterEqual(rand_index(labels, res.labels).value, 0.95)

    def test_gmm(self):
        X, labels = _well_separated_data()
        res = run_gmm(X, n_clusters=4, seed=0)
        self.assertGreaterEqual(rand_index(labels, res.labels).value, 0.95)

    def test_agglomerative(self):
        X, labels = _well_separated_data()
        res = run_agglomerative(X, n_clusters=4)
        self.assertGreaterEqual(rand_index(labels, res.labels).value, 0.95)

    def test_dbscan_with_reasonable_eps(self):
        X, labels = _well_separated_data()
        res = run_dbscan(X, eps=0.3, min_samples=3)
        non_noise = res.labels != -1
        self.assertTrue(non_noise.any())
        self.assertGreaterEqual(rand_index(labels[non_noise], res.labels[non_noise]).value, 0.9)


class TestDeterminism(unittest.TestCase):
    def test_dbscan_is_bitwise_reproducible(self):
        X, _ = _well_separated_data()
        r1 = run_dbscan(X, eps=0.3, min_samples=3)
        r2 = run_dbscan(X, eps=0.3, min_samples=3)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_agglomerative_is_bitwise_reproducible(self):
        X, _ = _well_separated_data()
        r1 = run_agglomerative(X, n_clusters=4)
        r2 = run_agglomerative(X, n_clusters=4)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_spectral_same_seed_is_reproducible(self):
        X, _ = _well_separated_data()
        r1 = run_spectral(X, n_clusters=4, seed=123)
        r2 = run_spectral(X, n_clusters=4, seed=123)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_gmm_same_seed_is_reproducible(self):
        X, _ = _well_separated_data()
        r1 = run_gmm(X, n_clusters=4, seed=123)
        r2 = run_gmm(X, n_clusters=4, seed=123)
        np.testing.assert_array_equal(r1.labels, r2.labels)


class TestDBSCANHasNoClusterCountParameter(unittest.TestCase):
    def test_run_dbscan_signature_has_no_n_clusters(self):
        import inspect
        params = inspect.signature(run_dbscan).parameters
        self.assertNotIn('n_clusters', params)
        self.assertIn('eps', params)
        self.assertIn('min_samples', params)


if __name__ == '__main__':
    unittest.main()
