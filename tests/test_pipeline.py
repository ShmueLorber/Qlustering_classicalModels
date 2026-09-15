"""
End-to-end pipeline tests: dataset -> algorithm -> metrics, across all four
benchmark datasets and all four classical baselines. These don't re-check
correctness (that's test_datasets/test_algorithms/test_metrics' job) -- they
check that the pieces actually compose without error and produce
self-consistent, in-range results, including the specific edge cases the
plan called out (no ground truth for QM9, DBSCAN noise-heavy results).
"""

import unittest

import numpy as np

from benchmark.algorithms import run_agglomerative, run_dbscan, run_gmm, run_spectral
from benchmark.datasets import (
    generate_ipr_dataset, generate_iris_dataset, generate_overlap3d_dataset,
    generate_qm9_dataset,
)
from benchmark.metrics import evaluate_clustering, stability

CENTERS_4 = np.array([
    [0.99, 0.11, 0.11], [0.11, 0.99, 0.11], [0.11, 0.11, 0.99], [1, 1, 1],
])


def _datasets():
    overlap_inputs, _, overlap_labels = generate_overlap3d_dataset(CENTERS_4, 60, width=0.15, seed=0)
    ipr_inputs, _, ipr_labels = generate_ipr_dataset(k=10, boundaries=(9, 2), num_states=50, seed=0)
    iris_inputs, _, iris_labels = generate_iris_dataset()
    qm9_inputs, _, qm9_labels = generate_qm9_dataset()
    return {
        'overlap3d': (np.array(overlap_inputs), overlap_labels, 4),
        'ipr':       (np.array(ipr_inputs), ipr_labels, 2),
        'iris':      (np.array(iris_inputs), iris_labels, 3),
        'qm9':       (np.array(qm9_inputs), qm9_labels, 2),  # q=2, per Table I
    }


def _assert_valid_metric_bundle(test_case, metrics, has_ground_truth):
    for name, result in metrics.items():
        with test_case.subTest(metric=name):
            test_case.assertIn(result.status, ('ok', 'undefined'))
            if result.status == 'ok':
                if name in ('rand_index',):
                    test_case.assertGreaterEqual(result.value, 0.0)
                    test_case.assertLessEqual(result.value, 1.0)
                elif name == 'adjusted_rand_index':
                    test_case.assertGreaterEqual(result.value, -1.0)
                    test_case.assertLessEqual(result.value, 1.0)
                elif name == 'compactness':
                    test_case.assertGreaterEqual(result.value, 0.0)
                elif name == 'silhouette':
                    test_case.assertGreaterEqual(result.value, -1.0)
                    test_case.assertLessEqual(result.value, 1.0)
            if name in ('rand_index', 'adjusted_rand_index') and not has_ground_truth:
                test_case.assertEqual(result.status, 'undefined')


class TestFullPipelineAllDatasetsAllAlgorithms(unittest.TestCase):
    def test_every_dataset_every_algorithm_runs_cleanly(self):
        datasets = _datasets()
        for ds_name, (X, labels, q) in datasets.items():
            has_gt = labels is not None
            algo_runs = {
                'spectral':      run_spectral(X, n_clusters=q, seed=0),
                'gmm':           run_gmm(X, n_clusters=q, seed=0),
                'agglomerative': run_agglomerative(X, n_clusters=q),
                'dbscan':        run_dbscan(X, eps=np.median(
                    np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)[
                        np.triu_indices(len(X), k=1)]) * 0.5, min_samples=3),
            }
            for algo_name, res in algo_runs.items():
                with self.subTest(dataset=ds_name, algorithm=algo_name):
                    metrics = evaluate_clustering(X, res.labels, y_true=labels)
                    _assert_valid_metric_bundle(self, metrics, has_ground_truth=has_gt)


class TestQM9HasNoGroundTruth(unittest.TestCase):
    def test_ri_ari_undefined_but_internal_metrics_ok(self):
        X, labels, q = _datasets()['qm9']
        self.assertIsNone(labels)
        res = run_gmm(X, n_clusters=q, seed=0)
        metrics = evaluate_clustering(X, res.labels, y_true=labels)
        self.assertEqual(metrics['rand_index'].status, 'undefined')
        self.assertEqual(metrics['adjusted_rand_index'].status, 'undefined')
        self.assertEqual(metrics['compactness'].status, 'ok')


class TestDBSCANNoiseHeavyPipeline(unittest.TestCase):
    def test_noise_heavy_result_still_evaluates(self):
        X, labels, q = _datasets()['overlap3d']
        res = run_dbscan(X, eps=0.02, min_samples=3)  # tiny eps -> mostly noise
        self.assertTrue((res.labels == -1).any())
        metrics = evaluate_clustering(X, res.labels, y_true=labels)
        noise_dropping_metrics = {'compactness', 'dunn_index', 'silhouette'}
        for name, result in metrics.items():
            with self.subTest(metric=name):
                self.assertIn(result.status, ('ok', 'undefined'))
                if result.status == 'ok' and (res.labels == -1).any() and name in noise_dropping_metrics:
                    self.assertIn('noise point', result.notes)


class TestStabilityAcrossRepeatedRuns(unittest.TestCase):
    def test_stochastic_algorithm_repeated_runs_on_real_data(self):
        X, labels, q = _datasets()['iris']
        runs = [run_gmm(X, n_clusters=q, seed=s).labels for s in range(10)]
        result = stability(runs)
        self.assertEqual(result.status, 'ok')
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)

    def test_deterministic_algorithm_repeated_runs_are_trivially_stable(self):
        X, labels, q = _datasets()['iris']
        runs = [run_agglomerative(X, n_clusters=q).labels for _ in range(10)]
        result = stability(runs)
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.value, 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
