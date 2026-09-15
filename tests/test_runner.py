"""
Tests for benchmark.dataset_instances and benchmark.runner (Step 6).
Directly implements the risk-area checklist from tests/TEST_PLAN.md's
"Step 6 (runner)" section: config/mapping safety, no label leakage,
seed integrity, no silent overwrites, and correct deterministic-vs-
stochastic run counts.
"""

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.algorithms import (
    run_agglomerative, run_dbscan, run_gmm, run_spectral,
)
from benchmark.dataset_instances import DATASET_INSTANCES, DatasetInstance
from benchmark.runner import _run_one, run_all, save_results

EXPECTED_INSTANCES = {
    'overlap3d_w0.1': 5, 'overlap3d_w0.3': 5,
    'ipr_gap7': 2, 'ipr_gap1': 2,
    'iris_full': 3, 'iris_reduced': 3,
    'qm9_q2': 2, 'qm9_q4': 4,
}


class TestDatasetInstanceRegistry(unittest.TestCase):
    def test_all_planned_instances_present_with_correct_q(self):
        self.assertEqual(set(DATASET_INSTANCES.keys()), set(EXPECTED_INSTANCES.keys()))
        for name, expected_q in EXPECTED_INSTANCES.items():
            with self.subTest(dataset=name):
                inst = DATASET_INSTANCES[name]
                self.assertIsInstance(inst, DatasetInstance)
                self.assertEqual(inst.q, expected_q)
                self.assertEqual(inst.name, name)
                self.assertEqual(len(inst.X), len(inst.y_true) if inst.y_true is not None else len(inst.X))

    def test_qm9_instances_share_same_data_different_q(self):
        q2, q4 = DATASET_INSTANCES['qm9_q2'], DATASET_INSTANCES['qm9_q4']
        np.testing.assert_array_equal(q2.X, q4.X)
        self.assertEqual(q2.q, 2)
        self.assertEqual(q4.q, 4)
        self.assertIsNone(q2.y_true)
        self.assertIsNone(q4.y_true)

    def test_overlap3d_widths_produce_different_data(self):
        w1, w3 = DATASET_INSTANCES['overlap3d_w0.1'], DATASET_INSTANCES['overlap3d_w0.3']
        self.assertFalse(np.allclose(w1.X, w3.X))
        self.assertEqual(w1.config['width'], 0.1)
        self.assertEqual(w3.config['width'], 0.3)

    def test_ipr_gap_configs_have_correct_boundaries(self):
        self.assertEqual(DATASET_INSTANCES['ipr_gap7'].config['boundaries'], (9, 2))
        self.assertEqual(DATASET_INSTANCES['ipr_gap1'].config['boundaries'], (6, 5))

    def test_iris_reduced_drops_sepal_width(self):
        self.assertIsNone(DATASET_INSTANCES['iris_full'].config['feature_indices'])
        self.assertEqual(DATASET_INSTANCES['iris_reduced'].config['feature_indices'], (0, 2, 3))
        self.assertEqual(DATASET_INSTANCES['iris_full'].X.shape[1], 4)
        self.assertEqual(DATASET_INSTANCES['iris_reduced'].X.shape[1], 3)


class TestNoLabelLeakage(unittest.TestCase):
    """Baseline algorithm wrappers must never see y_true during fitting --
    only evaluate_clustering (called after fit_predict) may consume it."""

    def test_wrapper_signatures_have_no_label_parameter(self):
        forbidden = {'y', 'y_true', 'labels', 'target', 'targets'}
        for fn in (run_spectral, run_gmm, run_dbscan, run_agglomerative):
            with self.subTest(fn=fn.__name__):
                params = set(inspect.signature(fn).parameters)
                self.assertEqual(params & forbidden, set(),
                                  f'{fn.__name__} exposes a label-like parameter: {params & forbidden}')

    def test_runner_never_passes_y_true_into_algorithm_call(self):
        dataset = DATASET_INSTANCES['iris_full']
        self.assertIsNotNone(dataset.y_true)  # sanity: this dataset HAS labels to leak

        for algo_name, spy_target in [
            ('spectral', 'benchmark.runner.ALGORITHMS'),
            ('gmm', 'benchmark.runner.ALGORITHMS'),
            ('agglomerative', 'benchmark.runner.ALGORITHMS'),
        ]:
            with self.subTest(algorithm=algo_name):
                from benchmark import algorithms as algo_module
                real_fn = algo_module.ALGORITHMS[algo_name]
                calls = []

                def spy(*args, **kwargs):
                    calls.append((args, kwargs))
                    return real_fn(*args, **kwargs)

                with patch.dict('benchmark.runner.ALGORITHMS', {algo_name: spy}):
                    _run_one(dataset, algo_name, seed=0)

                for args, kwargs in calls:
                    for value in list(args) + list(kwargs.values()):
                        if isinstance(value, np.ndarray) and value.shape == dataset.y_true.shape:
                            self.assertFalse(np.array_equal(value, dataset.y_true),
                                              f'{algo_name} received y_true as an argument')


class TestSeedIntegrity(unittest.TestCase):
    def test_stochastic_algorithm_uses_all_distinct_seeds(self):
        records = run_all(dataset_names=['iris_full'], algorithm_names=['gmm'], seeds=list(range(5)))
        seeds_used = sorted(r.seed for r in records)
        self.assertEqual(seeds_used, [0, 1, 2, 3, 4])

    def test_deterministic_algorithm_runs_exactly_once_regardless_of_seed_list(self):
        records = run_all(dataset_names=['iris_full'], algorithm_names=['agglomerative'], seeds=list(range(10)))
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].seed)
        self.assertTrue(records[0].is_deterministic)

    def test_stochastic_algorithm_run_count_matches_seed_list_length(self):
        records = run_all(dataset_names=['iris_full'], algorithm_names=['spectral'], seeds=list(range(3)))
        self.assertEqual(len(records), 3)
        self.assertTrue(all(not r.is_deterministic for r in records))


class TestNoOverwrite(unittest.TestCase):
    def test_run_all_produces_one_record_per_seed_no_duplicates(self):
        records = run_all(dataset_names=['iris_full'], algorithm_names=['gmm', 'agglomerative'],
                           seeds=list(range(4)))
        keys = [(r.dataset, r.algorithm, r.seed) for r in records]
        self.assertEqual(len(keys), len(set(keys)), 'duplicate (dataset, algorithm, seed) key found')
        # gmm: 4 stochastic runs + agglomerative: 1 deterministic run
        self.assertEqual(len(records), 5)

    def test_save_results_raises_on_duplicate_key(self):
        import tempfile
        records = run_all(dataset_names=['iris_full'], algorithm_names=['agglomerative'])
        duplicated = records + records  # force a collision
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                save_results(duplicated, out_dir=tmp)

    def test_save_results_writes_expected_row_count(self):
        import tempfile
        import csv
        records = run_all(dataset_names=['iris_full'], algorithm_names=['gmm'], seeds=list(range(3)))
        with tempfile.TemporaryDirectory() as tmp:
            save_results(records, out_dir=tmp)
            with open(f'{tmp}/runs.csv') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)


if __name__ == '__main__':
    unittest.main()
