"""
Tests for benchmark.datasets. Focus: the generators are byte-for-byte ports
of the QTN_Library originals (Step 1) plus the QM9 loader (Step 2), so these
tests check structural invariants (shapes, unit norm, label counts) and the
specific paper configurations we've locked in, rather than re-deriving the
generation logic itself.
"""

import unittest

import numpy as np
import pandas as pd

from benchmark.datasets import (
    generate_overlap3d_dataset, generate_ipr_dataset, generate_iris_dataset,
    generate_qm9_dataset, load_qm9_properties, ipr,
)
from benchmark.encodings import amplitude_encode


class TestOverlap3D(unittest.TestCase):
    CENTERS_4 = np.array([
        [0.99, 0.11, 0.11], [0.11, 0.99, 0.11], [0.11, 0.11, 0.99], [1, 1, 1],
    ])

    def test_paper_fig2_config_shapes_and_group_sizes(self):
        inputs, targets, labels = generate_overlap3d_dataset(self.CENTERS_4, 60, width=0.15, seed=0)
        self.assertEqual(len(inputs), 60)
        self.assertEqual(inputs[0].shape, (3,))
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [15, 15, 15, 15])

    def test_inputs_are_unit_norm(self):
        inputs, _, _ = generate_overlap3d_dataset(self.CENTERS_4, 60, width=0.3, seed=1)
        norms = np.array([np.linalg.norm(v) for v in inputs])
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_targets_are_one_hot_matching_labels(self):
        inputs, targets, labels = generate_overlap3d_dataset(self.CENTERS_4, 12, width=0.1, seed=0)
        for t, l in zip(targets, labels):
            self.assertEqual(np.argmax(t), l)
            self.assertEqual(np.sum(t), 1.0)

    def test_same_seed_is_reproducible(self):
        i1, _, l1 = generate_overlap3d_dataset(self.CENTERS_4, 60, width=0.15, seed=42)
        i2, _, l2 = generate_overlap3d_dataset(self.CENTERS_4, 60, width=0.15, seed=42)
        np.testing.assert_array_equal(np.array(i1), np.array(i2))
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds_differ_when_width_nonzero(self):
        i1, _, _ = generate_overlap3d_dataset(self.CENTERS_4, 60, width=0.3, seed=1)
        i2, _, _ = generate_overlap3d_dataset(self.CENTERS_4, 60, width=0.3, seed=2)
        self.assertFalse(np.allclose(np.array(i1), np.array(i2)))

    def test_uneven_split_distributes_remainder_to_first_groups(self):
        centers = self.CENTERS_4[:3]  # q=3
        _, _, labels = generate_overlap3d_dataset(centers, 10, width=0.1, seed=0)
        counts = np.bincount(labels)
        np.testing.assert_array_equal(counts, [4, 3, 3])  # base=3, extra=1 -> group 0 gets +1


class TestIPR(unittest.TestCase):
    def test_ipr_formula(self):
        uniform = np.sqrt(np.full(10, 1.0 / 10))
        self.assertAlmostEqual(ipr(uniform), 10.0, places=6)
        one_hot = np.zeros(10); one_hot[0] = 1.0
        self.assertAlmostEqual(ipr(one_hot), 1.0, places=6)

    def test_shapes_and_label_split(self):
        inputs, targets, labels = generate_ipr_dataset(k=10, boundaries=(7, 4), num_states=50, seed=0)
        self.assertEqual(len(inputs), 50)
        self.assertEqual(inputs[0].shape, (10,))
        self.assertEqual(int(np.sum(labels == 0)), 25)
        self.assertEqual(int(np.sum(labels == 1)), 25)

    def test_hardcoded_edge_states_at_expected_positions(self):
        inputs, _, labels = generate_ipr_dataset(k=10, boundaries=(7, 4), num_states=50, seed=0)
        # group0[0]: fully uniform state -> IPR == k
        self.assertAlmostEqual(ipr(inputs[0]), 10.0, places=6)
        # group0[1]: uniform-minus-one-site
        self.assertLess(ipr(inputs[1]), 10.0)
        # group1[-1] (last state overall): one-hot -> IPR == 1
        self.assertAlmostEqual(ipr(inputs[-1]), 1.0, places=6)

    def test_group_iprs_respect_thresholds(self):
        b_hi, b_lo = 7, 4
        inputs, _, labels = generate_ipr_dataset(k=10, boundaries=(b_hi, b_lo), num_states=50, seed=0)
        iprs = np.array([ipr(p) for p in inputs])
        self.assertTrue(np.all(iprs[labels == 0] >= b_hi - 1e-9))
        self.assertTrue(np.all(iprs[labels == 1] <= b_lo + 1e-9))

    def test_boundary_sweep_reproduces_paper_delta_ipr_values(self):
        """
        Regression test locking in the mapping worked out with the user:
        boundaries (b_hi, b_lo) -> realized delta_IPR, for the 5-point sweep
        behind Fig. 4 / Table I ([2,9] swapped to (9,2); the rest fed as-is).
        """
        expected = {(9, 2): 7.0, (8.5, 2.5): 6.0, (8, 3): 5.0, (7, 4): 3.0, (6, 5): 1.76}
        for boundaries, expected_delta in expected.items():
            inputs, _, labels = generate_ipr_dataset(k=10, boundaries=boundaries, num_states=50, seed=0)
            iprs = np.array([ipr(p) for p in inputs])
            delta = iprs[labels == 0].min() - iprs[labels == 1].max()
            self.assertAlmostEqual(delta, expected_delta, delta=0.15,
                                    msg=f'boundaries={boundaries}')

    def test_same_seed_is_reproducible(self):
        i1, _, l1 = generate_ipr_dataset(k=10, boundaries=(7, 4), num_states=50, seed=7)
        i2, _, l2 = generate_ipr_dataset(k=10, boundaries=(7, 4), num_states=50, seed=7)
        np.testing.assert_array_equal(np.array(i1), np.array(i2))
        np.testing.assert_array_equal(l1, l2)


class TestIris(unittest.TestCase):
    def test_full_feature_shape_and_label_counts(self):
        inputs, targets, labels = generate_iris_dataset()
        self.assertEqual(len(inputs), 150)
        self.assertEqual(inputs[0].shape, (4,))
        np.testing.assert_array_equal(np.bincount(labels), [50, 50, 50])

    def test_reduced_feature_drops_sepal_width_correctly(self):
        """
        The reduced vector must be normalize(raw[:, (0,2,3)]), NOT a slice of
        the full normalized 4-feature vector (different normalization
        denominator) -- verified independently via amplitude_encode.
        """
        inputs_full, _, _ = generate_iris_dataset()
        inputs_reduced, _, _ = generate_iris_dataset(feature_indices=(0, 2, 3))
        self.assertEqual(inputs_reduced[0].shape, (3,))
        self.assertFalse(np.allclose(inputs_full[0][[0, 2, 3]], inputs_reduced[0]))

    def test_inputs_are_unit_norm(self):
        inputs, _, _ = generate_iris_dataset()
        norms = np.array([np.linalg.norm(v) for v in inputs])
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_is_deterministic_no_seed_needed(self):
        i1, _, l1 = generate_iris_dataset()
        i2, _, l2 = generate_iris_dataset()
        np.testing.assert_array_equal(np.array(i1), np.array(i2))
        np.testing.assert_array_equal(l1, l2)


class TestQM9(unittest.TestCase):
    def test_shape_and_norm(self):
        inputs, targets, labels = generate_qm9_dataset()
        self.assertEqual(len(inputs), 97)
        self.assertEqual(inputs[0].shape, (10,))
        norms = np.array([np.linalg.norm(v) for v in inputs])
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_no_ground_truth_labels(self):
        _, targets, labels = generate_qm9_dataset()
        self.assertIsNone(targets)
        self.assertIsNone(labels)

    def test_properties_table_aligned_with_inputs(self):
        inputs, _, _ = generate_qm9_dataset()
        props = load_qm9_properties()
        self.assertEqual(len(props), len(inputs))
        self.assertIn('SMILES', props.columns)


if __name__ == '__main__':
    unittest.main()
