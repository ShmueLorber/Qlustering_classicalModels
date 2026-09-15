"""
benchmark.dataset_instances — the concrete (dataset, configuration) pairs
used for the classical-baseline comparison, matching the paper's Table I /
Fig. 7 columns exactly.

DatasetInstance bundles X, y_true, and q together with a human-readable name
as ONE frozen object, specifically to close off a class of silent
config/mapping bugs (e.g. a QM9 run accidentally evaluated under the q=4
config but logged/plotted as "QM9 q=2") -- see tests/TEST_PLAN.md, Step 6.
Nothing downstream should ever reconstruct (X, y_true, q) separately from
raw generator calls; it should look them up here by name.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from benchmark.datasets import (
    generate_ipr_dataset, generate_iris_dataset, generate_overlap3d_dataset,
    generate_qm9_dataset,
)

DATASET_GENERATION_SEED = 0  # fixed once, shared by every algorithm/run -- NOT one of the per-run algorithm seeds


@dataclass(frozen=True)
class DatasetInstance:
    name:   str                    # human-readable, unique -- e.g. "overlap3d_w0.3"
    X:      np.ndarray
    y_true: Optional[np.ndarray]   # None for qm9 (no ground-truth partition)
    q:      int                    # requested/true number of clusters
    config: dict                   # the exact generator kwargs used, for provenance


# Fig. 3 / Table I's 5-cluster position config (paper Sec. II B): centers as
# given in the paper text, un-normalized -- generate_overlap3d_dataset
# normalizes each internally.
_CENTERS_5 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [-1.5, 1.5, 1.5],
    [0, 1, 1.5],
])


def _build_registry() -> dict:
    registry = {}

    for width in (0.1, 0.3):
        inputs, _, labels = generate_overlap3d_dataset(
            _CENTERS_5, num_states=60, width=width, seed=DATASET_GENERATION_SEED)
        name = f'overlap3d_w{width}'
        registry[name] = DatasetInstance(
            name=name, X=np.array(inputs), y_true=labels, q=5,
            config={'centers': _CENTERS_5.tolist(), 'num_states': 60, 'width': width,
                    'seed': DATASET_GENERATION_SEED})

    # (b_hi, b_lo) pairs pinned down against the paper's Fig. 4 / Table I
    # anchor points (see conversation): boundaries=(9,2) -> delta_IPR~=7,
    # boundaries=(6,5) -> delta_IPR~=1.
    for gap_label, boundaries in (('7', (9, 2)), ('1', (6, 5))):
        inputs, _, labels = generate_ipr_dataset(
            k=10, boundaries=boundaries, num_states=50, seed=DATASET_GENERATION_SEED)
        name = f'ipr_gap{gap_label}'
        registry[name] = DatasetInstance(
            name=name, X=np.array(inputs), y_true=labels, q=2,
            config={'k': 10, 'boundaries': boundaries, 'num_states': 50,
                    'seed': DATASET_GENERATION_SEED})

    for variant_label, feature_indices in (('full', None), ('reduced', (0, 2, 3))):
        inputs, _, labels = generate_iris_dataset(feature_indices=feature_indices)
        name = f'iris_{variant_label}'
        registry[name] = DatasetInstance(
            name=name, X=np.array(inputs), y_true=labels, q=3,
            config={'feature_indices': feature_indices})

    qm9_inputs, _, qm9_labels = generate_qm9_dataset()
    for q in (2, 4):
        name = f'qm9_q{q}'
        registry[name] = DatasetInstance(
            name=name, X=np.array(qm9_inputs), y_true=qm9_labels, q=q, config={})

    return registry


DATASET_INSTANCES = _build_registry()
