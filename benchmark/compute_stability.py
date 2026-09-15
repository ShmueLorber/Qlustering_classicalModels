"""
benchmark.compute_stability -- Step 7: stability, consuming the RunRecords
Step 6 already produced and stored.

Loads results/raw_labels.npz (every run's full label array, keyed
"{dataset}__{algorithm}__seed{seed_or_none}" by runner.save_results()) and
groups the per-seed label arrays back by (dataset, algorithm) to feed
metrics.stability().

Stochastic algorithms (spectral, gmm) have 10 independent label arrays per
dataset -- stability is computed directly from those, per
metrics.stability()'s Hungarian-matched pairwise-agreement definition
(paper Methods Sec. IV C.e).

Deterministic algorithms (dbscan, agglomerative) only have ONE stored run
each (runner.run_all() does not re-run them per seed -- see its docstring),
so metrics.stability() cannot be evaluated (it requires >=2 runs). Per
benchmark.algorithms' and metrics.stability()'s own documented guidance,
this is reported as 1.0 with an explicit "deterministic" flag rather than
left blank or silently computed some other way: running a deterministic
algorithm again on the same data/params is bit-identical by construction,
so its true stability is trivially 1.0, but this is NOT resampled evidence
of robustness the way the stochastic algorithms' scores are.

Writes results/stability.csv: one row per (dataset, algorithm).
"""

import csv
from pathlib import Path

import numpy as np

from benchmark.algorithms import DETERMINISTIC_ALGORITHMS, STOCHASTIC_ALGORITHMS
from benchmark.dataset_instances import DATASET_INSTANCES
from benchmark.metrics import stability


def compute_all_stability(raw_labels_path: str = 'results/raw_labels.npz') -> list:
    d = np.load(raw_labels_path)

    # key format: "{dataset}__{algorithm}__seed{seed}" (stochastic)
    #          or "{dataset}__{algorithm}__seedNone"    (deterministic)
    groups = {}
    for key in d.files:
        dataset, algorithm, _seed_part = key.split('__')
        groups.setdefault((dataset, algorithm), []).append(d[key])

    rows = []
    for ds_name in DATASET_INSTANCES:
        for algo in ('spectral', 'gmm', 'dbscan', 'agglomerative'):
            label_arrays = groups.get((ds_name, algo))
            if label_arrays is None:
                continue

            if algo in STOCHASTIC_ALGORITHMS:
                result = stability(label_arrays)
                rows.append({
                    'dataset': ds_name, 'algorithm': algo,
                    'n_runs': len(label_arrays),
                    'stability_value': result.value,
                    'stability_status': result.status,
                    'stability_notes': result.notes,
                })
            elif algo in DETERMINISTIC_ALGORITHMS:
                assert len(label_arrays) == 1, (
                    f'{ds_name}/{algo}: expected exactly 1 stored run for a '
                    f'deterministic algorithm, got {len(label_arrays)}'
                )
                rows.append({
                    'dataset': ds_name, 'algorithm': algo,
                    'n_runs': 1,
                    'stability_value': 1.0,
                    'stability_status': 'deterministic',
                    'stability_notes': (
                        'deterministic algorithm: only 1 run was stored '
                        '(repeated runs on the same data/params are '
                        'bit-identical by construction, not resampled '
                        'stochastic evidence)'
                    ),
                })
    return rows


def save_stability(rows: list, out_path: str = 'results/stability.csv') -> None:
    fieldnames = ['dataset', 'algorithm', 'n_runs', 'stability_value', 'stability_status', 'stability_notes']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    rows = compute_all_stability()
    save_stability(rows)
    print(f'{len(rows)} (dataset, algorithm) stability rows written to results/stability.csv')
    for r in rows:
        v = f"{r['stability_value']:.4f}" if r['stability_value'] is not None else 'n/a'
        print(f"  {r['dataset']:16s} {r['algorithm']:14s} stability={v}  ({r['stability_status']})")
