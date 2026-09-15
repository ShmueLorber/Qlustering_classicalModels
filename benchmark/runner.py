"""
benchmark.runner — run every classical baseline on every dataset instance,
with controlled/recorded seeds, storing every individual run without
overwriting. This is Step 6 only: run + store. Aggregation/statistics is
Step 9; stability is Step 7 (both consume RunRecords produced here).

Stochastic algorithms (spectral, gmm) get one run per seed in SEEDS
(default range(10)) -- 10 genuinely different trials. Deterministic
algorithms (dbscan, agglomerative) are run exactly ONCE per dataset: running
them 10 times on the same data/params would produce 10 bit-identical rows,
which is not "10 independent trials" and would misleadingly suggest
stochastic evidence of stability if treated the same way downstream. Each
RunRecord's `is_deterministic` flag is the single source of truth Step 9
must branch on.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.algorithms import (
    ALGORITHMS, DETERMINISTIC_ALGORITHMS, STOCHASTIC_ALGORITHMS,
    estimate_dbscan_eps,
)
from benchmark.dataset_instances import DATASET_INSTANCES, DatasetInstance
from benchmark.metrics import evaluate_clustering

SEEDS = list(range(10))
DBSCAN_MIN_SAMPLES = 5


@dataclass(frozen=True)
class RunRecord:
    dataset:          str
    algorithm:        str
    seed:             Optional[int]   # None for deterministic algorithms' single run
    is_deterministic: bool
    labels:           np.ndarray
    runtime_seconds:  float
    params:           dict
    metrics:          dict           # name -> MetricResult
    warnings:         tuple = field(default_factory=tuple)


def _run_one(dataset: DatasetInstance, algorithm: str, seed: Optional[int]) -> RunRecord:
    warnings = []
    if algorithm == 'dbscan':
        eps = estimate_dbscan_eps(dataset.X, min_samples=DBSCAN_MIN_SAMPLES)
        result = ALGORITHMS['dbscan'](dataset.X, eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        n_noise = int(np.sum(result.labels == -1))
        if n_noise > 0:
            warnings.append(f'{n_noise}/{len(result.labels)} points labeled noise (-1)')
        n_found = len(np.unique(result.labels[result.labels != -1]))
        if n_found != dataset.q:
            warnings.append(f'found {n_found} non-noise clusters, requested q={dataset.q} not directly comparable (DBSCAN has no cluster-count parameter)')
    elif algorithm == 'agglomerative':
        result = ALGORITHMS['agglomerative'](dataset.X, n_clusters=dataset.q)
    elif algorithm == 'spectral':
        result = ALGORITHMS['spectral'](dataset.X, n_clusters=dataset.q, seed=seed)
    elif algorithm == 'gmm':
        result = ALGORITHMS['gmm'](dataset.X, n_clusters=dataset.q, seed=seed)
    else:
        raise ValueError(f'unknown algorithm {algorithm!r}')

    metrics = evaluate_clustering(dataset.X, result.labels, y_true=dataset.y_true)
    return RunRecord(
        dataset=dataset.name, algorithm=algorithm, seed=seed,
        is_deterministic=(algorithm in DETERMINISTIC_ALGORITHMS),
        labels=result.labels, runtime_seconds=result.runtime_seconds,
        params=result.params, metrics=metrics, warnings=tuple(warnings),
    )


def run_all(dataset_names=None, algorithm_names=None, seeds=SEEDS) -> list:
    """
    Runs every algorithm in `algorithm_names` (default: all 4) on every
    dataset in `dataset_names` (default: all 8 registered instances).
    Stochastic algorithms get one RunRecord per seed; deterministic
    algorithms get exactly one RunRecord (seed=None). Returns a flat,
    append-only list -- callers must never mutate/overwrite entries in place.
    """
    dataset_names = dataset_names or list(DATASET_INSTANCES.keys())
    algorithm_names = algorithm_names or list(ALGORITHMS.keys())

    records = []
    for ds_name in dataset_names:
        dataset = DATASET_INSTANCES[ds_name]
        for algo_name in algorithm_names:
            if algo_name in STOCHASTIC_ALGORITHMS:
                seeds_actually_used = set()
                for seed in seeds:
                    records.append(_run_one(dataset, algo_name, seed))
                    seeds_actually_used.add(seed)
                assert seeds_actually_used == set(seeds), (
                    f'{ds_name}/{algo_name}: expected seeds {set(seeds)}, '
                    f'actually used {seeds_actually_used}'
                )
            else:
                records.append(_run_one(dataset, algo_name, seed=None))
    return records


def _metric_result_to_dict(name, result):
    return {f'{name}_value': result.value, f'{name}_status': result.status, f'{name}_notes': result.notes}


def save_results(records: list, out_dir: str = 'results') -> None:
    """
    Writes two artifacts, neither of which overwrites individual run data:
      - runs.csv: one row per RunRecord, scalar metrics + provenance
        (dataset, algorithm, seed, params as JSON, warnings).
      - raw_labels.npz: every run's full label array, keyed by a unique
        "{dataset}__{algorithm}__seed{seed_or_none}" name -- needed by Step 7
        (stability) and Step 8 (comparison), which require the actual label
        arrays, not just the summary metrics.
    """
    import csv

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    label_arrays = {}
    rows = []
    seen_keys = set()
    for rec in records:
        key = f'{rec.dataset}__{rec.algorithm}__seed{rec.seed}'
        if key in seen_keys:
            raise RuntimeError(f'duplicate run key {key!r} -- would overwrite an existing result')
        seen_keys.add(key)
        label_arrays[key] = rec.labels

        row = {
            'dataset': rec.dataset, 'algorithm': rec.algorithm, 'seed': rec.seed,
            'is_deterministic': rec.is_deterministic, 'runtime_seconds': rec.runtime_seconds,
            'params': json.dumps(rec.params, default=str), 'warnings': '; '.join(rec.warnings),
        }
        for name, result in rec.metrics.items():
            row.update(_metric_result_to_dict(name, result))
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path / 'runs.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(out_path / 'raw_labels.npz', **label_arrays)


if __name__ == '__main__':
    all_records = run_all()
    save_results(all_records)
    print(f'{len(all_records)} runs completed and saved to results/')
