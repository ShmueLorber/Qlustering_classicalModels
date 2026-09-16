"""
benchmark.qlustering_scores -- runs the real Qlustering model (ClusteringModel
+ adam_train, from QTN_Library) on this repo's DatasetInstances, scored with
benchmark.metrics -- never training.metrics (see benchmark/metrics.py's
docstring on why RI/ARI aren't re-implemented a second time).

Setup (one-time): `pip install -e <path-to-QTN_Library>` into whichever
interpreter runs this script, so `core`/`models`/`training` import cleanly
without sys.path hacks.

Deliberately mirrors benchmark.runner's RunRecord/save_results shape so
results slot into the same results/*.csv + raw_labels.npz pipeline used by
the classical baselines (and benchmark.compute_stability, once 'qlustering'
is added to benchmark.algorithms.STOCHASTIC_ALGORITHMS) -- written to a
separate results_qlustering/ directory so it never clobbers the classical
run already committed under results/.
"""

import time

import numpy as np

from core.hamiltonian import random_H_vec  # QTN_Library
from models.base import Topology  # QTN_Library
from models.clustering import ClusteringModel  # QTN_Library
from training.optimizers import adam_train  # QTN_Library

from benchmark.dataset_instances import DATASET_INSTANCES
from benchmark.metrics import evaluate_clustering
from benchmark.runner import RunRecord, save_results

DEFAULT_N_STEPS = 300
DEFAULT_LATENT = 4
DEFAULT_LR = 0.05


def run_one(dataset, seed, n_steps=DEFAULT_N_STEPS, latent=DEFAULT_LATENT, lr=DEFAULT_LR):
    n_features = dataset.X.shape[1]
    topo = Topology(input_sites=n_features, latent=latent, output_sites=dataset.q)
    # gamma_dep pinned at 0.0 -- do not change (user instruction).
    model = ClusteringModel(topo, gamma_in=np.ones(n_features), gamma_out=np.ones(dataset.q), gamma_dep=0.0)
    H_init = random_H_vec(topo.input_sites, topo.latent, topo.output_sites, scale=0.5, seed=seed)

    inputs = list(dataset.X)
    # cost() never consults targets (clustering is unsupervised); adam_train
    # only reads them for its optional score_fn reporting, which we skip here.
    targets = dataset.y_true if dataset.y_true is not None else [0] * len(inputs)

    t0 = time.time()
    H_trained, _, _ = adam_train(model, H_init, inputs, targets, n_steps=n_steps, lr=lr, print_every=0)
    runtime = time.time() - t0

    labels = np.array([model.predict_label(H_trained, x) for x in inputs])
    metrics = evaluate_clustering(dataset.X, labels, y_true=dataset.y_true)

    return RunRecord(
        dataset=dataset.name, algorithm='qlustering', seed=seed, is_deterministic=False,
        labels=labels, runtime_seconds=runtime,
        params={
            'n_steps': n_steps, 'lr': lr, 'latent': latent,
            'diversity_penalty': model.diversity_penalty, 'diversity_decay': model.diversity_decay,
            'gamma_dep': 0.0,
        },
        metrics=metrics, warnings=(),
    )


def run_all(dataset_names=None, seeds=range(10), **kwargs):
    dataset_names = dataset_names or list(DATASET_INSTANCES.keys())
    return [run_one(DATASET_INSTANCES[name], seed, **kwargs) for name in dataset_names for seed in seeds]


if __name__ == '__main__':
    # Smoke test: position task only (paper's 5-cluster omega-sweep), checked
    # against the manuscript-transcribed Qlustering RI/ARI recorded in the
    # published "Qlustering Benchmark Matrix":
    #   omega=0.1 -> RI=1.00, ARI=1.00 (text gives only a qualitative/range statement)
    #   omega=0.3 -> RI=0.85, ARI=0.63 (text labels this "the four-cluster case" mid the
    #                five-cluster sweep -- likely this point, flagged for verification)
    PAPER = {
        'overlap3d_w0.1': {'RI': 1.00, 'ARI': 1.00},
        'overlap3d_w0.3': {'RI': 0.85, 'ARI': 0.63},
    }

    records = run_all(dataset_names=list(PAPER.keys()), seeds=range(5))
    save_results(records, out_dir='results_qlustering')

    print(f'{len(records)} Qlustering runs saved to results_qlustering/\n')
    for name in PAPER:
        runs = [r for r in records if r.dataset == name]
        ri = [r.metrics['rand_index'].value for r in runs]
        ari = [r.metrics['adjusted_rand_index'].value for r in runs]
        print(f'{name}:')
        print(f"  RI  = {np.mean(ri):.4f} +/- {np.std(ri):.4f}  per-seed: {[round(v, 3) for v in ri]}  (paper: {PAPER[name]['RI']})")
        print(f"  ARI = {np.mean(ari):.4f} +/- {np.std(ari):.4f}  per-seed: {[round(v, 3) for v in ari]}  (paper: {PAPER[name]['ARI']})")
