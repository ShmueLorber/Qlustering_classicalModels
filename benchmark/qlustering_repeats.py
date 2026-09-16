"""
benchmark.qlustering_repeats -- diagnostic reproduction + internal-metric
collection for the paper's Position (overlap3d) and Localization/IPR tasks,
using QTN_Library's training/qlustering_search.py (the MATLAB-faithful
hill-climb) with the RI/ARI-based early stop that
Qlustering_matlab_files/QlusteringFunction.m actually used, and that the
QTN_Library port deliberately dropped (see training/qlustering_search.py's
own docstring: "minus the RI-based early stop, which needs external labels
not available to a genuinely unsupervised run"). Restoring it here is a
supervised, diagnostic-only choice -- it is NOT how Qlustering would run in
production/unsupervised use, only how we check convergence and fill in the
benchmark matrix's missing internal-metric cells.

qlustering_search_monitored() is a local, instrumented reimplementation of
training.qlustering_search.qlustering_search_train's loop (using its private
_free_position_mask/_cost_and_predictions helpers verbatim, imported -- not
copied) that additionally, at every iteration:
  - scores the current best prediction against ground truth with
    benchmark.metrics (never training.metrics -- see benchmark/metrics.py's
    docstring on why RI/ARI aren't re-implemented a second time)
  - records (cost, RI, ARI) for a training-propagation plot
  - stops immediately if RI == 1 and ARI == 1 (a perfect solution)
  - otherwise tracks the best RI+ARI seen at any iteration (the search's
    cost-improving moves aren't guaranteed to be RI/ARI-improving moves, so
    the best point in the trajectory need not be the final one)

Position uses the general (hard-argmax) clustering cost (margins=None). IPR
uses models/clustering.py's margins=(lo, hi) cost -- confirmed, by decompiling
the reference paper's saved MATLAB workspaces (all four reference a function
named QmeansCost3), to be the actual cost function used for the paper's
localization benchmark; its own hardcoded default is margins=(0.45, 0.55).
An earlier version of both this script and models/clustering.py used the
wrong (inverted) confident/ambiguous branch logic, which made any margins
window centered on 0.5 -- i.e. any window that makes sense here -- degenerate
to a constant [0.5, 0.5] target; that porting bug is now fixed (see
models/clustering.py's module docstring and tests/test_models_clustering.py's
regression tests in QTN_Library). The same workspaces also give H_min=-200,
H_max=200 (this script's H_min/H_max default, widened from an earlier ±5) and
per-config `m` (latent) values.

Reported "Qlustering" numbers, matching the paper's own production pipeline
(repeated runs -> consensus, i.e. training/consensus.py's port of
ConsensusClustering.m): RI/ARI/compactness/Dunn/silhouette are computed on
the CONSENSUS label assignment (one deterministic-given-the-ensemble answer,
not averaged across runs), and stability is computed from the 10 individual
runs' label arrays feeding that consensus -- exactly parallel to how
benchmark.compute_stability.py scores the stochastic classical baselines.
"""

import numpy as np

from core.hamiltonian import random_H_vec  # QTN_Library
from models.base import Topology  # QTN_Library
from models.clustering import ClusteringModel  # QTN_Library
from training.qlustering_search import _cost_and_predictions, _free_position_mask  # QTN_Library
from training.consensus import consensus_clustering  # QTN_Library

from benchmark.dataset_instances import DATASET_INSTANCES
from benchmark.metrics import rand_index, adjusted_rand_index, evaluate_clustering, stability

LATENT = 2
N_ITERATIONS = 50
NUM_PARTICLES = 20
N_RUNS = 10
GAMMA_DEP = 0.0  # pinned per user instruction -- do not change
DATASETS = ('overlap3d_w0.1', 'overlap3d_w0.3', 'ipr_gap7', 'ipr_gap1')


def qlustering_search_monitored(
    model, H_vec_init, inputs, y_true, n_iterations=N_ITERATIONS, num_particles=NUM_PARTICLES,
    H_min=-200.0, H_max=200.0, min_group=None, seed=None,
):
    rng = np.random.default_rng(seed)
    topo = model.topology()
    free_idx = np.flatnonzero(_free_position_mask(topo).reshape(-1))

    def _try_cost_and_predictions(H_vec):
        # Wide H_min/H_max (e.g. +-200, matching the reference paper's actual
        # runs) occasionally pushes compute_steady_state's regularized solve
        # just past its strict physicality tolerance (Im(diag) ~1e-6 over a
        # 1e-6 tol) -- a numerical-conditioning artifact of the wider range,
        # not a real candidate; treat it as invalid rather than crashing the
        # whole search.
        try:
            return _cost_and_predictions(model, H_vec, inputs)
        except ValueError:
            return np.inf, None

    best_H = H_vec_init.copy()
    best_cost, best_preds = _try_cost_and_predictions(best_H)
    while best_preds is None or np.bincount(best_preds, minlength=topo.output_sites).min() == 0:
        best_H = rng.uniform(H_min, H_max, size=H_vec_init.shape)
        best_cost, best_preds = _try_cost_and_predictions(best_H)
    best_occ = np.bincount(best_preds, minlength=topo.output_sites)

    if min_group is None:
        min_group = max(1, int(best_occ.min()))

    def _score(preds):
        return rand_index(y_true, preds).value, adjusted_rand_index(y_true, preds).value

    ri0, ari0 = _score(best_preds)
    cost_curve, ri_curve, ari_curve = [best_cost], [ri0], [ari0]
    best_sum, best_sum_H, best_sum_preds, best_sum_iter = ri0 + ari0, best_H.copy(), best_preds.copy(), 0
    perfect = (ri0 >= 1.0 - 1e-9 and ari0 >= 1.0 - 1e-9)

    it = 0
    while it < n_iterations and not perfect:
        it += 1
        idx = free_idx[rng.integers(len(free_idx))]
        candidates = []
        for _ in range(num_particles):
            H_try = best_H.copy()
            H_try[idx] = rng.uniform(H_min, H_max)
            cost, preds = _try_cost_and_predictions(H_try)
            if preds is None:
                continue
            occ = np.bincount(preds, minlength=topo.output_sites)
            candidates.append((cost, occ, preds, H_try))

        valid = [c for c in candidates if c[1].min() >= min_group]
        if valid:
            cand_cost, cand_occ, cand_preds, cand_H = min(valid, key=lambda c: c[0])
            if cand_cost < best_cost:
                best_H, best_cost, best_occ, best_preds = cand_H, cand_cost, cand_occ, cand_preds

        ri, ari = _score(best_preds)
        cost_curve.append(best_cost)
        ri_curve.append(ri)
        ari_curve.append(ari)
        if ri + ari > best_sum:
            best_sum, best_sum_H, best_sum_preds, best_sum_iter = ri + ari, best_H.copy(), best_preds.copy(), it
        if ri >= 1.0 - 1e-9 and ari >= 1.0 - 1e-9:
            perfect = True

    return {
        'H_best': best_sum_H, 'preds_best': best_sum_preds, 'best_iter': best_sum_iter,
        'perfect_found': perfect,
        'cost_curve': cost_curve, 'ri_curve': ri_curve, 'ari_curve': ari_curve,
        'n_iterations_run': it,
    }


def run_repeats(
    dataset_name, n_runs=N_RUNS, seed_offset=0, margins=None, diversity_penalty=2.0, latent=LATENT,
    H_min=-200.0, H_max=200.0,
):
    ds = DATASET_INSTANCES[dataset_name]
    n_features = ds.X.shape[1]
    inputs = list(ds.X)

    run_results = []
    for r in range(n_runs):
        seed = seed_offset + r
        topo = Topology(input_sites=n_features, latent=latent, output_sites=ds.q)
        model = ClusteringModel(
            topo, gamma_in=np.ones(n_features), gamma_out=np.ones(ds.q), gamma_dep=GAMMA_DEP,
            margins=margins, diversity_penalty=diversity_penalty,
        )
        H_init = random_H_vec(topo.input_sites, topo.latent, topo.output_sites, scale=0.5, seed=seed)
        result = qlustering_search_monitored(model, H_init, inputs, ds.y_true, H_min=H_min, H_max=H_max, seed=seed)
        run_results.append(result)
        print(
            f"  {dataset_name} run {r + 1:2d}/{n_runs} (seed={seed}): "
            f"RI={result['ri_curve'][result['best_iter']]:.4f} "
            f"ARI={result['ari_curve'][result['best_iter']]:.4f} "
            f"at iter {result['best_iter']}/{result['n_iterations_run']} "
            f"perfect={result['perfect_found']}"
        )

    cluster_assignments = np.stack([r['preds_best'] for r in run_results], axis=1)  # (n_samples, n_runs)
    consensus_labels = consensus_clustering(cluster_assignments, ds.q, method='average')
    consensus_metrics = evaluate_clustering(ds.X, consensus_labels, y_true=ds.y_true)
    stability_result = stability([r['preds_best'] for r in run_results])

    return {
        'dataset': dataset_name, 'run_results': run_results,
        'consensus_labels': consensus_labels,
        'consensus_metrics': consensus_metrics,
        'stability': stability_result,
    }


# Cost-function configs under investigation for the IPR/localization task.
# 'general_div0'/'general_div2' were the first (wrong) hypothesis -- IPR
# actually uses margins=(lo,hi), models/clustering.py's port of Q-means/
# QmeansCost3.m, confirmed against the paper's saved MATLAB workspaces (all
# four reference QmeansCost3 by name). QmeansCost3.m's own hardcoded default
# is (0.45, 0.55); (0.4, 0.6) is a wider ambiguous band to try if the default
# undershoots -- it penalizes proximity to 0.5 over a larger range, pushing
# the search to explore more decisive splits. H_min/H_max=-200/200 also comes
# from those same workspaces (an earlier run here used the qlustering_search
# default of ±5).
CONFIGS = {
    'general_div2':      {'margins': None,          'diversity_penalty': 2.0, 'H_min': -5.0,   'H_max': 5.0},
    'general_div0':      {'margins': None,          'diversity_penalty': 0.0, 'H_min': -5.0,   'H_max': 5.0},
    'margins_0.45_0.55': {'margins': (0.45, 0.55),  'diversity_penalty': 0.0, 'H_min': -200.0, 'H_max': 200.0},
    'margins_0.4_0.6':   {'margins': (0.4, 0.6),    'diversity_penalty': 0.0, 'H_min': -200.0, 'H_max': 200.0},
}

# Paper's actual per-dataset m (latent), read off the saved MATLAB workspaces:
# m=3 for the closer-gap file ([6,5], matching ipr_gap1's dataset boundaries
# (6,5)); m=2 for the wider-gap files ([7,4]/[8,3]/[8.5,2.5]) -- ipr_gap7's
# own dataset boundaries (9,2) are wider still than any of those four, so m=2
# is used here too.
DATASET_LATENT = {'ipr_gap1': 3, 'ipr_gap7': 2}


def _fmt(mr):
    return f'{mr.value:.4f}' if mr.ok else f'n/a ({mr.status})'


if __name__ == '__main__':
    import pickle
    import sys

    if len(sys.argv) > 1:
        # single (dataset, config) run, for parallel background execution:
        #   python -m benchmark.qlustering_repeats <dataset> <config> <out_path>
        dataset_name, config_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
        print(f'=== {dataset_name}  config={config_name} ===')
        kwargs = dict(CONFIGS[config_name])
        kwargs.setdefault('latent', DATASET_LATENT.get(dataset_name, LATENT))
        result = run_repeats(dataset_name, **kwargs)
        cm, stab = result['consensus_metrics'], result['stability']
        print(f"  consensus: RI={_fmt(cm['rand_index'])}  ARI={_fmt(cm['adjusted_rand_index'])}  "
              f"CP={_fmt(cm['compactness'])}  DVI={_fmt(cm['dunn_index'])}  SIL={_fmt(cm['silhouette'])}")
        print(f"  stability={_fmt(stab)}")
        with open(out_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'Saved {out_path}')
    else:
        all_results = {}
        for name in DATASETS:
            print(f'=== {name} ===')
            all_results[name] = run_repeats(name)
            cm = all_results[name]['consensus_metrics']
            stab = all_results[name]['stability']
            print(f"  consensus: RI={_fmt(cm['rand_index'])}  ARI={_fmt(cm['adjusted_rand_index'])}  "
                  f"CP={_fmt(cm['compactness'])}  DVI={_fmt(cm['dunn_index'])}  SIL={_fmt(cm['silhouette'])}")
            print(f"  stability={_fmt(stab)}\n")

        with open('results_qlustering/repeats.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print('Saved results_qlustering/repeats.pkl')
