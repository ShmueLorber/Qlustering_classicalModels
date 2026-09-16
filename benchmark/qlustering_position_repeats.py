"""
benchmark.qlustering_position_repeats -- diagnostic-only reproduction attempt
for the paper's position task (overlap3d, 5 clusters), using QTN_Library's
training/qlustering_search.py (the MATLAB-faithful hill-climb) with the
RI/ARI-based early stop that Qlustering_matlab_files/QlusteringFunction.m
actually used, and that the QTN_Library port deliberately dropped (see
training/qlustering_search.py's own docstring: "minus the RI-based early
stop, which needs external labels not available to a genuinely unsupervised
run"). Restoring it here is a supervised, diagnostic-only choice -- it is
NOT how Qlustering would run in production/unsupervised use, only how we
check whether the search converges to the paper's reported RI/ARI at all.

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

Consensus combines the 10 runs' *best-found* label vectors via
training.consensus.consensus_clustering (verbatim import, pure math) and
scores the consensus assignment the same way.
"""

import numpy as np

from core.hamiltonian import random_H_vec  # QTN_Library
from models.base import Topology  # QTN_Library
from models.clustering import ClusteringModel  # QTN_Library
from training.qlustering_search import _cost_and_predictions, _free_position_mask  # QTN_Library
from training.consensus import consensus_clustering  # QTN_Library

from benchmark.dataset_instances import DATASET_INSTANCES
from benchmark.metrics import rand_index, adjusted_rand_index

LATENT = 2
N_ITERATIONS = 50
NUM_PARTICLES = 20
N_RUNS = 10
GAMMA_DEP = 0.0  # pinned per user instruction -- do not change


def qlustering_search_monitored(
    model, H_vec_init, inputs, y_true, n_iterations=N_ITERATIONS, num_particles=NUM_PARTICLES,
    H_min=-5.0, H_max=5.0, min_group=None, seed=None,
):
    rng = np.random.default_rng(seed)
    topo = model.topology()
    free_idx = np.flatnonzero(_free_position_mask(topo).reshape(-1))

    best_H = H_vec_init.copy()
    best_cost, best_preds = _cost_and_predictions(model, best_H, inputs)
    best_occ = np.bincount(best_preds, minlength=topo.output_sites)
    while best_occ.min() == 0:
        best_H = rng.uniform(H_min, H_max, size=H_vec_init.shape)
        best_cost, best_preds = _cost_and_predictions(model, best_H, inputs)
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
            cost, preds = _cost_and_predictions(model, H_try, inputs)
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


def run_repeats(dataset_name, n_runs=N_RUNS, seed_offset=0):
    ds = DATASET_INSTANCES[dataset_name]
    n_features = ds.X.shape[1]
    inputs = list(ds.X)

    run_results = []
    for r in range(n_runs):
        seed = seed_offset + r
        topo = Topology(input_sites=n_features, latent=LATENT, output_sites=ds.q)
        model = ClusteringModel(topo, gamma_in=np.ones(n_features), gamma_out=np.ones(ds.q), gamma_dep=GAMMA_DEP)
        H_init = random_H_vec(topo.input_sites, topo.latent, topo.output_sites, scale=0.5, seed=seed)
        result = qlustering_search_monitored(model, H_init, inputs, ds.y_true, seed=seed)
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
    consensus_ri = rand_index(ds.y_true, consensus_labels).value
    consensus_ari = adjusted_rand_index(ds.y_true, consensus_labels).value

    return {
        'dataset': dataset_name, 'run_results': run_results,
        'consensus_labels': consensus_labels,
        'consensus_ri': consensus_ri, 'consensus_ari': consensus_ari,
    }


if __name__ == '__main__':
    import pickle

    all_results = {}
    for name in ('overlap3d_w0.1', 'overlap3d_w0.3'):
        print(f'=== {name} ===')
        all_results[name] = run_repeats(name)
        ri_vals = [r['ri_curve'][r['best_iter']] for r in all_results[name]['run_results']]
        ari_vals = [r['ari_curve'][r['best_iter']] for r in all_results[name]['run_results']]
        print(f"  mean RI={np.mean(ri_vals):.4f}  mean ARI={np.mean(ari_vals):.4f}")
        print(f"  consensus RI={all_results[name]['consensus_ri']:.4f}  consensus ARI={all_results[name]['consensus_ari']:.4f}\n")

    with open('results_qlustering/position_repeats.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print('Saved results_qlustering/position_repeats.pkl')
