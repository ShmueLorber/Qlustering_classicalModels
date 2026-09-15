"""
benchmark.metrics — external, internal, and robustness clustering metrics.

rand_index/adjusted_rand_index are copied verbatim from
QTN_Library/training/metrics.py (itself a port of
Qlustering_matlab_files/randomIndex.m) -- the exact same definitions already
used for the existing Qlustering results, so external validation is
apples-to-apples between Qlustering and the classical baselines.

compactness, dunn_index, silhouette, and stability were NOT found anywhere
in QTN_Library or QTN (confirmed by grep), so they are implemented directly
from the paper's Methods Sec. IV C formulas:
  - compactness   Eq. after (14): CP = sum_i ||x_i - c_label(i)||^2
  - dunn_index:    DVI = min_{i!=j} delta(Ci,Cj) / max_k Delta(Ck)
  - silhouette:    standard per-point s(i) = (b(i)-a(i))/max(a(i),b(i)),
                   delegated to sklearn.metrics.silhouette_score, which
                   implements this exact definition -- not reimplemented
                   by hand, per "use standard implementations" (Step 3's
                   instruction applies equally well here). metric='sqeuclidean'
                   is passed explicitly: MATLAB's silhouette(phi,classification)
                   (Internal_metrics.mlx, the actual paper reference code)
                   defaults to squared Euclidean distance, whereas sklearn's
                   default is plain Euclidean -- matching the reference
                   exactly required overriding sklearn's default.
  - stability:     Sec. IV C.e, Stability = 2/(R(R-1)) * sum_{i<j} Match(Ri,Rj),
                   Match = fraction of matching labels after Hungarian
                   alignment. Cluster labels are arbitrary, so this (not raw
                   label equality) is the permutation-invariant way to
                   compare two runs.

Every function returns a MetricResult(value, status, notes) instead of a
bare float. status is 'ok', 'undefined', or 'error'; value is None whenever
status != 'ok'. This is deliberate: silhouette is mathematically undefined
for k=1 or k=n clusters, Dunn's denominator can be zero, DBSCAN's -1 noise
label doesn't belong to any real cluster, and RI/ARI need ground truth that
QM9 doesn't have. Forcing a numeric placeholder in any of these cases would
misrepresent the result, so callers must handle status explicitly instead.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score as _sklearn_silhouette_score

NOISE_LABEL = -1  # sklearn's DBSCAN convention for unclustered points


@dataclass(frozen=True)
class MetricResult:
    value:  Optional[float]
    status: str = 'ok'          # 'ok' | 'undefined' | 'error'
    notes:  str = ''            # human-readable context, e.g. why undefined,
                                 # or how many noise points were excluded

    @property
    def ok(self) -> bool:
        return self.status == 'ok'


# ---------------------------------------------------------------------------
# External metrics -- verbatim port of QTN_Library/training/metrics.py
# (Qlustering_matlab_files/randomIndex.m).
# ---------------------------------------------------------------------------

def _comb2(x):
    """n choose 2, elementwise; x may be a scalar or an ndarray of counts."""
    x = np.asarray(x, dtype=float)
    return x * (x - 1) / 2.0


def _contingency_table(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    table = np.zeros((len(true_classes), len(pred_classes)), dtype=int)
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            table[i, j] = np.sum((y_true == t) & (y_pred == p))
    return table


def rand_index(y_true, y_pred) -> MetricResult:
    """
    Fraction of sample pairs on which y_true and y_pred agree (both
    same-cluster or both different-cluster). Permutation-invariant to
    cluster/label relabeling. Undefined when no ground truth exists (QM9).
    """
    if y_true is None:
        return MetricResult(None, 'undefined', 'no ground-truth labels available (e.g. QM9)')

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    total_pairs = _comb2(n)
    if total_pairs == 0:
        return MetricResult(1.0, 'ok', 'fewer than 2 samples; trivially defined as 1.0')

    table          = _contingency_table(y_true, y_pred)
    sum_comb_table = np.sum(_comb2(table))
    sum_comb_rows  = np.sum(_comb2(table.sum(axis=1)))
    sum_comb_cols  = np.sum(_comb2(table.sum(axis=0)))

    agree_same = sum_comb_table
    agree_diff = total_pairs - sum_comb_rows - sum_comb_cols + sum_comb_table
    return MetricResult(float((agree_same + agree_diff) / total_pairs))


def adjusted_rand_index(y_true, y_pred) -> MetricResult:
    """Rand Index corrected for chance agreement (can be negative; 1.0 = perfect)."""
    if y_true is None:
        return MetricResult(None, 'undefined', 'no ground-truth labels available (e.g. QM9)')

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    total_pairs = _comb2(n)
    if total_pairs == 0:
        return MetricResult(1.0, 'ok', 'fewer than 2 samples; trivially defined as 1.0')

    table          = _contingency_table(y_true, y_pred)
    sum_comb_table = np.sum(_comb2(table))
    sum_comb_rows  = np.sum(_comb2(table.sum(axis=1)))
    sum_comb_cols  = np.sum(_comb2(table.sum(axis=0)))

    expected_index = sum_comb_rows * sum_comb_cols / total_pairs
    max_index      = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom          = max_index - expected_index
    if denom == 0:
        return MetricResult(1.0, 'ok', 'zero denominator (degenerate partition); defined as 1.0')
    return MetricResult(float((sum_comb_table - expected_index) / denom))


# ---------------------------------------------------------------------------
# Shared noise-handling helper for internal metrics.
# ---------------------------------------------------------------------------

def _drop_noise(X: np.ndarray, labels: np.ndarray):
    """
    Returns (X_clean, labels_clean, n_noise). Points with label ==
    NOISE_LABEL (DBSCAN's unclustered convention) don't belong to any real
    cluster, so internal metrics that need a centroid/diameter per cluster
    must exclude them rather than silently treating -1 as an ordinary
    cluster id.
    """
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    return np.asarray(X)[mask], labels[mask], int((~mask).sum())


# ---------------------------------------------------------------------------
# Internal metrics.
# ---------------------------------------------------------------------------

def compactness(X: np.ndarray, labels: np.ndarray) -> MetricResult:
    """
    CP = sum_i ||x_i - c_label(i)||^2, c_k = centroid of cluster k.
    Lower is more compact (tighter clusters). Paper Methods, Sec. IV C b.
    """
    X_clean, labels_clean, n_noise = _drop_noise(X, labels)
    notes = f'{n_noise} noise point(s) excluded' if n_noise else ''

    if len(X_clean) == 0:
        return MetricResult(None, 'undefined', 'all points were noise (label -1); no clusters to measure')

    cp = 0.0
    for c in np.unique(labels_clean):
        members = X_clean[labels_clean == c]
        centroid = members.mean(axis=0)
        cp += np.sum((members - centroid) ** 2)
    return MetricResult(float(cp), 'ok', notes)


def dunn_index(X: np.ndarray, labels: np.ndarray) -> MetricResult:
    """
    DVI = min_{i!=j} delta(Ci,Cj) / max_k Delta(Ck), where delta is the
    minimum pairwise distance between two clusters and Delta is a cluster's
    diameter (max pairwise distance within it). Higher is better (well
    separated, compact clusters). Paper Methods, Sec. IV C c.

    Undefined for fewer than 2 real clusters (no inter-cluster distance to
    take a minimum over), or when every cluster is a singleton/identical
    points (zero diameter -> division by zero).
    """
    X_clean, labels_clean, n_noise = _drop_noise(X, labels)
    notes = f'{n_noise} noise point(s) excluded' if n_noise else ''
    clusters = np.unique(labels_clean)

    if len(clusters) < 2:
        return MetricResult(None, 'undefined',
                             f'only {len(clusters)} real cluster(s) after excluding noise; '
                             'Dunn Index requires >=2 for an inter-cluster distance. ' + notes)

    members = {c: X_clean[labels_clean == c] for c in clusters}

    max_intra = 0.0
    for c, pts in members.items():
        if len(pts) < 2:
            continue
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        max_intra = max(max_intra, d.max())

    if max_intra == 0.0:
        return MetricResult(None, 'undefined',
                             'every cluster is a singleton or contains identical points '
                             '(zero intra-cluster diameter -> division by zero). ' + notes)

    min_inter = np.inf
    for i, ci in enumerate(clusters):
        for cj in clusters[i + 1:]:
            d = np.sqrt(((members[ci][:, None, :] - members[cj][None, :, :]) ** 2).sum(-1))
            min_inter = min(min_inter, d.min())

    return MetricResult(float(min_inter / max_intra), 'ok', notes)


def silhouette(X: np.ndarray, labels: np.ndarray) -> MetricResult:
    """
    Standard silhouette score (Rousseeuw 1987), delegated to
    sklearn.metrics.silhouette_score with metric='sqeuclidean' -- matching
    MATLAB's silhouette(phi,classification) default distance metric exactly
    (verified against Internal_metrics.mlx, the paper's reference code;
    sklearn's own default is plain, unsquared Euclidean, which would give
    numerically different values). Undefined for k=1 (no other cluster to
    separate from) or k==n_samples (every point its own cluster) -- sklearn
    raises ValueError in both cases, caught here and reported as 'undefined'
    rather than propagating an exception.
    """
    X_clean, labels_clean, n_noise = _drop_noise(X, labels)
    notes = f'{n_noise} noise point(s) excluded' if n_noise else ''
    n_clusters = len(np.unique(labels_clean))

    if not (2 <= n_clusters <= len(X_clean) - 1):
        return MetricResult(None, 'undefined',
                             f'silhouette requires 2 <= n_clusters <= n_samples-1, got '
                             f'n_clusters={n_clusters}, n_samples={len(X_clean)}. ' + notes)
    try:
        score = _sklearn_silhouette_score(X_clean, labels_clean, metric='sqeuclidean')
    except ValueError as e:
        return MetricResult(None, 'undefined', str(e))
    return MetricResult(float(score), 'ok', notes)


# ---------------------------------------------------------------------------
# Robustness metric: stability via Hungarian label alignment.
# ---------------------------------------------------------------------------

def _best_match_fraction(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Fraction of samples on which labels_a and labels_b agree after the
    optimal (Hungarian) relabeling of labels_b onto labels_a's label set.
    Treats NOISE_LABEL as its own ordinary class (a run that calls a point
    "noise" both times counts as agreement; noise in one run vs. a real
    cluster in the other counts as disagreement), rather than excluding it,
    since stability is about run-to-run consistency of the full assignment.
    """
    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)
    cost = np.zeros((len(classes_a), len(classes_b)), dtype=int)
    for i, a in enumerate(classes_a):
        for j, b in enumerate(classes_b):
            cost[i, j] = np.sum((labels_a == a) & (labels_b == b))
    row_ind, col_ind = linear_sum_assignment(-cost)  # maximize matched count
    matched = cost[row_ind, col_ind].sum()
    return float(matched / len(labels_a))


def stability(list_of_label_arrays: list) -> MetricResult:
    """
    Stability = 2/(R(R-1)) * sum_{i<j} Match(Ri, Rj), the mean pairwise
    Hungarian-matched label-alignment accuracy over R independent runs.
    Paper Methods, Sec. IV C e. 1.0 = every run agrees perfectly after
    optimal relabeling; lower = less run-to-run consistency.

    Requires R >= 2 runs. Note: for a deterministic algorithm (e.g. DBSCAN,
    Agglomerative -- see benchmark.algorithms.DETERMINISTIC_ALGORITHMS), all
    R runs on the same data/params are bit-identical, so this will trivially
    equal 1.0 -- report that fact rather than treating it as evidence of
    genuine stochastic robustness.
    """
    R = len(list_of_label_arrays)
    if R < 2:
        return MetricResult(None, 'undefined', f'need at least 2 runs to compute stability, got {R}')

    matches = []
    for i in range(R):
        for j in range(i + 1, R):
            matches.append(_best_match_fraction(
                np.asarray(list_of_label_arrays[i]), np.asarray(list_of_label_arrays[j])
            ))
    return MetricResult(float(np.mean(matches)), 'ok', f'{len(matches)} pairwise comparisons over {R} runs')


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, y_true=None) -> dict:
    """
    Convenience bundle: every applicable metric for a single clustering
    result. Does not include `stability`, which needs multiple runs and is
    therefore computed separately by the caller once all runs are in hand.
    """
    return {
        'rand_index':          rand_index(y_true, labels),
        'adjusted_rand_index': adjusted_rand_index(y_true, labels),
        'compactness':         compactness(X, labels),
        'dunn_index':          dunn_index(X, labels),
        'silhouette':          silhouette(X, labels),
    }
