"""
benchmark.algorithms — classical clustering baselines (scikit-learn), used
as comparison points against Qlustering.

Each run_*() wrapper takes a real-valued feature matrix X (n_samples,
n_features) and returns a ClusteringResult: predicted integer labels
(0..k-1, or -1 for DBSCAN noise points), wall-clock runtime, and the
parameters used. No algorithm is implemented from scratch here -- these are
thin, uniform wrappers around the corresponding sklearn estimators.

Determinism note (relevant to Step 6/7's repeated-run protocol):
  - run_spectral and run_gmm are stochastic (random_state seeds the k-means
    step on the spectral embedding, resp. the EM initialization) -- their
    output can vary across seeds on the same data.
  - run_dbscan and run_agglomerative are deterministic given fixed
    data/parameters -- there is no random_state to seed. Running either one
    across 10 "seeds" will produce 10 identical results; this is expected,
    not a bug, and should be reported as "deterministic" rather than
    plotted as if it were 10 independent stochastic trials.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

STOCHASTIC_ALGORITHMS = frozenset({'spectral', 'gmm'})
DETERMINISTIC_ALGORITHMS = frozenset({'dbscan', 'agglomerative'})


@dataclass(frozen=True)
class ClusteringResult:
    labels:          np.ndarray
    runtime_seconds: float
    algorithm:       str
    params:          dict = field(default_factory=dict)

    @property
    def is_deterministic(self) -> bool:
        return self.algorithm in DETERMINISTIC_ALGORITHMS


def run_spectral(X: np.ndarray, n_clusters: int, seed: int = None, **kwargs) -> ClusteringResult:
    """Graph/affinity-based clustering. Stochastic via `seed` (k-means on the spectral embedding)."""
    t0 = perf_counter()
    model = SpectralClustering(n_clusters=n_clusters, random_state=seed, **kwargs)
    labels = model.fit_predict(X)
    runtime = perf_counter() - t0
    return ClusteringResult(labels, runtime, 'spectral',
                             {'n_clusters': n_clusters, 'seed': seed, **kwargs})


def run_gmm(X: np.ndarray, n_clusters: int, seed: int = None, **kwargs) -> ClusteringResult:
    """Gaussian Mixture Model clustering (hard assignment = argmax posterior). Stochastic via `seed`."""
    t0 = perf_counter()
    model = GaussianMixture(n_components=n_clusters, random_state=seed, **kwargs)
    labels = model.fit(X).predict(X)
    runtime = perf_counter() - t0
    return ClusteringResult(labels, runtime, 'gmm',
                             {'n_clusters': n_clusters, 'seed': seed, **kwargs})


def estimate_dbscan_eps(X: np.ndarray, min_samples: int = 5) -> float:
    """
    k-distance heuristic (Ester et al. 1996): eps = median, over all points,
    of the distance to their min_samples-th nearest neighbor. A standard,
    reproducible (no randomness) way to pick eps from the data itself rather
    than a hand-tuned constant per dataset -- used so DBSCAN's one
    data-dependent parameter is chosen the same principled way everywhere in
    this benchmark rather than ad hoc per call site.
    """
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nn.kneighbors(X)
    kth_distances = distances[:, -1]
    return float(np.median(kth_distances))


def run_dbscan(X: np.ndarray, eps: float, min_samples: int = 5, **kwargs) -> ClusteringResult:
    """
    Density-based clustering; does not take a target cluster count. Label -1
    marks noise points -- callers/metrics must handle that explicitly (e.g.
    exclude noise from centroid-based internal metrics) rather than treating
    -1 as an ordinary cluster id. Deterministic given fixed eps/min_samples.
    """
    t0 = perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = model.fit_predict(X)
    runtime = perf_counter() - t0
    return ClusteringResult(labels, runtime, 'dbscan',
                             {'eps': eps, 'min_samples': min_samples, **kwargs})


def run_agglomerative(X: np.ndarray, n_clusters: int, linkage: str = 'average', **kwargs) -> ClusteringResult:
    """
    Hierarchical clustering, cut into n_clusters. Deterministic given fixed
    data/parameters. linkage='average' (UPGMA) matches the linkage Qlustering
    itself uses for its consensus-clustering step.
    """
    t0 = perf_counter()
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
    labels = model.fit_predict(X)
    runtime = perf_counter() - t0
    return ClusteringResult(labels, runtime, 'agglomerative',
                             {'n_clusters': n_clusters, 'linkage': linkage, **kwargs})


ALGORITHMS = {
    'spectral':      run_spectral,
    'gmm':           run_gmm,
    'dbscan':        run_dbscan,
    'agglomerative': run_agglomerative,
}
