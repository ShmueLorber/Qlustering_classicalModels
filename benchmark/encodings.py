"""
data.encodings — classical -> quantum state encodings for building datasets.

amplitude_encode() is the basic protocol used across every dataset in this
library: optionally select a subset of raw feature columns, then
L2-normalize onto the unit sphere so the result can be fed directly as
psi_in (amplitudes on the k input sites). models.classification and
models.clustering's encode() both call this directly, and dataset loaders
that start from raw, non-normalized tabular features (e.g.
data.datasets.generate_iris_dataset) should too, rather than re-implementing
the same normalization inline.

More elaborate encodings (basis encoding, entangled_qubits) can be added
here later, following the same (raw features) -> (unit vector) contract.
"""

import numpy as np


def amplitude_encode(x: np.ndarray, feature_indices=None) -> np.ndarray:
    """
    Select feature_indices columns of x (default: all columns), then
    L2-normalize the result onto the unit sphere.

    x: (..., n_features) -- a single feature vector or a batch of them.
    feature_indices: optional sequence of column indices to keep (e.g.
    (0, 2, 3) to drop column 1). Default keeps every feature.

    Returns an array unit-normalized along the last axis: same shape as x,
    or (..., len(feature_indices)) if a subset is given.
    """
    x = np.asarray(x, dtype=float)
    if feature_indices is not None:
        x = x[..., list(feature_indices)]
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / norm
