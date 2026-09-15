"""
benchmark.datasets — synthetic clustering dataset generators, plus the Iris
reference dataset.

Copied as-is from QTN_Library/data/datasets.py (which are ports of the user's
MATLAB research generators: Overlap3DWaveFunctionGenerator.m, IPRgenerator.m,
extractIris.m), so that the classical baselines in this benchmark are
evaluated on exactly the same datasets already used for Qlustering. Only the
import path (benchmark.encodings instead of data.encodings) and the iris.csv
path have been adjusted for integration into this project. The
dataset-generation logic itself is unmodified.

generate_qm9_dataset() reads a bundled, pre-extracted CSV (qm9_data/qm9_sid97.csv)
rather than depending on rdkit/qm9pack at benchmark runtime -- see
scripts/extract_qm9.py for how that CSV was produced (a near-verbatim copy of
the user's PythonProject/QM9_featureExtraction.py logic).
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.encodings import amplitude_encode

_IRIS_CSV = Path(__file__).parent / 'iris.csv'
_QM9_CSV = Path(__file__).parent / 'qm9_data' / 'qm9_sid97.csv'

QM9_PROPERTY_COLS = [
    'RotA_GHz', 'RotB_GHz', 'RotC_GHz', 'Dipole_debye', 'Polarizability_bohr3',
    'HOMO_au', 'LUMO_au', 'HOMO_LUMO_gap_au', 'R2_bohr2', 'ZPVE_au',
    'InternalEnergy_0K_au', 'InternalEnergy_298K_au', 'Enthalphy_298K_au',
    'GibbsFreeEnergy_298K_au', 'Heatcapacity_Cv_cal_mol_K',
]


def generate_overlap3d_dataset(
    centers: np.ndarray, num_states: int, width: float, seed: int = None,
) -> tuple[list, list, np.ndarray]:
    """
    Port of Overlap3DWaveFunctionGenerator.m.

    centers: (q, 3) array, one direction per group (need not be pre-normalized).
    width:   0 = samples sit exactly on the group's center direction,
             1 = samples are uniform on the unit 2-sphere (center ignored).
    num_states is split across the q groups as evenly as possible (remainder
    distributed to the first groups), unlike the MATLAB round()-based split.

    Returns (inputs, targets, labels): inputs are unit vectors in R^3,
    targets are one-hot(q), labels are 0..q-1 ints.
    """
    rng = np.random.default_rng(seed)
    q = centers.shape[0]
    base, extra = divmod(num_states, q)
    counts = [base + 1 if i < extra else base for i in range(q)]

    inputs, targets, labels = [], [], []
    eye_q = np.eye(q)
    for i in range(q):
        n = counts[i]
        if n == 0:
            continue
        center_dir = centers[i] / np.linalg.norm(centers[i])

        uniform_points = rng.standard_normal((n, 3))
        uniform_points /= np.linalg.norm(uniform_points, axis=1, keepdims=True)

        interpolated = (1 - width) * center_dir + width * uniform_points
        interpolated /= np.linalg.norm(interpolated, axis=1, keepdims=True)

        inputs.extend(interpolated[j] for j in range(n))
        targets.extend([eye_q[i]] * n)
        labels.extend([i] * n)

    return inputs, targets, np.array(labels)


def ipr(psi: np.ndarray) -> float:
    """Inverse participation ratio: 1/sum(|psi_i|^4). k (uniform) down to 1 (one-hot)."""
    return 1.0 / np.sum(psi ** 4)


def generate_ipr_dataset(
    k: int, boundaries: tuple, num_states: int, seed: int = None, max_attempts: int = 100_000,
) -> tuple[list, list, np.ndarray]:
    """
    Port of IPRgenerator.m.

    boundaries = (b_hi, b_lo), b_hi > b_lo (NOT [low, high] despite the MATLAB
    docstring -- confirmed against a real IPRgenerator(10,[7 5],20) run: group 1
    rejection-samples until ipr >= b_hi (delocalized), group 2 until
    ipr <= b_lo (localized), leaving a gap between the two thresholds.

    First half of num_states = label 0 (delocalized/high-IPR), second half =
    label 1 (localized/low-IPR). Includes the same 3 hardcoded edge states
    (fully uniform, uniform-minus-one-site, one-hot) at the same relative
    positions (start of group 0, and end of group 1).

    Returns (inputs, targets, labels): inputs are unit vectors in R^k,
    targets are one-hot(2), labels are 0/1 ints.
    """
    b_hi, b_lo = boundaries
    rng = np.random.default_rng(seed)
    n0 = num_states // 2
    n1 = num_states - n0

    def sample_until(low, high, threshold, is_floor):
        for _ in range(max_attempts):
            raw = low + (high - low) * rng.random(k)
            psi = raw / np.linalg.norm(raw)
            psi_ipr = ipr(psi)
            if (psi_ipr >= threshold) if is_floor else (psi_ipr <= threshold):
                return psi
        raise RuntimeError(
            f"generate_ipr_dataset: could not reach IPR threshold {threshold} "
            f"for k={k} within {max_attempts} attempts"
        )

    group0 = [sample_until(0.2, 0.6, b_hi, is_floor=True) for _ in range(n0)]
    group1 = [sample_until(0.0, 0.6, b_lo, is_floor=False) for _ in range(n1)]

    group0[0] = np.sqrt(np.full(k, 1.0 / k))
    if n0 > 1:
        near_uniform = np.zeros(k)
        near_uniform[1:] = np.sqrt(1.0 / (k - 1))
        group0[1] = near_uniform
    one_hot = np.zeros(k)
    one_hot[0] = 1.0
    group1[-1] = one_hot

    inputs = group0 + group1
    labels = np.array([0] * n0 + [1] * n1)
    eye2   = np.eye(2)
    targets = [eye2[l] for l in labels]
    return inputs, targets, labels


def generate_iris_dataset(feature_indices=None) -> tuple[list, list, np.ndarray]:
    """
    Reads the bundled benchmark/iris.csv (the canonical Fisher iris values, one
    row per sample, raw columns sepal_length/sepal_width/petal_length/
    petal_width/species) and amplitude-encodes it via
    benchmark.encodings.amplitude_encode.

    feature_indices: optional subset of the 4 raw columns
    (0=sepal_length, 1=sepal_width, 2=petal_length, 3=petal_width) to keep.
    Defaults to all 4. Pass (0, 2, 3) to replicate
    Qlustering_matlab_files/extractIris.m's column selection (drops sepal
    width) for comparison against the MATLAB reference.

    Returns (inputs, targets, labels): inputs are unit vectors in
    R^len(feature_indices or 4), targets are one-hot(3) (unused by
    clustering's unsupervised cost, kept for return-shape parity with the
    other generators), labels are 0/1/2 for setosa/versicolor/virginica
    (alphabetical order, matching MATLAB's grp2idx(species) minus 1).
    """
    species_order = ['setosa', 'versicolor', 'virginica']
    rows = []
    with open(_IRIS_CSV, newline='') as f:
        for row in csv.DictReader(f):
            rows.append((
                float(row['sepal_length']), float(row['sepal_width']),
                float(row['petal_length']), float(row['petal_width']),
                species_order.index(row['species']),
            ))

    X_raw  = np.array([r[:4] for r in rows])
    labels = np.array([r[4] for r in rows])
    X      = amplitude_encode(X_raw, feature_indices)

    eye3    = np.eye(3)
    inputs  = [X[i] for i in range(len(X))]
    targets = [eye3[l] for l in labels]
    return inputs, targets, labels


def generate_qm9_dataset() -> tuple[list, None, None]:
    """
    Loads the 97-molecule QM9 subset (Sec. II D of the paper): the smallest
    non-hydrogen-atom-count molecules from qm9pack, encoded via their Sorted
    Interatomic Distances (SID) descriptor -- see scripts/extract_qm9.py for
    how qm9_data/qm9_sid97.csv was generated.

    Unlike overlap3d/ipr/iris, QM9 has no ground-truth clustering partition
    (the paper relies on internal metrics -- compactness, Dunn Index,
    silhouette, stability -- rather than RI/ARI for this dataset). targets
    and labels are therefore both None here, deliberately, rather than a
    fabricated placeholder.

    Returns (inputs, targets, labels): inputs are unit vectors in R^10
    (amplitude-encoded SID vectors, satisfying sum(|Psi(j)|^2) == 1 as in the
    paper), targets and labels are None.

    For the paper's secondary descriptor-alignment analysis (comparing the
    Qlustering partition against binary splits of 14 molecular descriptors),
    use load_qm9_properties() to get the corresponding raw property table.
    """
    df = pd.read_csv(_QM9_CSV)
    sid_cols = [c for c in df.columns if c.startswith('sid_')]
    X_raw = df[sid_cols].to_numpy()
    X = amplitude_encode(X_raw)
    inputs = [X[i] for i in range(len(X))]
    return inputs, None, None


def load_qm9_properties() -> "pd.DataFrame":
    """
    Raw (non-normalized) QM9 molecular descriptors for the 97 molecules
    returned by generate_qm9_dataset(), in the same row order -- SMILES plus
    the 15 columns in QM9_PROPERTY_COLS (paper Fig. 6 lists 14; RotA/RotB/RotC
    are included individually here). Used for the descriptor-alignment
    analysis, not for clustering itself.
    """
    return pd.read_csv(_QM9_CSV)[['SMILES'] + QM9_PROPERTY_COLS]
