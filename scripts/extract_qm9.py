"""
One-off QM9 extraction script -- NOT part of the benchmark package's runtime
dependencies. Run once with an interpreter that has rdkit + qm9pack installed
(e.g. PythonProject/.venv2) to regenerate benchmark/qm9_data/qm9_sid97.csv:

    "C:\\Users\\User\\PycharmProjects\\PythonProject\\.venv2\\Scripts\\python.exe" scripts/extract_qm9.py

compute_sid() and smiles_to_geometry() are copied verbatim from the user's
PythonProject/QM9_featureExtraction.py -- unmodified. The only additions are:
  1. dropping molecules with a single non-hydrogen atom (CH4, NH3, H2O -- the
     first 3 rows of qm9pack's data), since a lone heavy atom has no pairwise
     distance and so no meaningful SID, per the user's own selection criterion
     for reaching the paper's 97-molecule subset;
  2. a fixed RDKit embedding random seed, so re-running this script reproduces
     the same cached output (the original script left this unset/random).

qm9pack's native row order is already ascending by non-hydrogen atom count
(verified directly against the raw CSV), so slicing the first 100 rows and
dropping the 3 single-heavy-atom ones reproduces the paper's "97 molecules
with the smallest number of non-hydrogen atoms" (Sec. II D) without any
extra sorting step.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import qm9pack
from rdkit import Chem
from rdkit.Chem import AllChem

N_CANDIDATES = 100
SID_LEN = 10
EMBED_SEED = 0xF00D
OUT_CSV = Path(__file__).parent.parent / 'benchmark' / 'qm9_data' / 'qm9_sid97.csv'

PROPERTY_COLS = [
    'RotA_GHz', 'RotB_GHz', 'RotC_GHz', 'Dipole_debye', 'Polarizability_bohr3',
    'HOMO_au', 'LUMO_au', 'HOMO_LUMO_gap_au', 'R2_bohr2', 'ZPVE_au',
    'InternalEnergy_0K_au', 'InternalEnergy_298K_au', 'Enthalphy_298K_au',
    'GibbsFreeEnergy_298K_au', 'Heatcapacity_Cv_cal_mol_K',
]


def compute_sid(positions, atom_numbers):
    """
    Compute Sorted Interatomic Distances (SID) descriptor.
    Removes hydrogen atoms and returns a sorted vector of all
    pairwise distances between remaining atoms.

    SID is efficient, rotation/translation invariant,
    and captures molecular geometry without explicit connectivity.
    """
    non_h = [i for i, z in enumerate(atom_numbers) if z != 1]
    positions = positions[non_h]

    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            distances.append(d)

    return np.sort(distances)


def smiles_to_geometry(smiles):
    """
    Converts a SMILES string into 3D coordinates and atomic numbers.
    Uses RDKit's ETKDG algorithm for molecular geometry embedding.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDG()
    params.randomSeed = EMBED_SEED
    AllChem.EmbedMolecule(mol, params)
    conf = mol.GetConformer()

    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return positions, atom_numbers


def non_h_count(elements_str):
    return sum(1 for e in ast.literal_eval(elements_str) if e != 'H')


def main():
    qm9_data = qm9pack.get_data('qm9')
    candidates = qm9_data.iloc[:N_CANDIDATES].copy()

    keep_mask = candidates['Elements'].apply(non_h_count) > 1
    dropped = candidates[~keep_mask]
    print(f"Dropping {len(dropped)} single-heavy-atom molecule(s): "
          f"{dropped['SMILES'].tolist()}")
    candidates = candidates[keep_mask]

    rows = []
    for _, row in candidates.iterrows():
        smiles = row['SMILES']
        try:
            R, Z = smiles_to_geometry(smiles)
            sid_vec = compute_sid(R, Z)
        except Exception as e:
            print(f"Skipping SMILES {smiles}: {e}")
            continue
        if len(sid_vec) > SID_LEN:
            raise RuntimeError(
                f"SMILES {smiles} produced SID length {len(sid_vec)} > {SID_LEN}; "
                "increase SID_LEN or investigate."
            )
        padded = np.pad(sid_vec, (0, SID_LEN - len(sid_vec)))
        rows.append({'SMILES': smiles, **{f'sid_{i}': padded[i] for i in range(SID_LEN)},
                     **{col: row[col] for col in PROPERTY_COLS}})

    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} molecules (target: 97).")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")


if __name__ == '__main__':
    main()
