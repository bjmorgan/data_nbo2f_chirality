"""Point-charge electrostatics of the twelve cis + OOF orbits against their DFT energies (NbO2F).

For each orbit representative (data/structures/orbit_representatives, ideal
positions, all on the same cubic cell):
  - the number of fluoride pairs sitting side by side on neighbouring parallel
    chains, i.e. separated by one lattice spacing a. Every rule-obeying arrangement
    has the same 27 cis fluoride pairs (separation a/sqrt2) per cell, which is
    checked, so the orbits differ in this next shell;
  - the Ewald energy with formal charges (Nb5+, O2-, F-). With the cation
    sublattice and the anion sites fixed, it varies only with the arrangement of
    the fluorides' excess charge (+1 relative to O2-). It is exactly proportional
    to the pair count, which is checked;
  - the DFT energy of the orbit in data/dft/3x3x3 (for orbit 11, the lower of its
    two settings), from dft_ops.csv.
All energies are relative to orbit 11 (P3_121).

Writes orbit_electrostatics.csv, one row per orbit. Prints the point-charge energy
per pair and the linear fit of the DFT energies against the pair count.

    python orbit_electrostatics.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPS = REPO / "data" / "structures" / "orbit_representatives"
DFT_OPS = REPO / "build" / "dft" / "dft_ops.csv"
OUT = REPO / "build" / "dft" / "orbit_electrostatics.csv"


def point_charge_meV_per_atom(atoms) -> float:
    s = AseAtomsAdaptor.get_structure(atoms)
    s.add_oxidation_state_by_element({"Nb": 5, "O": -2, "F": -1})
    return 1e3 * EwaldSummation(s).total_energy / len(atoms)


def side_by_side_pairs(atoms) -> int:
    """Fluoride pairs one lattice spacing apart (neighbouring parallel chains)."""
    f = np.array([i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "F"])
    a = atoms.cell.lengths()[0] / 3
    d = atoms.get_all_distances(mic=True)[np.ix_(f, f)] / a
    shells = np.round(d[np.triu_indices(len(f), 1)] ** 2 * 4).astype(int)   # (d/a)^2 in quarters
    assert (shells == 2).sum() == 27, "expected 27 cis fluoride pairs per cell"
    return int((shells == 4).sum())


def main() -> None:
    ops = pd.read_csv(DFT_OPS, dtype={"config": str})
    dft = ops[ops.cls == "orbit"].groupby("orbit").energy_per_atom.min()

    rows = []
    cells = set()
    for f in sorted(REPS.glob("orbit_*.vasp")):
        k = int(f.stem.split("_")[1])
        atoms = read(f, format="vasp")
        cells.add(tuple(np.round(atoms.cell.array.ravel(), 6)))
        rows.append({"orbit": k, "space_group": "_".join(f.stem.split("_")[3:-1]),
                     "side_by_side_pairs": side_by_side_pairs(atoms),
                     "point_charge": point_charge_meV_per_atom(atoms),
                     "dft": 1e3 * dft.loc[k]})
    assert len(cells) == 1, "the orbit representatives are not on the same cell"
    df = pd.DataFrame(rows).set_index("orbit")
    for col in ("point_charge", "dft"):
        df[f"{col}_meV_per_atom"] = df[col] - df.loc[11, col]
    df = df.drop(columns=["point_charge", "dft"]).sort_values("dft_meV_per_atom")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, float_format="%.6f")

    x = df.side_by_side_pairs
    per_pair = df.point_charge_meV_per_atom.drop(index=11) / x.drop(index=11)
    assert np.allclose(per_pair, per_pair.iloc[0]), "point-charge energy is not proportional to the pair count"
    slope, intercept = np.polyfit(x, df.dft_meV_per_atom, 1)
    r2 = np.corrcoef(x, df.dft_meV_per_atom)[0, 1] ** 2
    print(df.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\npoint charges: {per_pair.iloc[0]:.4f} meV/atom per side-by-side pair (exactly proportional)")
    print(f"DFT against the pair count: slope {slope:.4f} meV/atom per pair, intercept {intercept:+.3f}, R^2 {r2:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
