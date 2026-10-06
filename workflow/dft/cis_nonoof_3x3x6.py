"""DFT and cluster-expansion energies of the lowest cis arrangements on the 3x3x6 cell that break OOF.

data/dft/3x3x6/cis_nonoof_NN hold arrangements with exact cis coordination at every Nb
that are not OOF on every chain: two z-chains carry their two F one site apart. They are
the four such classes of lowest cluster-expansion energy on this cell, numbered in that
order. Each is compared with the homochiral helix (stack_000000), computed with the same
settings in the same cell.

Writes cis_nonoof_3x3x6.csv: for each run, the space group of its ideal structure, whether
it is chiral, and its energy above the helix (meV/atom) in the cluster expansion (on the
ideal structure, POSCAR.init) and in DFT (the final run, converged in one ionic step).

    python cis_nonoof_3x3x6.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from ase.io import read
from icet import ClusterExpansion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "dft" / "3x3x6"
CE = REPO / "data" / "ce" / "paircut9_5_5_ardr_n96.ce"
OUT = REPO / "build" / "dft" / "cis_nonoof_3x3x6.csv"
A_START = 3.9                    # lattice constant of the ideal starting structures (A)
A_CE = 3.902                     # lattice constant of the cluster expansion (A)


def ideal(path: Path):
    """An ideal structure, rescaled to the cluster expansion's lattice constant."""
    atoms = read(path, format="vasp")
    atoms.set_cell(np.diag(np.round(atoms.cell.lengths() / A_START)) * A_CE, scale_atoms=True)
    return atoms


def energy_per_atom(outcar: Path) -> float:
    atoms = read(outcar)
    return atoms.get_potential_energy() / len(atoms)


def main() -> None:
    ce = ClusterExpansion.read(str(CE))
    helix = DATA / "stack_000000"
    ce_helix = ce.predict(ideal(helix / "POSCAR.init"))
    dft_helix = energy_per_atom(helix / "OUTCAR")

    rows = []
    for d in sorted(DATA.glob("cis_nonoof_*")):
        init = ideal(d / "POSCAR.init")
        sym = spglib.get_symmetry_dataset((init.cell[:], init.get_scaled_positions(), init.get_atomic_numbers()),
                                          symprec=1e-3)
        rows.append({"run": d.name, "space_group": sym.international,
                     "chiral": bool(all(np.linalg.det(r) > 0 for r in sym.rotations)),
                     "ce_meV_per_atom": 1e3 * (ce.predict(init) - ce_helix),
                     "dft_meV_per_atom": 1e3 * (energy_per_atom(d / "OUTCAR") - dft_helix)})
    df = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"all above the homochiral helix in DFT: {bool((df.dft_meV_per_atom > 0).all())}; "
          f"lowest {df.dft_meV_per_atom.min():.3f} meV/atom")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
