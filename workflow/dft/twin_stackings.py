"""Energies of the 3x3x6 stackings of the P3_121 helix, relative to the homochiral one.

Each directory data/dft/3x3x6/stack_<s> holds a relaxation of the stacking s: in each
(001) plane of cations, 0 keeps the P3_121 labelling and 1 exchanges its x and y
labels (the P3_221 enantiomer), so every change between adjacent planes is a twin
boundary between the two hands. stack_000000 is the homochiral reference, computed
with the same settings in the same cell.

Writes twin_stackings.csv: for each stacking, the number of twin boundaries per cell,
the widths of the domains between them (in planes), the energy per atom, the energy
above the homochiral stacking (meV/atom), and the energy per boundary, both per
cation column (meV) and per unit area (mJ/m^2, using the relaxed cell's (001) area).

    python twin_stackings.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "dft" / "3x3x6"
OUT = REPO / "build" / "dft" / "twin_stackings.csv"
MEV_PER_A2_TO_MJ_PER_M2 = 16.021766        # 1 meV/A^2 = 16.02 mJ/m^2


def boundaries(s: str) -> list[int]:
    """Plane indices k with a twin boundary between planes k and k+1 (periodic)."""
    return [k for k in range(len(s)) if s[k] != s[(k + 1) % len(s)]]


def domain_widths(s: str) -> list[int]:
    b = boundaries(s)
    return sorted((b[(i + 1) % len(b)] - b[i]) % len(s) or len(s) for i in range(len(b)))


def main() -> None:
    rows = []
    for d in sorted(DATA.glob("stack_*")):
        s = d.name.split("_")[1]
        atoms = read(d / "OUTCAR")
        a, b = atoms.cell.array[0], atoms.cell.array[1]
        columns = 9                                  # 3 x 3 cation columns cross each (001) plane
        rows.append({"stacking": s, "atoms": len(atoms), "energy": atoms.get_potential_energy(),
                     "column_area_A2": float(np.linalg.norm(np.cross(a, b))) / columns,
                     "boundaries_per_cell": len(boundaries(s)),
                     "domain_widths": ";".join(map(str, domain_widths(s)))})
    df = pd.DataFrame(rows)
    ref = df.loc[df.stacking == "0" * len(df.stacking.iloc[0]), "energy"].item()
    df["energy_per_atom_eV"] = df.energy / df.atoms
    df["dE_meV_per_atom"] = 1000.0 * (df.energy - ref) / df.atoms
    per_column = 1000.0 * (df.energy - ref) / (df.boundaries_per_cell * 9)
    df["boundary_meV_per_column"] = per_column.where(df.boundaries_per_cell > 0)
    df["boundary_mJ_per_m2"] = (per_column / df.column_area_A2 * MEV_PER_A2_TO_MJ_PER_M2).where(df.boundaries_per_cell > 0)
    df = df.sort_values("boundaries_per_cell")[
        ["stacking", "boundaries_per_cell", "domain_widths", "energy_per_atom_eV", "dE_meV_per_atom",
         "boundary_meV_per_column", "boundary_mJ_per_m2"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
