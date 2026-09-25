"""Ordering of the twelve cis + OOF orbits in r2SCAN compared with PBEsol.

For each orbit representative in ``data/dft/3x3x3``, ``data/dft/3x3x3_r2scan``
holds an r2SCAN single point at its relaxed PBEsol geometry, with the same
k-point mesh and numerical settings, so the two energies at each geometry differ
only in the exchange-correlation functional. The PBEsol energy at that geometry
is the energy of the final PBEsol run, which converged in a single ionic step
from it. Orbit 11 (P3_121) is represented twice, in two settings.

Writes r2scan_ladder.csv: for each configuration, its orbit, the space group of
the orbit, and its energy above the lowest orbit-11 configuration in each
functional (meV/atom), in ascending PBEsol order. Prints whether orbit 11 is
lowest in both functionals and the rank correlation of the other eleven orbits.
The r2SCAN energies are single points at the PBEsol geometries, so they test the
ordering; they are not relaxed r2SCAN energies.

    python r2scan_ladder.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DFT = REPO / "data" / "dft"
REP = REPO / "data" / "structures" / "orbit_representatives"
BUILD = REPO / "build" / "dft"
DFT_OPS = BUILD / "dft_ops.csv"
OUT = BUILD / "r2scan_ladder.csv"


def energy_per_atom(outcar: Path) -> float:
    """Energy per atom, read as in dft_ops.py."""
    atoms = read(outcar)
    return atoms.get_potential_energy() / len(atoms)


def main() -> None:
    ops = pd.read_csv(DFT_OPS, dtype={"config": str})
    orbit_of = ops[ops.cls == "orbit"].set_index("config")["orbit"].astype(int).to_dict()
    space_group = {int(f.stem.split("_")[1]): "_".join(f.stem.split("_")[3:-1])   # orbit_<idx>_sg<sg>_<HM...>_stab<n>
                   for f in REP.glob("orbit_*.vasp")}

    rows = []
    for d in sorted((DFT / "3x3x3_r2scan").glob("config_*")):
        config = d.name.split("_")[1]
        orbit = orbit_of[config]
        rows.append({
            "config": config,
            "orbit": orbit,
            "space_group": space_group[orbit],
            "pbesol": energy_per_atom(DFT / "3x3x3" / d.name / "OUTCAR"),
            "r2scan": energy_per_atom(d / "OUTCAR"),
        })
    df = pd.DataFrame(rows)
    for functional in ("pbesol", "r2scan"):
        ground = df.loc[df.orbit == 11, functional].min()
        df[f"{functional}_meV_per_atom"] = 1000.0 * (df[functional] - ground)
    df = df.sort_values("pbesol_meV_per_atom")[
        ["config", "orbit", "space_group", "pbesol_meV_per_atom", "r2scan_meV_per_atom"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")

    others = df[df.orbit != 11]
    p, r = others.pbesol_meV_per_atom, others.r2scan_meV_per_atom
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\norbit 11 lowest: PBEsol {bool((p > 0).all())}, r2SCAN {bool((r > 0).all())}")
    print(f"rank correlation of the other eleven orbits: {p.rank().corr(r.rank()):.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
