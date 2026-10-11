"""TaO2F compared with NbO2F on the achiral arrangements of the 3x3x6 cell.

Each run in data/dft/TaO2F_3x3x6 has the starting structure of the NbO2F run of the same
name in data/dft/3x3x6, with Ta in place of Nb, and the same INCAR and KPOINTS: the
homochiral helix (stack_000000), the three stackings of its two hands (stack_<s>) and
the twelve classes achiral_NN, which with the stackings are every achiral class on the
cell that obeys both rules.

Writes tao2f_3x3x6.csv: for each achiral class, its space group (from achiral_3x3x6.csv),
its stacking for the stack_<s> runs, and its energy above the homochiral helix in the
same cell in each compound (meV/atom), in ascending NbO2F order. Prints the range of the
TaO2F energies and whether every class lies above the helix, the Spearman rank
correlation of the two compounds' energies, every pair of classes whose order differs
between them with its gap in each, and the ratio of the TaO2F to the NbO2F energy of
each stacking.

    python tao2f_3x3x6.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DFT = REPO / "data" / "dft"
ACHIRAL = REPO / "build" / "dft" / "achiral_3x3x6.csv"
OUT = REPO / "build" / "dft" / "tao2f_3x3x6.csv"
HELIX = "stack_000000"


def energy_per_atom(outcar: Path) -> float:
    """Energy per atom, read as in dft_ops.py."""
    atoms = read(outcar)
    return atoms.get_potential_energy() / len(atoms)


def main() -> None:
    sg = pd.read_csv(ACHIRAL).set_index("run").space_group
    runs = sorted(d.name for d in (DFT / "TaO2F_3x3x6").iterdir() if d.is_dir())
    assert HELIX in runs and sorted(set(runs) - {HELIX}) == sorted(sg.index), "expected the helix and every achiral class"

    for r in runs:                                   # same starting structure, Ta in place of Nb
        ta, nb = (read(DFT / d / r / "POSCAR.init", format="vasp") for d in ("TaO2F_3x3x6", "3x3x6"))
        assert np.allclose(ta.cell, nb.cell) and np.allclose(ta.positions, nb.positions)
        assert [s.replace("Nb", "Ta") for s in nb.get_chemical_symbols()] == ta.get_chemical_symbols()

    ref = {c: energy_per_atom(DFT / d / HELIX / "OUTCAR") for c, d in (("tao2f", "TaO2F_3x3x6"), ("nbo2f", "3x3x6"))}
    rows = []
    for r in runs:
        if r == HELIX:
            continue
        rows.append({
            "run": r,
            "space_group": sg[r],
            "stacking": r.split("_")[1] if r.startswith("stack_") else "",
            "tao2f_meV_per_atom": 1000.0 * (energy_per_atom(DFT / "TaO2F_3x3x6" / r / "OUTCAR") - ref["tao2f"]),
            "nbo2f_meV_per_atom": 1000.0 * (energy_per_atom(DFT / "3x3x6" / r / "OUTCAR") - ref["nbo2f"]),
        })
    df = pd.DataFrame(rows).sort_values("nbo2f_meV_per_atom")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")

    ta, nb = df.tao2f_meV_per_atom, df.nbo2f_meV_per_atom
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n{len(df)} achiral classes; TaO2F: all above the helix {bool((ta > 0).all())}, "
          f"{ta.min():.3f} to {ta.max():.3f} meV/atom; NbO2F: {nb.min():.3f} to {nb.max():.3f} meV/atom")
    print(f"Spearman rank correlation of the two compounds' energies: {ta.rank().corr(nb.rank()):.3f}")
    classes = list(df.itertuples())
    swaps = [(a, b) for a in classes for b in classes
             if a.nbo2f_meV_per_atom < b.nbo2f_meV_per_atom and a.tao2f_meV_per_atom > b.tao2f_meV_per_atom]
    print(f"pairs in a different order in the two compounds: {len(swaps)}")
    for a, b in swaps:
        print(f"  {a.run} / {b.run}: NbO2F gap {b.nbo2f_meV_per_atom - a.nbo2f_meV_per_atom:.3f}, "
              f"TaO2F gap {a.tao2f_meV_per_atom - b.tao2f_meV_per_atom:.3f} meV/atom (reversed)")
    st = df[df.stacking != ""]
    for s in st.itertuples():
        print(f"stacking {s.stacking}: TaO2F {s.tao2f_meV_per_atom:.3f}, NbO2F {s.nbo2f_meV_per_atom:.3f} meV/atom, "
              f"ratio {s.tao2f_meV_per_atom / s.nbo2f_meV_per_atom:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
