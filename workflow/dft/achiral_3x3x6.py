"""DFT energies of the achiral arrangements on the 3x3x6 cell, against the cluster expansion and their blocks.

data/dft/3x3x6 holds relaxations of every achiral class of arrangement on the 3x3x6
cell that obeys both ordering rules (enumerated by methods/dft/enumerate_3x3x6.py):
the stackings stack_<s> of the P3_121 helix and the classes achiral_NN, which this script
reads. It also holds the homochiral helix, stack_000000, computed with the same settings
in the same cell.

For each achiral class it reads the labelling from POSCAR.init and finds its space
group and the orbits of its six three-plane windows. It compares the energy above
the homochiral helix in DFT with the cluster-expansion prediction and with the mean
of the 3x3x3 DFT energies of the window orbits (relative to orbit 11).

Writes achiral_3x3x6.csv: for each class, its space group, window orbits and the
three energies (meV/atom), in ascending DFT order. Prints whether every class lies
above the helix, the range of their energies, and the rank correlations of the DFT
energies with the other two.

    python achiral_3x3x6.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from icet import ClusterExpansion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from build_stackings import A, fluorides, space_group  # noqa: E402
from enumerate_3x3x6 import CE, atoms_for_ce, valid_blocks  # noqa: E402

DATA = REPO / "data" / "dft" / "3x3x6"
DFT_OPS = REPO / "build" / "dft" / "dft_ops.csv"
OUT = REPO / "build" / "dft" / "achiral_3x3x6.csv"
HELIX = "stack_000000"


def labels_from_poscar(path: Path) -> np.ndarray:
    """Each cation's fluoride-free axis, from an ideal structure."""
    atoms = read(path, format="vasp")
    shape = tuple(int(round(n)) for n in atoms.cell.lengths() / A)
    F = np.zeros((3,) + shape, dtype=bool)
    for p, sym in zip(atoms.get_scaled_positions() * shape, atoms.get_chemical_symbols()):
        if sym == "Nb":
            continue
        ax = int(np.argmax(np.abs(p - np.round(p))))
        b = np.round(p).astype(int)
        b[ax] = int(np.floor(p[ax]))
        F[(ax,) + tuple(b % shape)] = sym == "F"
    L = np.empty(shape, dtype=np.int64)
    for c in itertools.product(*map(range, shape)):
        free = [ax for ax in range(3) if not F[(ax,) + c]
                and not F[(ax,) + tuple((np.array(c) - np.eye(3, dtype=int)[ax]) % shape)]]
        L[c] = free[0]
    assert np.array_equal(fluorides(L), F)
    return L


def energy_per_atom(outcar: Path) -> float:
    """Energy per atom, read as in dft_ops.py."""
    atoms = read(outcar)
    return atoms.get_potential_energy() / len(atoms)


def main() -> None:
    ops = pd.read_csv(DFT_OPS, dtype={"config": str})
    orbits = ops[ops.cls == "orbit"].groupby("orbit").energy_per_atom.min()
    orbit_meV = (1000.0 * (orbits - orbits.loc[11])).to_dict()

    orbit_of = valid_blocks()
    ce = ClusterExpansion.read(str(CE))
    helix = labels_from_poscar(DATA / HELIX / "POSCAR.init")
    e_helix, ce_helix = energy_per_atom(DATA / HELIX / "OUTCAR"), ce.predict(atoms_for_ce(helix))

    rows = []
    for d in sorted([*DATA.glob("stack_*"), *DATA.glob("achiral_*")]):
        if not d.is_dir() or d.name == HELIX:
            continue
        L = labels_from_poscar(d / "POSCAR.init")
        sg, _, _, improper = space_group(fluorides(L))
        assert improper, f"{d.name} is chiral"
        nz = L.shape[2]
        windows = sorted(orbit_of[tuple(L[:, :, (z + k) % nz].tobytes() for k in range(3))] for z in range(nz))
        rows.append({
            "run": d.name,
            "space_group": sg,
            "window_orbits": " ".join(map(str, windows)),
            "dft_meV_per_atom": 1000.0 * (energy_per_atom(d / "OUTCAR") - e_helix),
            "ce_meV_per_atom": 1000.0 * (ce.predict(atoms_for_ce(L)) - ce_helix),
            "block_mean_meV_per_atom": float(np.mean([orbit_meV[k] for k in windows])),
        })
    df = pd.DataFrame(rows).sort_values("dft_meV_per_atom")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")

    dft = df.dft_meV_per_atom
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n{len(df)} achiral classes; all above the homochiral helix: {bool((dft > 0).all())}; "
          f"{dft.min():.3f} to {dft.max():.3f} meV/atom")
    print(f"rank correlation of the DFT energies with the CE {dft.rank().corr(df.ce_meV_per_atom.rank()):.2f}, "
          f"with the block means {dft.rank().corr(df.block_mean_meV_per_atom.rank()):.2f}")
    diff = dft - df.block_mean_meV_per_atom
    print(f"DFT minus block mean: {diff.min():+.3f} to {diff.max():+.3f} meV/atom")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
