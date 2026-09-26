"""Build per-plane stackings of the P3_121 helix: the twin-boundary structures.

In the P3_121 ground state each Nb is labelled by its fluoride-free axis,
L(i, j, k) = G[(i + j + k) mod 3] with G = (x, y, z). Exchanging the x and y
labels within a single (001) plane of cations preserves both ordering rules
(cis at every Nb, one F per three sites along every chain), so each plane k
carries an independent choice s(k): 0 keeps the helix and 1 takes its mirror
image, the P3_221 enantiomer, in that plane. Wherever s changes between
adjacent planes there is a twin boundary between the two hands.

For each stacking string s this writes the ideal cubic structure on a
3 x 3 x len(s) cell (a = 3.9 A, the starting geometry of the 3x3x3 set) to
<out>/stack_<s>/POSCAR.init. Every structure is checked for exact cis + OOF
and its space group is reported. The construction is checked first against the
3x3x3 orbits: s = 000 must give orbit 11 (P3_121, 54 operations in the cell)
and s = 001 orbit 10 (C2, 6 operations), the only orbits with those symmetries.

    python build_stackings.py --out <directory> 000000 000111 001011 010101

Not part of the `make` pipeline: the structures are the starting geometries of
VASP relaxations run externally.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import spglib

A = 3.9                        # starting lattice constant of the 3x3x3 set (Angstrom)
N = 3                          # cations per in-plane cell edge
G = [(0, 1, 2), (1, 0, 2)]     # fluoride-free axis at residue (i + j + k) mod 3; s = 1 swaps x and y


def labels(s: str) -> np.ndarray:
    L = np.empty((N, N, len(s)), dtype=int)
    for i, j, k in itertools.product(range(N), range(N), range(len(s))):
        L[i, j, k] = G[int(s[k])][(i + j + k) % 3]
    return L


def fluorides(L: np.ndarray) -> np.ndarray:
    """F[a, i, j, k]: True if the axis-a anion between cation (i,j,k) and (i,j,k) + e_a is F.

    Along a chain on axis a, the cation labelled a has O on both sides; the chain's
    F sits on the next bond but one, between the other two cations of the period.
    """
    shape = L.shape
    F = np.zeros((3,) + shape, dtype=bool)
    for c in itertools.product(*map(range, shape)):
        a = L[c]
        nxt = list(c)
        nxt[a] = (c[a] + 1) % shape[a]
        F[(a,) + tuple(nxt)] = True
    return F


def obeys_rules(F: np.ndarray) -> tuple[bool, bool]:
    """(cis at every Nb, exactly one F per three sites and period three on every chain)."""
    shape = F.shape[1:]
    cis = True
    for c in itertools.product(*map(range, shape)):
        per_axis = []
        for a in range(3):
            prev = list(c)
            prev[a] = (c[a] - 1) % shape[a]
            per_axis.append(int(F[(a,) + c]) + int(F[(a,) + tuple(prev)]))
        cis &= sum(per_axis) == 2 and max(per_axis) == 1
    oof = True
    for a in range(3):
        n = shape[a]
        for rest in itertools.product(*[range(shape[d]) for d in range(3) if d != a]):
            idx = list(rest)
            idx.insert(a, slice(None))
            seq = F[a][tuple(idx)]
            oof &= n % 3 == 0 and all(seq[t] == seq[(t + 3) % n] for t in range(n)) and seq[:3].sum() == 1
    return cis, oof


def sites(F: np.ndarray) -> tuple[list, list, list]:
    """Fractional coordinates of Nb, O and F, cations in (i, j, k) order with k fastest."""
    shape = np.array(F.shape[1:])
    nb, o, f = [], [], []
    for c in itertools.product(*map(range, shape)):
        nb.append(np.array(c, float) / shape)
        for a in range(3):
            r = np.array(c, float)
            r[a] += 0.5
            (f if F[(a,) + c] else o).append(r / shape)
    return nb, o, f


def space_group(F: np.ndarray) -> tuple[str, int, int, bool]:
    """Space group symbol and number, operations in the cell, and whether any is improper."""
    shape = F.shape[1:]
    nb, o, f = sites(F)
    positions = np.array(nb + o + f)
    numbers = [41] * len(nb) + [8] * len(o) + [9] * len(f)
    ds = spglib.get_symmetry_dataset((np.diag(shape).astype(float), positions, numbers), symprec=1e-3)
    improper = bool(any(np.linalg.det(R) < 0 for R in ds.rotations))
    return ds.international, ds.number, len(ds.rotations), improper


def poscar(F: np.ndarray, comment: str) -> str:
    shape = F.shape[1:]
    nb, o, f = sites(F)
    lines = [comment, "   1.0000000000000000"]
    lines += ["    " + "    ".join(f"{x:.16f}" for x in row) for row in np.diag(shape) * A]
    lines += ["  Nb   O    F", f"  {len(nb)}  {len(o)}  {len(f)}", "Direct"]
    lines += ["  " + "  ".join(f"{x:.16f}" for x in r) for r in nb + o + f]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stackings", nargs="+", help="per-plane stacking strings, e.g. 000111")
    parser.add_argument("--out", type=Path, required=True, help="directory to write stack_<s>/POSCAR.init into")
    args = parser.parse_args()

    for s, expected in [("000", ("P3_121", 54)), ("001", ("C2", 6))]:
        sg, _, nops, _ = space_group(fluorides(labels(s)))
        assert (sg, nops) == expected, f"3x3x3 stacking {s} gives {sg} with {nops} operations, expected {expected}"

    print(f"{'stacking':>14s}  {'cell':>7s}  boundaries  space group     operations  chirality")
    for s in args.stackings:
        if len(s) % 3 or set(s) - {"0", "1"}:
            raise ValueError(f"{s}: a stacking is a string of 0s and 1s whose length is a multiple of 3")
        F = fluorides(labels(s))
        cis, oof = obeys_rules(F)
        if not (cis and oof):
            raise AssertionError(f"{s} violates the ordering rules (cis {cis}, OOF {oof})")
        sg, number, nops, improper = space_group(F)
        boundaries = sum(s[k] != s[(k + 1) % len(s)] for k in range(len(s)))
        cell = f"3x3x{len(s)}"
        out = args.out / f"stack_{s}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "POSCAR.init").write_text(poscar(F, f"NbO2F {cell} stacking {s}"))
        print(f"{s:>14s}  {cell:>7s}  {boundaries:10d}  {sg:8s} (#{number:3d})  {nops:10d}  "
              f"{'achiral' if improper else 'chiral'}")


if __name__ == "__main__":
    main()
