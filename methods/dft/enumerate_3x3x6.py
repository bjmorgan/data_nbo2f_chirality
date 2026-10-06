"""Enumerate the arrangements on the 3x3x6 cell that obey both ordering rules.

A labelling of the 3x3x6 cell (each Nb labelled by its fluoride-free axis) obeys
cis + OOF exactly if and only if every window of three consecutive (001) planes,
repeated periodically, is a valid 3x3x3 labelling:
  - cis at a cation involves only its own plane and the planes on either side;
  - a z-chain is period three along z if and only if every three consecutive sites
    hold exactly one z label, which each window guarantees;
  - x- and y-chains lie within one plane, whose period in x and y is the cell's.
So the arrangements are the cyclic sequences of six planes in which every three
consecutive planes form one of the 10,752 valid 3x3x3 labellings: the twelve orbit
representatives (data/structures/orbit_representatives) in every orientation and
position.

Prints the number of labellings and their classes up to the symmetry of the cell
(including improper operations, so the two hands of a chiral class count once),
with each achiral class's space group, the orbits of its six three-plane windows,
and its cluster-expansion energy relative to the homochiral P3_121 helix. The
stackings made by build_stackings.py (000000, 000111, 001011, 010101) are marked.

With --out DIR, writes DIR/achiral_NN/POSCAR.init for each achiral class that is
not one of those stackings, numbered in order of increasing cluster-expansion
energy. Each representative is the lexicographically first labelling of its class.

    python enumerate_3x3x6.py [--out DIR]
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from icet import ClusterExpansion

from build_stackings import fluorides, labels as stacking_labels, obeys_rules, poscar, space_group

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORBITS = REPO / "data" / "structures" / "orbit_representatives"
CE = REPO / "data" / "ce" / "paircut9_5_5_ardr_n96.ce"
SHAPE = (3, 3, 6)
A_CE = 3.902                                   # lattice constant of the cluster-expansion reference

OPS = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        R = np.zeros((3, 3), dtype=int)
        for r, (c, s) in enumerate(zip(perm, signs)):
            R[r, c] = s
        OPS.append(R)


def axis_image(R):
    return np.array([int(np.nonzero(R[:, ax])[0][0]) for ax in range(3)])


def transform3(R, L):
    """A signed permutation R (about a cation) applied to a periodic 3x3x3 labelling."""
    out = np.empty_like(L)
    img = axis_image(R)
    for p in itertools.product(range(3), repeat=3):
        out[tuple((R @ np.array(p)) % 3)] = img[L[p]]
    return out


def labels_from_poscar(path):
    a = read(path, format="vasp")
    frac = a.get_scaled_positions() * 3
    F = np.zeros((3, 3, 3, 3), dtype=bool)
    for p, sym in zip(frac, a.get_chemical_symbols()):
        if sym == "Nb":
            continue
        ax = int(np.argmax(np.abs(p - np.round(p))))
        b = np.round(p).astype(int)
        b[ax] = int(np.floor(p[ax]))
        F[(ax,) + tuple(b % 3)] = sym == "F"
    L = np.empty((3, 3, 3), dtype=int)
    for c in itertools.product(range(3), repeat=3):
        free = [ax for ax in range(3)
                if not F[(ax,) + c] and not F[(ax,) + tuple((np.array(c) - np.eye(3, dtype=int)[ax]) % 3)]]
        L[c] = free[0]
    assert np.array_equal(fluorides(L), F)
    return L


def valid_blocks():
    """The 10,752 valid 3x3x3 labellings as plane triples, and the orbit of each."""
    orbit_of = {}
    for f in sorted(ORBITS.glob("orbit_*.vasp")):
        k = int(f.stem.split("_")[1])
        L0 = labels_from_poscar(f)
        for R in OPS:
            M = transform3(R, L0)
            for t in itertools.product(range(3), repeat=3):
                B = np.roll(M, t, axis=(0, 1, 2))
                orbit_of.setdefault(tuple(B[:, :, z].tobytes() for z in range(3)), k)
    assert len(orbit_of) == 10752
    return orbit_of


# the symmetry operations of the 3x3x6 torus that keep z as the long axis, each as an
# index permutation with an axis relabelling, so all images of a labelling come at once
CELL_OPS = [R for R in OPS if abs(R[2, 2]) == 1]
SITES = np.array(list(itertools.product(*map(range, SHAPE))))
PERMS, AXMAPS = [], []
for R in CELL_OPS:
    for t in itertools.product(*map(range, SHAPE)):
        dest = np.ravel_multi_index(tuple(((SITES @ R.T + np.array(t)) % SHAPE).T), SHAPE)
        perm = np.empty(len(SITES), dtype=int)
        perm[dest] = np.arange(len(SITES))
        PERMS.append(perm)
        AXMAPS.append(axis_image(R))
PERMS, AXMAPS = np.array(PERMS), np.array(AXMAPS)


def canonical(L: np.ndarray) -> np.ndarray:
    """The lexicographically first image of L under the symmetry of the cell."""
    images = np.take_along_axis(AXMAPS, L.ravel()[PERMS], axis=1)
    return images[np.lexsort(images.T[::-1])[0]].reshape(SHAPE)


def atoms_for_ce(L):
    F = fluorides(L)
    sym, pos = [], []
    for c in itertools.product(*map(range, L.shape)):
        sym.append("Nb")
        pos.append(np.array(c, float))
        for a in range(3):
            r = np.array(c, float)
            r[a] += 0.5
            sym.append("F" if F[(a,) + c] else "O")
            pos.append(r)
    return Atoms(sym, positions=np.array(pos) * A_CE, cell=np.diag(L.shape) * A_CE, pbc=True)


def enumerate_labellings(orbit_of) -> list[np.ndarray]:
    """Every labelling of the 3x3x6 cell whose six three-plane windows are all valid blocks."""
    nxt = defaultdict(list)
    for p0, p1, p2 in orbit_of:
        nxt[(p0, p1)].append(p2)
    solutions = []
    for p0, p1, p2 in orbit_of:
        for p3 in nxt[(p1, p2)]:
            for p4 in nxt[(p2, p3)]:
                for p5 in nxt[(p3, p4)]:
                    if (p4, p5, p0) in orbit_of and (p5, p0, p1) in orbit_of:
                        solutions.append(np.stack([np.frombuffer(p, dtype=np.int64).reshape(3, 3)
                                                   for p in (p0, p1, p2, p3, p4, p5)], axis=2))
    return solutions


def classify(solutions) -> dict[bytes, tuple[np.ndarray, int]]:
    """Classes under the symmetry of the cell: canonical labelling and number of labellings in each."""
    classes = {}
    for L in solutions:
        C = canonical(L)
        key = C.tobytes()
        if key in classes:
            classes[key] = (classes[key][0], classes[key][1] + 1)
        else:
            classes[key] = (C.astype(int), 1)
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write achiral_NN/POSCAR.init here")
    args = parser.parse_args()

    orbit_of = valid_blocks()
    solutions = enumerate_labellings(orbit_of)
    classes = {key: L for key, (L, _) in classify(solutions).items()}
    stackings = {canonical(stacking_labels(s)).tobytes(): s for s in ("000000", "000111", "001011", "010101")}

    ce = ClusterExpansion.read(str(CE))
    e_helix = ce.predict(atoms_for_ce(stacking_labels("000000")))
    achiral = []
    for key, L in classes.items():
        F = fluorides(L)
        assert all(obeys_rules(F))
        sg, _, _, improper = space_group(F)
        if improper:
            windows = sorted(orbit_of[tuple(L[:, :, (z + d) % 6].tobytes() for d in range(3))] for z in range(6))
            achiral.append((1e3 * (ce.predict(atoms_for_ce(L)) - e_helix), sg, windows, stackings.get(key, ""), L))
    achiral.sort(key=lambda r: r[0])

    print(f"{len(solutions)} labellings of the 3x3x6 cell obey both rules; {len(classes)} classes, "
          f"{len(achiral)} of them achiral")
    print("achiral classes (CE energy above the homochiral helix, meV/atom):")
    n = 0
    for e, sg, windows, stack, L in achiral:
        name = f"stack_{stack}" if stack else f"achiral_{(n := n + 1):02d}"
        print(f"  {name:13s} {sg:5s} {e:6.3f}  window orbits {' '.join(map(str, windows))}")
        if args.out and not stack:
            d = args.out / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "POSCAR.init").write_text(poscar(fluorides(L), f"NbO2F 3x3x6 {name} ({sg})"))


if __name__ == "__main__":
    main()
