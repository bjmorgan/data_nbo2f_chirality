"""Lower bound on the cluster-expansion energy of rule-obeying arrangements, at any cell size.

For arrangements that obey both rules (cis at every Nb, OOF on every chain), on any
cell or on the infinite lattice, the P3_121 helix has the lowest energy in the fitted
cluster expansion (data/ce/paircut9_5_5_ardr_n96.ce), and is the only arrangement with
that energy, up to hand.

Windows. There is one window per cation: the 3x3x3 block of cations centred on it, with
the 108 anions bonded to them. On period-three chains the window's anions are fixed by
its 27 cation labels, and those labels form one of the 10,752 valid 3x3x3 labellings:
the twelve orbits (data/structures/orbit_representatives) in every orientation and
position.

Decomposition. Each cluster's energy is shared equally among the windows that contain
it. Every cluster of the model fits inside a window, which is checked, so the energy
per atom is exactly the mean of the window energies. It is therefore at least the lowest
window energy, with equality only if every window attains it.

Bound. Every P3_121 window has the same energy, and every other window lies above it.
Every P3_121 window has exactly one P3_121 neighbour on each side along each axis that
agrees with it on their overlap, and that neighbour continues the same helix. So an
arrangement whose windows are all P3_121 is a single homochiral helix.

Checks:
  - the cluster enumeration reproduces icet's cluster vector on a random 6x6x6
    arrangement that obeys neither rule;
  - window sums reproduce the icet energies of 3x3x6 stackings that are not
    three-periodic along z;
  - each orbit's window energies average to its icet energy;
  - every one of the 10,752 labellings obeys both rules.

Writes ce_window_bound.csv: for each orbit, its number of labellings, its CE energy and
the lowest and highest energy of its windows, all in meV/atom above P3_121. Prints the
checks, the table, the lowest window that is not P3_121, and the uniqueness check.

    python ce_window_bound.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from icet import ClusterExpansion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from build_stackings import fluorides, labels as stacking_labels, obeys_rules  # noqa: E402
from enumerate_3x3x6 import OPS, valid_blocks  # noqa: E402

CE = REPO / "data" / "ce" / "paircut9_5_5_ardr_n96.ce"
OUT = REPO / "build" / "ce" / "ce_window_bound.csv"
A_REF = 3.902                                  # lattice constant of the cluster expansion (A)
HELIX = 11                                     # the P3_121 orbit
TOL = 1e-9                                     # eV/atom


def canon_translation(sites):
    """Sites (doubled integer coordinates) translated so that the smallest lies in the origin cell."""
    s = sorted(map(tuple, sites))
    t = 2 * np.floor_divide(np.array(s[0]), 2)
    return tuple(sorted(tuple(np.array(x) - t) for x in s))


def axis_of(site) -> int:
    odd = [d for d in range(3) if site[d] % 2]
    assert len(odd) == 1, f"{site} is not an anion site"
    return odd[0]


def is_F(F: np.ndarray, site) -> bool:
    """Whether the anion at a doubled-coordinate site of a periodic F array is F."""
    a = axis_of(site)
    shape = F.shape[1:]
    idx = [(site[d] // 2) % shape[d] if d != a else ((site[d] - 1) // 2) % shape[d] for d in range(3)]
    return bool(F[(a,) + tuple(idx)])


def atoms_from_F(F: np.ndarray) -> Atoms:
    shape = F.shape[1:]
    sym, pos = [], []
    for c in itertools.product(*map(range, shape)):
        sym.append("Nb")
        pos.append(np.array(c, float))
        for a in range(3):
            r = np.array(c, float)
            r[a] += 0.5
            sym.append("F" if F[(a,) + c] else "O")
            pos.append(r)
    return Atoms(sym, positions=np.array(pos) * A_REF, cell=np.diag(shape) * A_REF, pbc=True)


def n_windows(shape) -> int:
    """Number of windows that contain a cluster, given as doubled-coordinate anion sites."""
    count = 1
    for d in range(3):
        lo, hi = -10 ** 6, 10 ** 6
        for s in shape:
            if s[d] % 2:            # along the site's own axis
                lo, hi = max(lo, s[d] - 5), min(hi, s[d] + 1)
            else:                   # perpendicular to it
                lo, hi = max(lo, s[d] - 4), min(hi, s[d])
        count *= len([o for o in range(lo, hi + 1) if o % 2 == 0])
    return count


def main() -> None:
    ce = ClusterExpansion.read(str(CE))
    cs = ce.get_cluster_space_copy()
    J = np.asarray(ce.parameters)

    # each orbit of the cluster space as its translation classes of site sets
    shapes = []
    for r in cs.as_list[1:]:
        rep = np.array(cs.get_coordinates_of_representative_cluster(r["orbit_index"])) / A_REF
        D = np.rint(2 * rep).astype(int)
        assert np.allclose(2 * rep, D, atol=1e-6)
        images = sorted({canon_translation((R @ D.T).T) for R in OPS})
        assert len(images) == r["multiplicity"], (r["orbit_index"], len(images), r["multiplicity"])
        shapes.append((r["index"], r["order"], images))

    # the cluster enumeration reproduces icet's cluster vector, which fixes the sign of the point function
    rng = np.random.default_rng(0)
    Fr = np.zeros((3, 6, 6, 6), dtype=bool)
    Fr.reshape(-1)[rng.choice(Fr.size, Fr.size // 3, replace=False)] = True
    cv_icet = cs.get_cluster_vector(atoms_from_F(Fr))
    theta = None
    for cand in ({False: -1.0, True: 1.0}, {False: 1.0, True: -1.0}):
        cv = [1.0]
        for _, _, shps in shapes:
            cv.append(np.mean([np.prod([cand[is_F(Fr, tuple(np.array(s) + 2 * np.array(t)))] for s in shp])
                               for shp in shps for t in itertools.product(range(6), repeat=3)]))
        if np.allclose(cv, cv_icet, atol=1e-10):
            theta = cand
    assert theta is not None, "the cluster enumeration does not reproduce icet's cluster vector"
    print("check: cluster vector reproduced on a random 6x6x6 arrangement")

    # the window at the origin: cations {0,1,2}^3 (doubled coordinates 0, 2, 4) and their 108 anions
    W0 = [p for a in range(3) for p in itertools.product(*[(-1, 1, 3, 5) if d == a else (0, 2, 4) for d in range(3)])]
    W0_index = {s: i for i, s in enumerate(W0)}
    assert len(W0) == 108

    clusters = {}                                  # order -> (site indices, weights) of the clusters in W0
    for cvi, order, shps in shapes:
        for shp in shps:
            n = n_windows(shp)
            assert n > 0, f"a cluster of orbit {cvi} fits in no window"
            for t in itertools.product(range(-4, 5), repeat=3):
                inst = [tuple(np.array(s) + 2 * np.array(t)) for s in shp]
                if all(s in W0_index for s in inst):
                    idx, wts = clusters.setdefault(order, ([], []))
                    idx.append([W0_index[s] for s in inst])
                    wts.append(J[cvi] / len(shps) / n)
    clusters = {o: (np.array(i), np.array(w)) for o, (i, w) in clusters.items()}
    print(f"check: every cluster of the {len(shapes)} cluster orbits fits in a window")

    def window_energies(blocks) -> np.ndarray:
        thetas = np.array([[theta[is_F(F, s)] for s in W0] for F in map(fluorides, blocks)])
        e = np.full(len(thetas), J[0])
        for idx, w in clusters.values():
            e += (np.prod(thetas[:, idx], axis=2) * w).sum(axis=1)
        return e

    def windows_of(L: np.ndarray) -> list[np.ndarray]:
        """The 3x3x3 label block of every window of a periodic arrangement."""
        return [L[np.ix_(*[(np.arange(3) + o[d]) % L.shape[d] for d in range(3)])]
                for o in itertools.product(*map(range, L.shape))]

    # window sums reproduce icet on arrangements that are not three-periodic along z
    err = max(abs(window_energies(windows_of(stacking_labels(s))).mean()
                  - ce.predict(atoms_from_F(fluorides(stacking_labels(s)))))
              for s in ("000000", "000111", "001011", "010101", "000001", "000101"))
    assert err < TOL, err
    print(f"check: window sums reproduce icet on 3x3x6 stackings to {err:.1e} eV/atom")

    # every window of a rule-obeying arrangement is one of the 10,752 valid labellings
    orbit_of = valid_blocks()
    blocks = [np.stack([np.frombuffer(p, dtype=np.int64).reshape(3, 3) for p in k], axis=2).astype(int)
              for k in orbit_of]
    orbit = np.array(list(orbit_of.values()))
    assert all(all(obeys_rules(fluorides(B))) for B in blocks)
    print(f"check: all {len(blocks)} valid 3x3x3 labellings obey both rules")

    E = window_energies(blocks)
    e_orbit = {k: ce.predict(atoms_from_F(fluorides(blocks[int(np.nonzero(orbit == k)[0][0])]))) for k in range(12)}
    err = max(abs(E[orbit == k].mean() - e_orbit[k]) for k in range(12))
    assert err < TOL, err
    print(f"check: each orbit's window energies average to its icet energy, to {err:.1e} eV/atom")

    e11 = e_orbit[HELIX]
    assert np.allclose(E[orbit == HELIX], e11, atol=TOL), "P3_121 windows differ in energy"
    rows = []
    for k in sorted(range(12), key=lambda k: e_orbit[k]):
        d = np.where(np.abs(E[orbit == k] - e11) < TOL, 0.0, 1e3 * (E[orbit == k] - e11))
        rows.append({"orbit": k, "labellings": int((orbit == k).sum()), "ce_meV_per_atom": 1e3 * (e_orbit[k] - e11),
                     "window_min_meV_per_atom": d.min(), "window_max_meV_per_atom": d.max()})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")
    print("\nwindow energies, meV/atom above P3_121:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    others = df[df.orbit != HELIX]
    low = others.loc[others.window_min_meV_per_atom.idxmin()]
    assert low.window_min_meV_per_atom > 0
    print(f"\nlowest window that is not P3_121: {low.window_min_meV_per_atom:+.3f} meV/atom (orbit {int(low.orbit)})")

    # uniqueness: P3_121 windows join up only as a single helix
    helix = [B for B, k in zip(blocks, orbit) if k == HELIX]
    for d in range(3):
        for b in helix:
            nxt = [c for c in helix if np.array_equal(np.take(c, [0, 1], axis=d), np.take(b, [1, 2], axis=d))]
            prv = [c for c in helix if np.array_equal(np.take(c, [1, 2], axis=d), np.take(b, [0, 1], axis=d))]
            assert len(nxt) == 1 and np.array_equal(nxt[0], np.roll(b, -1, axis=d))
            assert len(prv) == 1 and np.array_equal(prv[0], np.roll(b, 1, axis=d))
    print(f"check: each of the {len(helix)} P3_121 windows has exactly one P3_121 neighbour on each side "
          f"along each axis, which continues its helix")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
