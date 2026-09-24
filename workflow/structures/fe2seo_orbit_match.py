"""Identify the iron/vacancy order of beta-Fe2SeO with one of the twelve orbits.

beta-Fe2SeO is an antiperovskite in which one third of the iron sites are
vacant. Its iron sites are the vertices of corner-sharing O-centred octahedra:
the same network as the anion sites of ReO3-type NbO2F, with O in place of Nb,
and iron and vacancies in place of O and F in the same 2:1 ratio. This script
maps the refined structure of beta-Fe2SeO onto the NbO2F labelling and compares
it with the twelve cis + OOF orbit representatives of the 3x3x3 cell.

Both materials pass through one routine. Each octahedron centre (Nb, or O in
beta-Fe2SeO) is placed on the simple-cubic network of centres, and each of its
six vertices is tested for the majority species (O, or Fe) within a quarter of
the cubic lattice constant. A vertex with no majority atom holds the minority
species (F, or a vacancy). For each structure the script then checks that

  - every centre has two minority vertices, in cis positions;
  - the minority sites repeat every three cubic cells;
  - every chain along a cubic axis holds one minority site per three.

It labels each centre by the axis with no minority vertex and compares the
labellings under the full symmetry group of the parent 3x3x3 cell (48 point
operations x 27 translations).

Writes build/structures/fe2seo_orbit_match.csv, one row per temperature of the
beta-Fe2SeO refinement.

    python fe2seo_orbit_match.py
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORBITS = REPO / "data" / "structures" / "orbit_representatives"
FE2SEO = REPO / "data" / "structures" / "beta_fe2seo"
OUT = REPO / "build" / "structures" / "fe2seo_orbit_match.csv"

N = 3        # period of the cis + OOF orbits along each cubic axis
TOL = 0.25   # a vertex holds the majority species if one lies within TOL x (cubic lattice constant)


def images(atoms: Atoms, symbol: str, n: int) -> np.ndarray:
    """Cartesian positions of `symbol` in the (2n+1)^3 block of cells around the origin cell."""
    frac = atoms.get_scaled_positions()[np.array(atoms.get_chemical_symbols()) == symbol]
    shifts = np.array(list(itertools.product(range(-n, n + 1), repeat=3)))
    return ((frac[:, None, :] + shifts[None, :, :]).reshape(-1, 3)) @ atoms.cell.array


def cubic_lattice(centres: np.ndarray, a0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the simple-cubic network of centres: an ideal lattice and each centre's integer site.

    `a0` is the lattice constant implied by the volume per centre. Nearest-neighbour
    vectors give the three cubic axes; each centre is then assigned its integer site,
    and the lattice (origin and the three cubic lattice vectors) is refined by least
    squares, so that the ideal sites absorb any overall strain of the network and
    leave only local displacements.

    Returns (integer sites, origin, lattice vectors as rows).
    """
    d = centres[:, None, :] - centres[None, :, :]
    r = np.linalg.norm(d, axis=2)
    vecs = d[(r > 0.8 * a0) & (r < 1.2 * a0)]
    axes: list[list[np.ndarray]] = []
    for v in vecs / np.linalg.norm(vecs, axis=1)[:, None]:
        for group in axes:
            if abs(v @ group[0]) > 0.9:
                group.append(v * np.sign(v @ group[0]))
                break
        else:
            axes.append([v])
    if len(axes) != 3:
        raise ValueError(f"expected 3 nearest-neighbour directions, found {len(axes)}")
    U, _, Vt = np.linalg.svd(np.array([np.mean(g, axis=0) for g in axes]))
    lattice, origin = (U @ Vt) * a0, centres[0]
    for _ in range(3):
        sites = np.rint((centres - origin) @ np.linalg.inv(lattice)).astype(int)
        design = np.hstack([sites, np.ones((len(sites), 1))])
        fit, *_ = np.linalg.lstsq(design, centres, rcond=None)
        lattice, origin = fit[:3], fit[3]
    return sites, origin, lattice


def minority_sites(atoms: Atoms, centre: str, majority: str) -> dict:
    """Map a structure onto the 3x3x3 torus of bonds and test the two ordering rules."""
    centres = images(atoms, centre, 1)
    a0 = (atoms.cell.volume / atoms.get_chemical_symbols().count(centre)) ** (1 / 3)
    sites, origin, lattice = cubic_lattice(centres, a0)
    a = float(np.mean(np.linalg.norm(lattice, axis=1)))
    ideal = origin + sites @ lattice
    centre_offsets = np.linalg.norm(centres - ideal, axis=1)
    if centre_offsets.max() > TOL * a:
        raise ValueError("centres do not sit on a simple-cubic network")
    majority_xyz = images(atoms, majority, 2)
    minority = {}                     # (axis, i, j, k) -> vertex holds the minority species
    periodic = True
    occupied_offsets, vacant_clearances = [], []
    for site, r in zip(sites, ideal):
        for axis, side in itertools.product(range(3), (1, -1)):
            dist = np.min(np.linalg.norm(majority_xyz - (r + side * 0.5 * lattice[axis]), axis=1))
            is_minority = bool(dist > TOL * a)
            (vacant_clearances if is_minority else occupied_offsets).append(dist)
            bond = site.copy()
            if side == -1:
                bond[axis] -= 1
            key = (axis,) + tuple(int(x) for x in bond % N)
            if minority.setdefault(key, is_minority) != is_minority:
                periodic = False
    cis = all(
        (per_axis := [minority[(ax,) + tuple(c)] + minority[(ax,) + tuple((np.array(c) - np.eye(3, dtype=int)[ax]) % N)]
                      for ax in range(3)]) and sum(per_axis) == 2 and max(per_axis) == 1
        for c in itertools.product(range(N), repeat=3)
    )
    chains = all(
        sum(minority[(ax,) + tuple(np.insert(np.array(p), ax, t))] for t in range(N)) == 1
        for ax in range(3) for p in itertools.product(range(N), repeat=2)
    )
    return {
        "minority": minority, "periodic": periodic, "cis": cis, "chains": chains,
        "a": a, "n_centres": len(centres), "max_centre_offset": float(centre_offsets.max()),
        "max_occupied_offset": max(occupied_offsets), "min_vacant_clearance": min(vacant_clearances),
    }


def labels(minority: dict) -> np.ndarray:
    """Label each centre by its axis with no minority vertex."""
    out = np.empty((N, N, N), dtype=int)
    for c in itertools.product(range(N), repeat=3):
        free = [ax for ax in range(3)
                if not minority[(ax,) + c]
                and not minority[(ax,) + tuple((np.array(c) - np.eye(3, dtype=int)[ax]) % N)]]
        out[c] = free[0]
    return out


def point_operations() -> list[np.ndarray]:
    """The 48 signed permutation matrices: the point group of the cubic parent."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), dtype=int)
            for row, (col, s) in enumerate(zip(perm, signs)):
                R[row, col] = s
            ops.append(R)
    return ops


POINT_OPS = point_operations()


def canonical(L: np.ndarray) -> bytes:
    """Canonical form of a label array under the point operations and the 27 translations."""
    best = None
    for R in POINT_OPS:
        axis_image = [int(np.nonzero(R[:, ax])[0][0]) for ax in range(3)]
        M = np.empty_like(L)
        for p in itertools.product(range(N), repeat=3):
            M[tuple(R @ np.array(p) % N)] = axis_image[L[p]]
        for t in itertools.product(range(N), repeat=3):
            key = np.roll(M, t, axis=(0, 1, 2)).tobytes()
            if best is None or key < best:
                best = key
    return best


def main() -> None:
    orbits = {}
    for f in sorted(ORBITS.glob("orbit_*.vasp")):
        tok = f.stem.split("_")                          # orbit_<idx>_sg<sg>_<HM...>_stab<n>
        m = minority_sites(read(f, format="vasp"), centre="Nb", majority="O")
        assert m["periodic"] and m["cis"] and m["chains"], f"{f.name} fails the ordering rules"
        orbits[canonical(labels(m["minority"]))] = (int(tok[1]), "_".join(tok[3:-1]))
    assert len(orbits) == 12, "orbit representatives are not twelve distinct orbits"

    rows = []
    for f in sorted(FE2SEO.glob("beta_Fe2SeO_*K.cif")):
        T = int(f.stem.split("_")[-1].rstrip("K"))
        m = minority_sites(read(f), centre="O", majority="Fe")
        ok = m["periodic"] and m["cis"] and m["chains"]
        idx, hm = orbits.get(canonical(labels(m["minority"])), (None, "")) if ok else (None, "")
        rows.append({
            "temperature_K": T,
            "cubic_a_A": round(m["a"], 4),
            "O_checked": m["n_centres"],
            "cis_everywhere": m["cis"],
            "period_three": m["periodic"],
            "one_vacancy_per_chain_period": m["chains"],
            "orbit": idx,
            "orbit_space_group": hm,
            "max_O_offset_A": round(m["max_centre_offset"], 3),
            "max_fe_offset_A": round(m["max_occupied_offset"], 3),
            "min_vacancy_clearance_A": round(m["min_vacant_clearance"], 3),
        })
        print(f"{T:4d} K: cis {m['cis']}, period three {m['periodic']}, chain rule {m['chains']} "
              f"-> orbit {idx} ({hm}); O within {m['max_centre_offset']:.2f} A and Fe within "
              f"{m['max_occupied_offset']:.2f} A of their ideal sites, nearest Fe to a vacancy "
              f"{m['min_vacant_clearance']:.2f} A")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
