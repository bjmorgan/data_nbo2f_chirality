"""Arrangements with exact cis coordination at every Nb on the 3x3x6 cell, whether OOF or not.

Label each Nb by its fluoride-free axis. Exact cis at every Nb (two F, on different axes)
holds exactly when, along every chain on axis a, the cations not labelled a come in
adjacent pairs that share an F: every maximal (cyclic) run of labels other than a has
even length.
  - x- and y-chains (three sites) hold one or three cations labelled with the chain's
    axis; three makes the chain F-free.
  - z-chains (six sites) may also hold no z label, F-O-F-O-F-O in either of two phases,
    or two adjacent z labels, O-O-F-O-F-O.
OOF on every chain is the special case in which every chain has exactly one cation in
three labelled with its axis, spaced three apart.

Counting (`count`): a dynamic programme over the z-planes. Each z-chain carries one of
five states (no z label yet, or the parities of the run before its first z label and of
its current run), and each plane is weighted by the number of valid planes with its
pattern of z labels.

Energy (`LayerModel`): a layer holds the nine z-bond anions below a cation plane and the
eighteen in-plane anions of that plane. Exact cis makes a layer a (z-bond pattern below,
valid plane) pair, and fixes the pattern above: Z_above = [label != z] - Z_below. The
cluster expansion reaches at most two layers along z (pairs to 8.73 A; triplets and
quadruplets within 5 A), so the CE energy per atom of a cyclic stack of six layers is
    E = J0 + sum_k [ u(l_k) + v(l_k, l_{k+1}) + t(l_k, l_{k+2}) ],
with t made of pairs only. The tables are built from icet's supercell orbit list.

Search (`LayerModel.search`): every stack of six layers is enumerated. The first layer is
one representative of each class of layers under the in-plane symmetry (translations and
the signed x/y permutations), taken as the lowest-ranked layer of its stack, and the last
layer follows from closing the stack.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from icet import ClusterExpansion

from build_stackings import fluorides as oof_fluorides, labels as stacking_labels

HERE = Path(__file__).resolve().parent
CE = HERE.parents[1] / "data" / "ce" / "paircut9_5_5_ardr_n96.ce"
SHAPE = (3, 3, 6)
NL = SHAPE[2]
A_CE = 3.902                                   # lattice constant of the cluster expansion (A)
A_START = 3.9                                  # lattice constant of the ideal DFT starting structures (A)


# ---------------------------------------------------------------- constraint model

def valid_planes() -> np.ndarray:
    """3x3 planes (labels 0, 1, 2 = x, y, z) whose x- and y-chains each hold one or three own-axis labels."""
    out = []
    for flat_ in itertools.product(range(3), repeat=9):
        p = np.array(flat_).reshape(3, 3)            # p[i, j]: i along x, j along y
        if all((p[:, j] == 0).sum() in (1, 3) for j in range(3)) and all((p[i, :] == 1).sum() in (1, 3) for i in range(3)):
            out.append(p)
    return np.array(out)


def plane_in_plane_F(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x- and y-bond F of a valid plane: fx[i, j] is the bond from (i, j) to (i+1, j), fy[i, j] to (i, j+1)."""
    fx = np.zeros((3, 3), dtype=bool)
    fy = np.zeros((3, 3), dtype=bool)
    for j in range(3):
        own = [i for i in range(3) if p[i, j] == 0]
        if len(own) == 1:
            fx[(own[0] + 1) % 3, j] = True               # the other two cations pair up across this bond
    for i in range(3):
        own = [j for j in range(3) if p[i, j] == 1]
        if len(own) == 1:
            fy[i, (own[0] + 1) % 3] = True
    return fx, fy


def count(nz: int) -> tuple[int, int]:
    """Numbers of labellings and of F arrangements of the 3x3xnz cell with exact cis at every Nb."""
    planes = valid_planes()
    mult = defaultdict(int)
    for p in planes:
        mult[tuple((p == 2).ravel().astype(int))] += 1
    ns, nc = 5, 9
    digits = np.array(list(itertools.product(range(ns), repeat=nc)))[:, ::-1]
    pow5 = ns ** np.arange(nc)
    counts = np.zeros(ns ** nc)
    counts[0] = 1.0
    for k in range(nz):
        T = -np.ones((ns, 2), dtype=int)                 # T[state, is_z] -> next state (-1: invalid)
        T[0, 0], T[0, 1] = 0, 1 + 2 * (k % 2)
        for p1 in (0, 1):
            for pc in (0, 1):
                s = 1 + 2 * p1 + pc
                T[s, 0] = 1 + 2 * p1 + (pc ^ 1)
                if pc == 0:
                    T[s, 1] = 1 + 2 * p1
        new = np.zeros_like(counts)
        live = np.nonzero(counts)[0]
        for m, w in mult.items():
            nd = T[digits[live], np.array(m)[None, :]]
            ok = (nd >= 0).all(axis=1)
            np.add.at(new, nd[ok] @ pow5, counts[live[ok]] * w)
        counts = new
    d = digits
    col_ok = ((d == 0) & (nz % 2 == 0)) | ((d > 0) & (((d - 1) // 2 + (d - 1) % 2) % 2 == 0))
    valid = col_ok.all(axis=1)
    free = (d == 0).sum(axis=1)                          # z-chains with no z label: two phases each
    return int(round(counts[valid].sum())), int(round((counts[valid] * 2.0 ** free[valid]).sum()))


def check_cis(F: np.ndarray) -> bool:
    """Exactly two F at every Nb, on different axes."""
    for c in itertools.product(*map(range, SHAPE)):
        per_axis = sorted(int(F[(a,) + c]) + int(F[(a,) + tuple((np.array(c) - np.eye(3, dtype=int)[a]) % SHAPE)])
                          for a in range(3))
        if per_axis != [0, 1, 1]:
            return False
    return True


def oof_everywhere(F: np.ndarray) -> bool:
    """One F in every three sites, period three, on every chain."""
    for a in range(3):
        n = SHAPE[a]
        for rest in itertools.product(*[range(SHAPE[d]) for d in range(3) if d != a]):
            idx = list(rest)
            idx.insert(a, slice(None))
            seq = F[a][tuple(idx)]
            if not (n % 3 == 0 and all(seq[t] == seq[(t + 3) % n] for t in range(n)) and seq[:3].sum() == 1):
                return False
    return True


# ---------------------------------------------------------------- symmetry of the cell on F arrangements

OPS = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        R = np.zeros((3, 3), dtype=int)
        for r, (c, s) in enumerate(zip(perm, signs)):
            R[r, c] = s
        if abs(R[2, 2]) == 1:                            # z stays the long axis; improper operations included
            OPS.append(R)
BONDS = np.array([(a,) + c for a in range(3) for c in itertools.product(*map(range, SHAPE))])
MID = np.array([2 * np.array(b[1:]) + np.eye(3, dtype=int)[b[0]] for b in BONDS])   # doubled midpoints
_TWICE = 2 * np.array(SHAPE)
_BOND_INDEX = {tuple(m % _TWICE): n for n, m in enumerate(MID)}
PERMS = np.array([[_BOND_INDEX[tuple(m)] for m in (MID @ R.T + 2 * np.array(t)) % _TWICE]
                  for R in OPS for t in itertools.product(*map(range, SHAPE))])


def canonical(F: np.ndarray) -> bytes:
    """The lexicographically first image of an F arrangement under the symmetry of the cell."""
    v = np.array([F[tuple(b)] for b in BONDS])
    images = np.zeros_like(PERMS, dtype=bool)
    images[np.arange(len(PERMS))[:, None], PERMS] = v[None, :]
    packed = np.packbits(images, axis=1)
    return packed[np.lexsort(packed.T[::-1])[0]].tobytes()


def atoms_from_F(F: np.ndarray, a: float = A_CE) -> Atoms:
    """Ideal structure: Nb on the cation sites, F where F is True and O elsewhere on the bonds."""
    sym, pos = [], []
    for c in itertools.product(*map(range, SHAPE)):
        sym.append("Nb")
        pos.append(np.array(c, float))
        for ax in range(3):
            r = np.array(c, float)
            r[ax] += 0.5
            sym.append("F" if F[(ax,) + c] else "O")
            pos.append(r)
    return Atoms(sym, positions=np.array(pos) * a, cell=np.diag(SHAPE) * a, pbc=True)


def F_from_poscar(path: Path) -> np.ndarray:
    """F arrangement of an ideal 3x3x6 structure."""
    atoms = read(path, format="vasp")
    F = np.zeros((3,) + SHAPE, dtype=bool)
    for s, p in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions() * SHAPE):
        if s == "F":
            m = np.round(2 * p).astype(int) % _TWICE
            F[tuple(BONDS[_BOND_INDEX[tuple(m)]])] = True
    return F


# ---------------------------------------------------------------- layer model and search

_CATIONS = list(itertools.product(*map(range, SHAPE)))


def _slot(a: int, c) -> tuple[int, int]:
    """(layer, slot) of the anion on axis a at cation c; a z-bond belongs to the layer above it."""
    i, j, k = c
    if a == 2:
        return (k + 1) % NL, 3 * i + j                   # slots 0-8: z-bonds below the plane
    return k, 9 + 9 * a + 3 * i + j                     # slots 9-17: x-bonds; 18-26: y-bonds


def _layer_spins(F: np.ndarray) -> np.ndarray:
    """The 27-slot F pattern of each of the six layers of an F arrangement."""
    out = np.zeros((NL, 27), dtype=bool)
    for c in _CATIONS:
        for a in range(3):
            layer, s = _slot(a, c)
            out[layer, s] = F[(a,) + c]
    return out


class LayerModel:
    """The cluster expansion on the 3x3x6 cell, split into layer terms, for exact-cis stacks."""

    def __init__(self, ce: ClusterExpansion):
        self.ce = ce
        self.planes = valid_planes()
        self._layers()
        self._terms()
        self._tables()
        self.e_helix = ce.predict(atoms_from_F(oof_fluorides(stacking_labels("000000"))))

    def _layers(self) -> None:
        rows = []
        bits = 2 ** np.arange(8, -1, -1)
        for pi, p in enumerate(self.planes):
            nonz = (p != 2).ravel().astype(int)
            fx, fy = plane_in_plane_F(p)
            for zb in itertools.product((0, 1), repeat=9):
                zb = np.array(zb)
                za = nonz - zb
                if (za < 0).any() or (za > 1).any():
                    continue
                rows.append((pi, int(zb @ bits), int(za @ bits), np.concatenate([zb, fx.ravel(), fy.ravel()]).astype(bool)))
        self.pidx = np.array([r[0] for r in rows])
        self.zb = np.array([r[1] for r in rows])
        self.za = np.array([r[2] for r in rows])
        self.F = np.array([r[3] for r in rows])          # (layers, 27)
        self.S = np.where(self.F, 1.0, -1.0)

    def _terms(self) -> None:
        """Per-atom coefficient and (relative layer, slot) sites of every cluster with base layer 0."""
        cs = self.ce.get_cluster_space_copy()
        atoms = atoms_from_F(oof_fluorides(stacking_labels("000000")))
        sol = cs.orbit_list.get_supercell_orbit_list(atoms, cs.fractional_position_tolerance)
        slot_of_atom, coord_of_atom = {}, {}
        for n, c in enumerate(_CATIONS):
            for a in range(3):
                idx = 4 * n + 1 + a
                slot_of_atom[idx] = _slot(a, c)[1]
                coord_of_atom[idx] = c[2] + 1 if a == 2 else c[2]
        J = self.ce.parameters
        self.J0 = J[0]
        self.terms = []
        for oi, orbit in enumerate(sol.orbits):
            coef = J[oi + 1] / len(orbit.clusters)
            for cl in orbit.clusters:
                sites = [(ls.index, int(ls.unitcell_offset[2])) for ls in cl.lattice_sites]
                coords = [coord_of_atom[i] + NL * oz for i, oz in sites]
                base = min(coords)
                if base % NL:
                    continue                             # every cluster appears once per base layer
                self.terms.append((coef, tuple(sorted((cz - base, slot_of_atom[i]) for (i, _), cz in zip(sites, coords)))))
        span2 = [rel for _, rel in self.terms if max(dl for dl, _ in rel) == 2]
        assert max(dl for _, rel in self.terms for dl, _ in rel) == 2
        assert all(len(rel) == 2 and all(dl != 1 for dl, _ in rel) for rel in span2), \
            "clusters reaching two layers up must be pairs with no site in the middle layer"

    def _tables(self) -> None:
        S = self.S
        n = len(S)
        self.u = np.zeros(n)
        vA, vB, vc = [], [], []
        K2 = np.zeros((27, 27))
        for c, rel in self.terms:
            d = [dl for dl, _ in rel]
            if max(d) == 0:
                self.u += c * np.prod(S[:, [s for _, s in rel]], axis=1)
            elif max(d) == 1:
                vA.append(tuple(s for dl, s in rel if dl == 0))
                vB.append(tuple(s for dl, s in rel if dl == 1))
                vc.append(c)
            else:
                (_, a), (_, b) = rel
                K2[a, b] += c
        uA, uB = sorted(set(vA)), sorted(set(vB))
        Ai = {s: i for i, s in enumerate(uA)}
        Bi = {s: i for i, s in enumerate(uB)}
        Kv = np.zeros((len(uA), len(uB)))
        for a, b, c in zip(vA, vB, vc):
            Kv[Ai[a], Bi[b]] += c
        featA = np.stack([np.prod(S[:, list(s)], axis=1) for s in uA], axis=1)
        featB = np.stack([np.prod(S[:, list(s)], axis=1) for s in uB], axis=1)
        PA = featA @ Kv
        self.M2 = S @ K2
        groups = defaultdict(list)
        for i in range(n):
            groups[self.zb[i]].append(i)
        self.by_zb = {k: np.array(g) for k, g in groups.items()}
        self.pos_in_group = np.empty(n, dtype=np.int64)
        for g in self.by_zb.values():
            self.pos_in_group[g] = np.arange(len(g))
        self.succ_ptr = np.zeros(n + 1, dtype=np.int64)
        for a in range(n):
            self.succ_ptr[a + 1] = self.succ_ptr[a] + len(self.by_zb.get(self.za[a], ()))
        self.v_flat = np.empty(self.succ_ptr[-1])
        for a in range(n):
            g = self.by_zb.get(self.za[a])
            if g is not None:
                self.v_flat[self.succ_ptr[a]:self.succ_ptr[a + 1]] = featB[g] @ PA[a]

    def v(self, a, b):
        return self.v_flat[self.succ_ptr[a] + self.pos_in_group[b]]

    def t(self, a, b):
        return np.einsum("...i,...i->...", self.M2[a], self.S[b])

    def energy_of_F(self, F: np.ndarray) -> float:
        """CE energy per atom of any F arrangement on the cell, from the cluster terms."""
        L = np.where(_layer_spins(F), 1.0, -1.0)
        e = self.J0
        for c, rel in self.terms:
            for base in range(NL):
                p = 1.0
                for dl, s in rel:
                    p *= L[(base + dl) % NL, s]
                e += c * p
        return e

    def F_of(self, stack) -> np.ndarray:
        F = np.zeros((3,) + SHAPE, dtype=bool)
        for k, layer in enumerate(stack):
            f = self.F[layer]
            F[2][:, :, (k - 1) % NL] = f[:9].reshape(3, 3)     # the bond below plane k is the bond above plane k-1
            F[0][:, :, k] = f[9:18].reshape(3, 3)
            F[1][:, :, k] = f[18:27].reshape(3, 3)
        return F

    def _ranks(self) -> np.ndarray:
        coords = ([("z", 2 * i, 2 * j) for i, j in itertools.product(range(3), repeat=2)]
                  + [("xy", 2 * i + 1, 2 * j) for i, j in itertools.product(range(3), repeat=2)]
                  + [("xy", 2 * i, 2 * j + 1) for i, j in itertools.product(range(3), repeat=2)])
        index = {c: n for n, c in enumerate(coords)}
        perms = []
        for swap in (False, True):
            for sx, sy in itertools.product((1, -1), repeat=2):
                for tx, ty in itertools.product(range(3), repeat=2):
                    perm = np.empty(27, dtype=int)
                    for n, (kind, x, y) in enumerate(coords):
                        X, Y = (y, x) if swap else (x, y)
                        perm[n] = index[(kind, (sx * X + 2 * tx) % 6, (sy * Y + 2 * ty) % 6)]
                    perms.append(perm)
        return (self.F[:, np.array(perms)] * (2 ** np.arange(26, -1, -1))).sum(axis=2).min(axis=1)

    def search(self, window_meV: float, progress=None) -> tuple[np.ndarray, np.ndarray]:
        """Energies (meV/atom above the helix) and layers of every stack within the window."""
        rank = self._ranks()
        starts = [np.nonzero(rank == r)[0][0] for r in np.unique(rank)]
        thr = self.e_helix + window_meV / 1000.0
        u, v, t, za, zb = self.u, self.v, self.t, self.za, self.zb
        empty = np.array([], dtype=np.int64)
        kept_e, kept_s = [], []
        for si, l0 in enumerate(starts):
            r0 = rank[l0]
            G = {k: g[rank[g] >= r0] for k, g in self.by_zb.items()}
            H = defaultdict(list)
            for n in np.nonzero((za == zb[l0]) & (rank >= r0))[0]:
                H[zb[n]].append(n)
            H = {k: np.array(h) for k, h in H.items()}
            for l1 in G.get(za[l0], empty):
                for l2 in G.get(za[l1], empty):
                    e2 = self.J0 + u[l0] + u[l1] + u[l2] + v(l0, l1) + v(l1, l2) + t(l0, l2)
                    l3s = G.get(za[l2], empty)
                    pa3, pa4 = [], []
                    for z3 in np.unique(za[l3s]):
                        a = l3s[za[l3s] == z3]
                        b = G.get(z3, empty)
                        if len(b):
                            pa3.append(np.repeat(a, len(b)))
                            pa4.append(np.tile(b, len(a)))
                    if not pa3:
                        continue
                    L3, L4 = np.concatenate(pa3), np.concatenate(pa4)
                    q3, q4, q5 = [], [], []
                    for z4 in np.unique(za[L4]):
                        c = H.get(z4)
                        if c is None:
                            continue
                        m = za[L4] == z4
                        q3.append(np.repeat(L3[m], len(c)))
                        q4.append(np.repeat(L4[m], len(c)))
                        q5.append(np.tile(c, m.sum()))
                    if not q3:
                        continue
                    L3, L4, L5 = np.concatenate(q3), np.concatenate(q4), np.concatenate(q5)
                    e = (e2 + u[L3] + v(l2, L3) + t(np.full(len(L3), l1), L3)
                         + u[L4] + v(L3, L4) + t(np.full(len(L4), l2), L4)
                         + u[L5] + v(L4, L5) + t(L3, L5)
                         + v(L5, np.full(len(L5), l0)) + t(L4, np.full(len(L4), l0)) + t(L5, np.full(len(L5), l1)))
                    m = e <= thr
                    if m.any():
                        k = int(m.sum())
                        kept_e.append(e[m])
                        kept_s.append(np.stack([np.full(k, l0), np.full(k, l1), np.full(k, l2), L3[m], L4[m], L5[m]], axis=1))
            if progress:
                progress(si + 1, len(starts))
        E = np.concatenate(kept_e) if kept_e else np.zeros(0)
        return 1e3 * (E - self.e_helix), (np.concatenate(kept_s) if kept_s else np.zeros((0, NL), dtype=int))

    def kinds(self, stacks: np.ndarray) -> np.ndarray:
        """Kind of OOF break of each stack, read from its planes' z labels ("OOF" if none)."""
        zmask = (self.planes == 2).reshape(len(self.planes), 9)[self.pidx[stacks]]      # (n, 6, 9)
        plane_z = zmask.sum(axis=2)
        col_z = zmask.sum(axis=1)
        k = np.arange(NL)
        hi = np.where(zmask, k[None, :, None], -1).max(axis=1)
        lo = np.where(zmask, k[None, :, None], NL).min(axis=1)
        apart3 = (col_z == 2) & (hi - lo == 3)
        flags = [("F-free x/y chain", (plane_z == 1).any(axis=1)),          # a plane with one z label
                 ("period-2 z-chain", (col_z == 0).any(axis=1)),            # three F
                 ("F-free z-chain", (col_z == 6).any(axis=1)),
                 ("one-F z-chain", (col_z == 4).any(axis=1)),
                 ("two F one site apart", ((col_z == 2) & ~apart3).any(axis=1))]
        out = np.array([" + ".join(name for name, f in flags if f[i]) or "OOF" for i in range(len(stacks))], dtype=object)
        return out
