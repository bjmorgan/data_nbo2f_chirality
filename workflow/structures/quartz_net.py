"""The net of cations linked through their occupied anion sites, compared with quartz.

In the P3_121 arrangement every cation has four O neighbours and every O bridges two
cations, so the cations joined through their O form a 4-connected net. This script
builds that net for

  - the ideal P3_121 arrangement (data/structures/orbit_representatives, orbit 11),
    with Nb linked through O;
  - the BaZnO2 and BaCoO2 refinements (data/structures/bazno2_type), with Zn or Co
    linked through O;
  - alpha-quartz as the reference for the qtz net, with Si linked through O, built from
    its space group (P3_221) and published cell and coordinates (Levien, Prewitt and
    Weidner, Am. Mineral. 65, 920 (1980)),

and computes for each its coordination sequence (the number of vertices 1 to SHELLS
bonds away, and their cumulative sum) and its vertex symbol (for each pair of edges at a
vertex, the length of the shortest rings through both, with their number, as in the
RCSR database). A node is linked to another when they share an O within CUTOFF of both, and
each node must have four links and each linking O two.

Writes quartz_net.csv: for each structure, the coordination sequence, its cumulative
sum and the vertex symbol, which are the same at every vertex (checked).

    python quartz_net.py
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.spacegroup import crystal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORBIT_11 = next((REPO / "data" / "structures" / "orbit_representatives").glob("orbit_11_*.vasp"))
CIFS = REPO / "data" / "structures" / "bazno2_type"
OUT = REPO / "build" / "structures" / "quartz_net.csv"
SHELLS = 10
CUTOFF = {"Si": 1.8, "Nb": 2.2, "Zn": 2.4, "Co": 2.4}     # node-O bond cutoff (A)


def alpha_quartz() -> Atoms:
    """Si on 3a (x, 0, 2/3) of P3_221 in its standard setting; the published O position shifted by the same 2/3 along c."""
    q = crystal(["Si", "O"], basis=[(0.4697, 0.0, 2 / 3), (0.4135, 0.2669, 0.1191 + 2 / 3)], spacegroup=154,
                cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120])
    assert q.get_chemical_formula() == "O6Si3"
    return q


def net(atoms: Atoms, node: str) -> dict[int, list[tuple[int, tuple[int, int, int]]]]:
    """Periodic graph: node index -> [(neighbour index, cell offset of the neighbour)]."""
    sym = np.array(atoms.get_chemical_symbols())
    nodes, ox = np.nonzero(sym == node)[0], np.nonzero(sym == "O")[0]
    cell = atoms.cell.array
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    links = {i: [] for i in range(len(nodes))}
    for o in ox:
        bonded = [(i, tuple(s)) for i, n in enumerate(nodes) for s in shifts
                  if np.linalg.norm(atoms.positions[n] + s @ cell - atoms.positions[o]) < CUTOFF[node]]
        assert len(bonded) == 2, f"an O is bonded to {len(bonded)} {node}"
        (i, si), (j, sj) = bonded
        links[i].append((j, tuple(np.subtract(sj, si))))
        links[j].append((i, tuple(np.subtract(si, sj))))
    assert all(len(v) == 4 for v in links.values()), f"not every {node} has four links"
    return links


def step(links, v):
    i, c = v
    return [(j, tuple(np.add(c, d))) for j, d in links[i]]


def coordination_sequence(links, start: int) -> list[int]:
    seen = {(start, (0, 0, 0))}
    shell, out = [(start, (0, 0, 0))], []
    for _ in range(SHELLS):
        nxt = []
        for v in shell:
            for w in step(links, v):
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        out.append(len(nxt))
        shell = nxt
    return out


def distance(links, u, w, limit: int) -> int:
    """Graph distance between two vertices, or limit + 1 if it is longer than limit."""
    seen, shell = {u}, [u]
    for d in range(1, limit + 1):
        nxt = []
        for v in shell:
            for x in step(links, v):
                if x == w:
                    return d
                if x not in seen:
                    seen.add(x)
                    nxt.append(x)
        shell = nxt
    return 0 if u == w else limit + 1


def vertex_symbol(links, start: int) -> str:
    """For each pair of edges at a vertex, the shortest rings through both: length and number.

    A ring is a circuit with no shortcut: no two of its vertices are closer in the net
    than along the circuit.
    """
    v0 = (start, (0, 0, 0))
    terms = []
    for a, b in itertools.combinations(step(links, v0), 2):
        # every shortest path from a to b that avoids v0, by breadth-first search
        dist, preds, shell = {a: 0}, {a: []}, [a]
        while b not in dist:
            nxt = set()
            for v in shell:
                for w in step(links, v):
                    if w == v0 or (w in dist and w not in nxt):
                        continue
                    if w not in dist:
                        dist[w], preds[w] = dist[v] + 1, []
                        nxt.add(w)
                    preds[w].append(v)
            shell = list(nxt)

        def paths(w):
            return [[a]] if w == a else [p + [w] for u in preds[w] for p in paths(u)]

        circuits = [[v0] + p for p in paths(b)]
        n = len(circuits[0])
        rings = sum(all(distance(links, c[i], c[k], n // 2) >= min(k - i, n - k + i)
                        for i, k in itertools.combinations(range(n), 2)) for c in circuits)
        terms.append((n, rings))
    terms.sort()
    return "·".join(f"{n}" if k == 1 else f"{n}({k})" for n, k in terms)


def describe(name: str, atoms: Atoms, node: str) -> dict:
    links = net(atoms, node)
    cs = {i: coordination_sequence(links, i) for i in links}
    vs = {i: vertex_symbol(links, i) for i in links}
    assert len({tuple(v) for v in cs.values()}) == 1 and len(set(vs.values())) == 1, f"{name}: vertices differ"
    seq, sym = cs[0], vs[0]
    print(f"{name:34s} {' '.join(map(str, seq))}  (cumulative {sum(seq)})  {sym}")
    return {"structure": name, "node": node, "vertices_per_cell": len(links),
            **{f"shell_{k + 1}": n for k, n in enumerate(seq)}, f"cumulative_{SHELLS}": sum(seq), "vertex_symbol": sym}


def main() -> None:
    rows = [describe("alpha-quartz (qtz reference)", alpha_quartz(), "Si"),
            describe("P3_121 arrangement (orbit 11)", read(ORBIT_11, format="vasp"), "Nb")]
    for f in sorted(CIFS.glob("COD*.cif")):
        atoms = read(f)
        b = next(s for s in ("Zn", "Co") if s in atoms.get_chemical_symbols())
        rows.append(describe(f"Ba{b}O2 (COD {f.stem.split('_')[0][3:]})", atoms, b))
    ref = [rows[0][k] for k in rows[0] if k.startswith("shell_")] + [rows[0]["vertex_symbol"]]
    same = [[r[k] for k in r if k.startswith("shell_")] + [r["vertex_symbol"]] == ref for r in rows]
    print(f"same coordination sequence and vertex symbol as quartz: {all(same)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
