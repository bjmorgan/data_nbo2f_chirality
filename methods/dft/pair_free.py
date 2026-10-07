"""Rule-obeying arrangements with no side-by-side fluoride pairs.

With OOF order, a chain along axis a is fixed by its phase: the position (mod 3) of its
cations whose fluoride-free axis is a, one in every three. Two neighbouring parallel chains
have side-by-side fluoride pairs (two F one lattice spacing apart, on parallel bonds)
exactly when their phases are equal, in which case every fluoride on them is paired; any
other F-F pair at that separation is trans across a cation, which cis excludes. Cis at
every cation holds exactly when each cation is the fluoride-free cation of exactly one of
its three chains.

So the pair-free rule-obeying arrangements of a box of cations are the solutions of a
constraint problem with one phase per chain: at every cation exactly one of its three
chains has its phase there, and neighbouring parallel chains have different phases.
`solutions` enumerates them all, by depth-first search with arc consistency, on either
  - a periodic cell, each length a multiple of 3; or
  - an open window of the infinite lattice, each length at least 3, in which chains end
    at the faces and only neighbours inside the window are compared.
"""
from __future__ import annotations

import itertools

import numpy as np


def solutions(shape: tuple[int, int, int], periodic: bool = True, pair_free: bool = True):
    """Every rule-obeying arrangement of the box, as the cations' fluoride-free axes.

    With pair_free=False, the side-by-side condition is dropped, so every rule-obeying
    arrangement is returned.
    """
    L = shape
    if periodic:
        assert all(n % 3 == 0 for n in L), "OOF order needs every cell length to be a multiple of 3"
    else:
        assert all(n >= 3 for n in L), "a chain shorter than 3 does not fix its phase"
    # variables: one per chain; chain (a, transverse position) -> index
    chains = []
    for a in range(3):
        others = [d for d in range(3) if d != a]
        for p in itertools.product(range(L[others[0]]), range(L[others[1]])):
            chains.append((a, p))
    index = {c: n for n, c in enumerate(chains)}
    nvar = len(chains)

    def chain_of(a, c):
        return index[(a, tuple(c[d] for d in range(3) if d != a))]

    cations = list(itertools.product(*map(range, L)))
    # each cation: its three chains and the phase that makes it the fluoride-free cation of each
    triples = [tuple((chain_of(a, c), c[a] % 3) for a in range(3)) for c in cations]
    on_var = [[] for _ in range(nvar)]
    for t, tri in enumerate(triples):
        for v, _ in tri:
            on_var[v].append(t)
    # neighbouring parallel chains must differ
    neighbours = [set() for _ in range(nvar)]
    if pair_free:
        for (a, p), n in index.items():
            others = [d for d in range(3) if d != a]
            for k in range(2):
                for step in (1, -1):
                    q = list(p)
                    q[k] += step
                    if periodic:
                        q[k] %= L[others[k]]
                    elif not 0 <= q[k] < L[others[k]]:
                        continue
                    neighbours[n].add(index[(a, tuple(q))])

    def hits(d, need):
        return {int(x == need) for x in d}

    def propagate(dom, changed):
        """Arc consistency from the variables whose domains changed; False on a wipe-out."""
        queue = list(changed)
        while queue:
            v = queue.pop()
            if len(dom[v]) == 1:
                val = next(iter(dom[v]))
                for w in neighbours[v]:
                    if val in dom[w]:
                        dom[w] = dom[w] - {val}
                        if not dom[w]:
                            return False
                        queue.append(w)
            for t in on_var[v]:
                tri = triples[t]
                for k, (u, need) in enumerate(tri):
                    (a, na), (b, nb) = [tri[m] for m in range(3) if m != k]
                    sums = {x + y for x in hits(dom[a], na) for y in hits(dom[b], nb)}
                    keep = frozenset(val for val in dom[u] if 1 - (val == need) in sums)
                    if keep != dom[u]:
                        if not keep:
                            return False
                        dom[u] = keep
                        queue.append(u)
        return True

    def search(dom):
        free = [v for v in range(nvar) if len(dom[v]) > 1]
        if not free:
            yield dom
            return
        v = min(free, key=lambda u: (len(dom[u]), -len(on_var[u])))
        for val in sorted(dom[v]):
            new = list(dom)
            new[v] = frozenset({val})
            if propagate(new, [v]):
                yield from search(new)

    start = [frozenset({0, 1, 2})] * nvar
    if not propagate(start, range(nvar)):
        return
    for dom in search(start):
        labels = np.full(L, -1, dtype=int)
        for c, tri in zip(cations, triples):
            own = [a for a, (v, need) in enumerate(tri) if need in dom[v]]
            assert len(own) == 1
            labels[c] = own[0]
        yield labels


def tilings(blocks, shape: tuple[int, int, int]) -> set[bytes]:
    """The periodic 3x3x3 labellings `blocks`, repeated out to fill the box."""
    reps = [-(-n // 3) for n in shape]
    return {np.tile(B, reps)[: shape[0], : shape[1], : shape[2]].tobytes() for B in blocks}
