"""Rule-obeying arrangements with no side-by-side fluoride pairs: only P3_121.

Side-by-side pairs are fluorides one lattice spacing apart on neighbouring parallel
chains. Two searches find which rule-obeying arrangements (cis at every Nb, OOF on every
chain) have none:

  - every rule-obeying labelling of the 3x3x6 cell (methods/dft/enumerate_3x3x6.py), with
    its side-by-side pairs counted from its fluorides;
  - the constraint search of methods/dft/pair_free.py, on periodic cells and on open
    windows of the infinite lattice. Run without the pair-free condition it must give the
    10,752 valid 3x3x3 labellings and the 3x3x6 labellings of the first search.

In both, the pair-free arrangements are exactly the 24 labellings of the P3_121 helix
(orbit 11: both hands, in every orientation and position), repeated out to fill the box.
The open 4x4x4 window makes this hold on any cell. In a pair-free rule-obeying
arrangement, periodic or not, every 4x4x4 window is part of one P3_121 tiling. Windows one
step apart share a 3x4x4 block. That block holds a whole 3x3x3 period, which fixes the
tiling, so the two windows are part of the same tiling. The whole arrangement is
therefore P3_121.

Writes two files:
  - pairs_3x3x6.csv: for each number of side-by-side pairs per 3x3x6 cell, how many
    rule-obeying labellings and symmetry classes have it;
  - pair_free.csv: for each box searched, its boundaries, and the number of pair-free
    rule-obeying arrangements and of P3_121 tilings among them.

    python pair_free_search.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from build_stackings import fluorides, space_group  # noqa: E402
from enumerate_3x3x6 import classify, enumerate_labellings, valid_blocks  # noqa: E402
from pair_free import solutions, tilings  # noqa: E402

OUT_PAIRS = REPO / "build" / "dft" / "pairs_3x3x6.csv"
OUT_SEARCH = REPO / "build" / "dft" / "pair_free.csv"
HELIX_ORBIT = 11
PERIODIC = [(3, 3, 3), (3, 3, 6), (6, 6, 6), (3, 6, 9), (6, 6, 12), (9, 9, 9), (12, 12, 12)]
OPEN = [(3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)]


def side_by_side_pairs(L: np.ndarray) -> int:
    """F pairs one lattice spacing apart on neighbouring parallel chains, per cell."""
    F = fluorides(L)
    return int(sum((F[a] & np.roll(F[a], 1, axis=d)).sum() for a in range(3) for d in range(3) if d != a))


def main() -> None:
    orbit_of = valid_blocks()
    blocks = {k: np.stack([np.frombuffer(p, dtype=np.int64).reshape(3, 3) for p in k], axis=2).astype(int)
              for k in orbit_of}
    helix = [B for k, B in blocks.items() if orbit_of[k] == HELIX_ORBIT]
    assert len(helix) == 24 and space_group(fluorides(helix[0]))[0] == "P3_121"

    # the constraint search, without the pair-free condition, reproduces the rule-obeying labellings
    t0 = time.time()
    assert {S.tobytes() for S in solutions((3, 3, 3), pair_free=False)} == {B.tobytes() for B in blocks.values()}
    labellings = enumerate_labellings(orbit_of)
    assert {S.tobytes() for S in solutions((3, 3, 6), pair_free=False)} == {L.astype(int).tobytes() for L in labellings}
    print(f"constraint search reproduces the {len(blocks)} valid 3x3x3 and {len(labellings)} 3x3x6 "
          f"rule-obeying labellings ({time.time() - t0:.0f} s)")

    # side-by-side pairs over every rule-obeying 3x3x6 labelling
    pairs = [side_by_side_pairs(L) for L in labellings]
    zero = {L.astype(int).tobytes() for L, n in zip(labellings, pairs) if n == 0}
    assert zero == tilings(helix, (3, 3, 6))
    rows = []
    for n in sorted(set(pairs)):
        group = [L for L, m in zip(labellings, pairs) if m == n]
        rows.append({"side_by_side_pairs": n, "labellings": len(group), "classes": len(classify(group))})
    pd.DataFrame(rows).to_csv(OUT_PAIRS, index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # pair-free rule-obeying arrangements on periodic cells and open windows
    rows = []
    for periodic, shapes in ((True, PERIODIC), (False, OPEN)):
        for shape in shapes:
            found = {S.tobytes() for S in solutions(shape, periodic=periodic)}
            helix_tilings = tilings(helix, shape)
            assert found == helix_tilings, shape
            rows.append({"box": "x".join(map(str, shape)), "boundaries": "periodic" if periodic else "open",
                         "pair_free": len(found), "p3121_tilings": len(found & helix_tilings)})
    df = pd.DataFrame(rows)
    OUT_SEARCH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SEARCH, index=False)
    print(df.to_string(index=False))
    print(f"wrote {OUT_PAIRS} and {OUT_SEARCH}")


if __name__ == "__main__":
    main()
