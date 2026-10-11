"""Which stackings of the P3_121 helix on the 3x3x6 cell are achiral.

A stacking keeps the helix's labelling in each (001) plane of cations (0) or takes that
of its mirror image, P3_221 (1), so there are 2^6 = 64 stacking strings
(methods/dft/build_stackings.py). This script groups them into classes under the
symmetry of the cell (improper operations included, as in enumerate_3x3x6.py), finds
the space group of each stacking and whether it is chiral, and matches each class to its
row in classes_3x3x6.csv by its number of labellings, the space group of its canonical
labelling and its window orbits.

Writes stackings_3x3x6.csv: for each stacking, the number of exchanged planes, its class
(the row of classes_3x3x6.csv), the space group and whether it is chiral. Checks that
the achiral stackings are exactly those with three exchanged planes, and that their
classes are the three stack_* classes of the DFT set.

    python stackings_3x3x6.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from build_stackings import fluorides, labels, space_group  # noqa: E402
from enumerate_3x3x6 import canonical, classify, enumerate_labellings, valid_blocks  # noqa: E402

CLASSES = REPO / "build" / "dft" / "classes_3x3x6.csv"
OUT = REPO / "build" / "dft" / "stackings_3x3x6.csv"


def main() -> None:
    orbit_of = valid_blocks()
    sizes = {k: n for k, (_, n) in classify(enumerate_labellings(orbit_of)).items()}
    table = pd.read_csv(CLASSES).fillna({"dft_run": ""})

    rows = []
    for s in ("".join(b) for b in itertools.product("01", repeat=6)):
        L = labels(s)
        C = canonical(L)
        key = C.tobytes()
        sg, _, _, improper = space_group(fluorides(L))
        class_sg = space_group(fluorides(C))[0]           # as listed in classes_3x3x6.csv, for one hand
        windows = " ".join(map(str, sorted(orbit_of[tuple(L[:, :, (z + k) % 6].tobytes() for k in range(3))]
                                           for z in range(6))))
        match = table[(table.labellings == sizes[key]) & (table.space_group == class_sg) & (table.window_orbits == windows)]
        assert len(match) == 1, (s, sizes[key], class_sg, windows)
        rows.append({"stacking": s, "planes_exchanged": s.count("1"), "class": int(match["class"].iloc[0]),
                     "space_group": sg, "chiral": not improper, "dft_run": match.dft_run.iloc[0]})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = df.groupby("class").agg(stackings=("stacking", "size"), example=("stacking", "first"),
                                      planes_exchanged=("planes_exchanged", lambda v: sorted(set(v))),
                                      space_group=("space_group", "first"), chiral=("chiral", "first"),
                                      dft_run=("dft_run", "first"))
    print(summary.to_string())
    achiral = df[~df.chiral]
    print(f"\n{len(df)} stackings in {len(summary)} classes; {len(achiral)} achiral, in {achiral['class'].nunique()} classes")
    assert set(achiral.stacking) == {s for s in df.stacking if s.count("1") == 3}, "achiral != three planes exchanged"
    assert sorted(achiral.dft_run.unique()) == ["stack_000111", "stack_001011", "stack_010101"]
    print("the achiral stackings are exactly those with three exchanged planes, in the three stack_* classes")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
