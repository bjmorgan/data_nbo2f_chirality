"""Every arrangement on the 3x3x6 cell that obeys both ordering rules, grouped into symmetry classes.

Enumerates the labellings of the 3x3x6 cell whose six three-plane windows are all valid
3x3x3 blocks (methods/dft/enumerate_3x3x6.py) and groups them into classes under the
symmetry of the cell: the 16 point operations that keep z as the long axis, improper
ones included, and the 54 translations, so the two hands of a chiral class count once.

Writes classes_3x3x6.csv, one row per class in order of increasing cluster-expansion
energy: the number of labellings in the class, its space group and whether it is
chiral, the orbits of its six three-plane windows, its CE energy above the homochiral
helix (meV/atom), and the name of its DFT run in data/dft/3x3x6, where there is one (the
homochiral helix and the fifteen achiral classes). Prints the totals.

    python classes_3x3x6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from icet import ClusterExpansion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from build_stackings import fluorides, labels as stacking_labels, obeys_rules, space_group  # noqa: E402
from enumerate_3x3x6 import CE, atoms_for_ce, canonical, classify, enumerate_labellings, valid_blocks  # noqa: E402

OUT = REPO / "build" / "dft" / "classes_3x3x6.csv"
STACKINGS = ("000000", "000111", "001011", "010101")


def main() -> None:
    orbit_of = valid_blocks()
    solutions = enumerate_labellings(orbit_of)
    classes = classify(solutions)
    stacking_of = {canonical(stacking_labels(s)).tobytes(): s for s in STACKINGS}

    ce = ClusterExpansion.read(str(CE))
    e_helix = ce.predict(atoms_for_ce(stacking_labels("000000")))
    rows = []
    for key, (L, n) in classes.items():
        F = fluorides(L)
        assert all(obeys_rules(F))
        sg, _, _, improper = space_group(F)
        windows = sorted(orbit_of[tuple(L[:, :, (z + d) % 6].tobytes() for d in range(3))] for z in range(6))
        rows.append({"labellings": n, "space_group": sg, "chiral": not improper,
                     "window_orbits": " ".join(map(str, windows)),
                     "ce_meV_per_atom": 1e3 * (ce.predict(atoms_for_ce(L)) - e_helix),
                     "dft_run": f"stack_{stacking_of[key]}" if key in stacking_of else ""})
    df = pd.DataFrame(rows).sort_values("ce_meV_per_atom", kind="stable").reset_index(drop=True)

    # the achiral classes other than the stackings are numbered by CE energy, as in data/dft/3x3x6
    n = 0
    for i in df.index:
        if not df.at[i, "dft_run"] and not df.at[i, "chiral"]:
            n += 1
            df.at[i, "dft_run"] = f"achiral_{n:02d}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index_label="class", float_format="%.6f")

    print(f"{df.labellings.sum()} labellings of the 3x3x6 cell obey both rules; {len(df)} classes, "
          f"{(~df.chiral).sum()} of them achiral ({(~df.chiral).sum() - 3} besides the three stackings of the helix)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
