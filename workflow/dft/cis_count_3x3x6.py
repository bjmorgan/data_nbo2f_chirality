"""Number of arrangements with exact cis coordination at every Nb on the 3x3x3 and 3x3x6 cells.

Counts the labellings and the F arrangements with exact cis at every Nb
(methods/dft/cis_3x3x6.py). On 3x3x3 every such arrangement is OOF on every chain, so
the count must equal the number of valid 3x3x3 labellings (the twelve orbits in every
orientation and position), which is checked. On 3x3x6 the number that are OOF on every
chain is taken from classes_3x3x6.csv.

Writes cis_count.csv: for each cell, the numbers of exact-cis labellings and F
arrangements, and of F arrangements that are OOF on every chain.

    python cis_count_3x3x6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from cis_3x3x6 import count  # noqa: E402
from enumerate_3x3x6 import valid_blocks  # noqa: E402

CLASSES = REPO / "build" / "dft" / "classes_3x3x6.csv"
OUT = REPO / "build" / "dft" / "cis_count.csv"


def main() -> None:
    lab3, arr3 = count(3)
    blocks = len(valid_blocks())
    assert lab3 == arr3 == blocks, (lab3, arr3, blocks)
    lab6, arr6 = count(6)
    oof6 = int(pd.read_csv(CLASSES).labellings.sum())
    df = pd.DataFrame([{"cell": "3x3x3", "cis_labellings": lab3, "cis_arrangements": arr3, "oof_arrangements": blocks},
                       {"cell": "3x3x6", "cis_labellings": lab6, "cis_arrangements": arr6, "oof_arrangements": oof6}])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"on 3x3x6, {arr6 - oof6} exact-cis arrangements are not OOF on every chain")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
