"""Assemble the cluster-expansion training set from the DFT relaxations.

For each 6x6x6 configuration, reads the relaxed structure and energy from its
OUTCAR and writes a ``{path, energy_per_atom, "relaxed geometry"}`` entry
(the geometry ASE-jsonio-encoded). Reproduces the deposited
``data/ce/training_set_refitB_n96.json``.

    python build_training_set.py

Runnable to rebuild the training set from the DFT data; not part of the `make`
figure pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

from ase.io import read
from ase.io.jsonio import encode

HERE = Path(__file__).resolve().parent
DFT = HERE.parents[1] / "data" / "dft" / "6x6x6"
OUT = HERE.parents[1] / "data" / "ce" / "training_set_refitB_n96.json"


def main() -> None:
    rows = []
    for d in sorted(p for p in DFT.iterdir() if p.is_dir()):
        outcar = d / "OUTCAR"
        if not outcar.exists():
            continue
        atoms = read(outcar)
        rows.append({
            "path": f"./6x6x6/{d.name}/",
            "energy_per_atom": atoms.get_potential_energy() / len(atoms),
            "relaxed geometry": encode(atoms),
        })
    rows.sort(key=lambda r: r["path"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows))
    print(f"wrote {OUT}  ({len(rows)} structures)")


if __name__ == "__main__":
    main()
