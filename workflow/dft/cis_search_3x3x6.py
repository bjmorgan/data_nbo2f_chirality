"""Exhaustive cluster-expansion search over every exact-cis arrangement on the 3x3x6 cell.

Uses the layer model of methods/dft/cis_3x3x6.py, after checking that its layer terms
reproduce ClusterExpansion.predict on the helix, its stackings and random arrangements.
Every stack of six layers within KIND_WINDOW meV/atom of the homochiral helix is found.

Writes two files:
  - cis_classes_3x3x6.csv: every class within CLASS_WINDOW meV/atom of the helix, under
    the symmetry of the cell (improper operations included), in order of increasing CE
    energy: its energy, whether it is OOF on every chain, its kind of OOF break, its
    space group and whether it is chiral, and the name of its DFT run in data/dft/3x3x6
    if it has one;
  - cis_break_kinds_3x3x6.csv: for each kind of OOF break found within KIND_WINDOW, the
    lowest energy and the space group of that class.
Checks that the runs cis_nonoof_NN are the lowest classes that break OOF, in order.

    python cis_search_3x3x6.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from icet import ClusterExpansion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "dft"))
from build_stackings import fluorides, labels as stacking_labels, space_group  # noqa: E402
from cis_3x3x6 import CE, SHAPE, LayerModel, F_from_poscar, atoms_from_F, canonical, check_cis, oof_everywhere  # noqa: E402

RUNS = REPO / "data" / "dft" / "3x3x6"
OUT_CLASSES = REPO / "build" / "dft" / "cis_classes_3x3x6.csv"
OUT_KINDS = REPO / "build" / "dft" / "cis_break_kinds_3x3x6.csv"
CLASS_WINDOW = 2.5                                   # meV/atom above the helix
KIND_WINDOW = 6.0


def main() -> None:
    ce = ClusterExpansion.read(str(CE))
    model = LayerModel(ce)

    # the layer terms reproduce the cluster expansion, on rule-obeying and on random arrangements
    rng = np.random.default_rng(0)
    tests = [fluorides(stacking_labels(s)) for s in ("000000", "000111", "001011", "010101")]
    for _ in range(4):
        F = np.zeros((3,) + SHAPE, dtype=bool)
        F.reshape(-1)[rng.choice(F.size, F.size // 3, replace=False)] = True
        tests.append(F)
    err = max(abs(model.energy_of_F(F) - ce.predict(atoms_from_F(F))) for F in tests)
    assert err < 1e-9, err
    print(f"{len(model.F)} layers; layer terms reproduce the CE to {err:.1e} eV/atom")

    t0 = time.time()
    E, stacks = model.search(KIND_WINDOW, progress=lambda i, n: print(f"  {i}/{n} start layers, {time.time() - t0:.0f} s",
                                                                      flush=True) if i % 100 == 0 or i == n else None)
    kinds = model.kinds(stacks)
    print(f"{len(E)} stacks within {KIND_WINDOW} meV/atom of the helix")

    # DFT runs on this cell, by class
    run_of = {}
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and (d / "POSCAR.init").exists():
            run_of[canonical(F_from_poscar(d / "POSCAR.init"))] = d.name

    # classes within CLASS_WINDOW
    order = np.argsort(E, kind="stable")
    classes = {}
    for i in order[E[order] <= CLASS_WINDOW]:
        F = model.F_of(stacks[i])
        key = canonical(F)
        if key not in classes:
            assert check_cis(F)
            e_check = 1e3 * (ce.predict(atoms_from_F(F)) - model.e_helix)
            assert abs(e_check - E[i]) < 1e-6, (e_check, E[i])
            sg, _, _, improper = space_group(F)
            classes[key] = {"ce_meV_per_atom": E[i], "oof": oof_everywhere(F), "kind": kinds[i],
                            "space_group": sg, "chiral": not improper, "dft_run": run_of.get(key, "")}
    df = pd.DataFrame(classes.values())
    OUT_CLASSES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CLASSES, index_label="class", float_format="%.6f")

    non = df[~df.oof]
    named = [r for r in non.dft_run if r]
    expected = sorted(n for n in run_of.values() if n.startswith("cis_nonoof_"))
    assert named == expected and list(non.dft_run[: len(expected)]) == expected, (named, expected)

    # lowest of each kind of break within KIND_WINDOW
    rows = []
    for kind in sorted(set(kinds), key=lambda k: E[kinds == k].min()):
        idx = np.nonzero(kinds == kind)[0]
        i = idx[np.argmin(E[idx])]
        sg, _, _, improper = space_group(model.F_of(stacks[i]))
        rows.append({"kind": kind, "lowest_meV_per_atom": E[i], "space_group": sg, "chiral": not improper,
                     "stacks_in_window": len(idx)})
    kd = pd.DataFrame(rows)
    kd.to_csv(OUT_KINDS, index=False, float_format="%.6f")

    print(f"\n{len(df)} classes within {CLASS_WINDOW} meV/atom of the helix: {df.oof.sum()} OOF on every chain, "
          f"{len(non)} not")
    print(f"lowest classes that break OOF (meV/atom above the helix): "
          + ", ".join(f"{e:.3f}{' (' + r + ')' if r else ''}" for e, r in zip(non.ce_meV_per_atom[:6], non.dft_run[:6])))
    print("\nlowest of each kind of OOF break within the window:")
    print(kd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {OUT_CLASSES} and {OUT_KINDS}")


if __name__ == "__main__":
    main()
