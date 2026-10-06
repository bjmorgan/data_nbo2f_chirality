"""The cluster expansion against DFT on the low-energy manifold, in and out of the training set.

Three sets. Each is referenced to P3_121 within its own set of DFT calculations,
because the sets use different cells and k-point meshes:
  - orbits: the twelve cis + OOF orbit representatives
    (data/structures/orbit_representatives) against the 3x3x3 set (data/dft/3x3x3;
    for orbit 11, the lower of its two settings), relative to orbit 11;
  - training: the 6x6x6 training structures within WINDOW meV/atom of P3121_tiled
    in DFT, relative to P3121_tiled. Each is also refitted without the structure,
    and the held-out prediction is the refit's energy of the structure relative
    to its energy of P3121_tiled;
  - 3x3x6: the fifteen achiral classes in data/dft/3x3x6, relative to the
    homochiral helix in the same cell (stack_000000).
A structure counts as in the training set if its cluster vector equals that of a
training structure; a tiling of a structure has the same cluster vector.

The CE energies are from the production model. Each uncertainty is the standard
deviation of the CE energy difference under the ARDR posterior of the fit,
sqrt(d^T Sigma d), where d is the difference of cluster vectors over the active
ECIs.

Writes ce_vs_dft_low_energy.csv: for each structure, its set, name, orbit (orbit
set only), whether it is in the training set, and the DFT energy, CE energy, CE
standard deviation and (training set only) held-out CE energy, all in meV/atom.
Prints the RMS difference and rank correlation for each set.

    python ce_vs_dft.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from icet import ClusterExpansion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "methods" / "cluster_expansion"))
from ce_fit import A_REF, ardr_posterior, cluster_space, fit_ardr, fit_data  # noqa: E402

DFT = REPO / "data" / "dft"
TRAINING_SET = REPO / "data" / "ce" / "training_set_refitB_n96.json"
CE = REPO / "data" / "ce" / "paircut9_5_5_ardr_n96.ce"
ORBIT_REPS = REPO / "data" / "structures" / "orbit_representatives"
DFT_OPS = REPO / "build" / "dft" / "dft_ops.csv"
OUT = REPO / "build" / "ce" / "ce_vs_dft_low_energy.csv"
WINDOW = 2.5                               # meV/atom above P3_121 in DFT, for the training set
REFERENCE = "P3121_tiled"


def energy_per_atom(outcar: Path) -> float:
    atoms = read(outcar)
    return atoms.get_potential_energy() / len(atoms)


def ideal_on_reference(path: Path):
    """An ideal structure, rescaled to the cluster expansion's lattice constant."""
    atoms = read(path, format="vasp")
    shape = np.round(atoms.cell.lengths() / A_REF).astype(int)
    atoms.set_cell(np.diag(shape) * A_REF, scale_atoms=True)
    return atoms


def main() -> None:
    cs = cluster_space()
    A, y = fit_data(cs, TRAINING_SET)
    names = [Path(e).name for e in pd.read_json(TRAINING_SET).path]
    coef, active, sigma = ardr_posterior(A, y)
    ce = ClusterExpansion.read(str(CE))
    assert np.allclose(coef, ce.parameters, rtol=1e-8, atol=1e-12), "refit does not reproduce the deposited ECIs"

    def in_training(cv) -> bool:
        return bool(np.any(np.all(np.isclose(A, cv, atol=1e-10), axis=1)))

    def sd(cv, cv_ref) -> float:
        d = (cv - cv_ref)[active]
        return 1e3 * float(np.sqrt(d @ sigma @ d))

    rows = []

    # orbits: 3x3x3 DFT set against the ideal representatives
    ops = pd.read_csv(DFT_OPS, dtype={"config": str})
    orb = ops[ops.cls == "orbit"]
    dft = {k: min(energy_per_atom(DFT / "3x3x3" / f"config_{c}" / "OUTCAR") for c in g.config)
           for k, g in orb.groupby("orbit")}
    cvs = {int(f.stem.split("_")[1]): cs.get_cluster_vector(ideal_on_reference(f))
           for f in ORBIT_REPS.glob("orbit_*.vasp")}
    for k in sorted(cvs):
        rows.append({"set": "orbits", "name": f"orbit {k}", "orbit": k, "in_training": in_training(cvs[k]),
                     "dft": 1e3 * (dft[k] - dft[11]), "ce": 1e3 * (cvs[k] - cvs[11]) @ coef,
                     "ce_sd": sd(cvs[k], cvs[11]), "ce_held_out": np.nan})

    # training: the 6x6x6 training structures in the low-energy window
    ref = names.index(REFERENCE)
    for i in np.argsort(y):
        e = 1e3 * (y[i] - y[ref])
        if i == ref or e > WINDOW:
            continue
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        held_out = 1e3 * (A[i] - A[ref]) @ fit_ardr(A[mask], y[mask])
        rows.append({"set": "training", "name": names[i], "orbit": np.nan, "in_training": True,
                     "dft": e, "ce": 1e3 * (A[i] - A[ref]) @ coef, "ce_sd": sd(A[i], A[ref]),
                     "ce_held_out": held_out})

    # 3x3x6: every achiral class against the helix in the same cell
    helix = DFT / "3x3x6" / "stack_000000"
    e_helix = energy_per_atom(helix / "OUTCAR")
    cv_helix = cs.get_cluster_vector(ideal_on_reference(helix / "POSCAR.init"))
    for d in sorted(p for p in [*(DFT / "3x3x6").glob("stack_*"), *(DFT / "3x3x6").glob("achiral_*")] if p != helix):
        cv = cs.get_cluster_vector(ideal_on_reference(d / "POSCAR.init"))
        rows.append({"set": "3x3x6", "name": d.name, "orbit": np.nan, "in_training": in_training(cv),
                     "dft": 1e3 * (energy_per_atom(d / "OUTCAR") - e_helix), "ce": 1e3 * (cv - cv_helix) @ coef,
                     "ce_sd": sd(cv, cv_helix), "ce_held_out": np.nan})

    df = pd.DataFrame(rows).rename(columns={c: f"{c}_meV_per_atom" for c in ("dft", "ce", "ce_sd", "ce_held_out")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")

    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    for (s, inside), g in df.groupby(["set", "in_training"], sort=False):
        g = g[~((g.set == "orbits") & (g.orbit == 11))]
        if g.empty:
            continue
        r = g.ce_meV_per_atom - g.dft_meV_per_atom
        rho = f"{g.dft_meV_per_atom.rank().corr(g.ce_meV_per_atom.rank()):.2f}" if len(g) > 2 else "-"
        line = (f"{s:8s} {'in' if inside else 'out of':>6s} training, n = {len(g):2d}: RMS {np.sqrt(np.mean(r ** 2)):.3f} meV/atom, "
                f"Spearman {rho}, median sd {g.ce_sd_meV_per_atom.median():.3f}")
        if s == "training":
            h = g.ce_held_out_meV_per_atom - g.dft_meV_per_atom
            line += f"; held out: RMS {np.sqrt(np.mean(h ** 2)):.3f}, Spearman {g.dft_meV_per_atom.rank().corr(g.ce_held_out_meV_per_atom.rank()):.2f}"
        print(line)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
