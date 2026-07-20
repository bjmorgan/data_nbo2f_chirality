"""Leave-one-out comparison of sparse vs dense ARDR models.

Caches cluster vectors once and refits the model leaving out each
training structure in turn, predicting the held-out structure with
both threshold_lambda values. Produces per-structure errors and
summary statistics.

Usage:
    python loocv_compare.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from ase.io.jsonio import decode
from icet import ClusterExpansion, StructureContainer
from icet.tools import map_structure_to_reference
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import StandardScaler

from ce_tools import _resolve_ce_path, load_ce


def fit_ardr_match_trainstation(
    A: np.ndarray, y: np.ndarray, threshold_lambda: float,
) -> np.ndarray:
    """Fit ARDR with trainstation's standardise/de-standardise protocol."""
    A = A.copy()
    scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
    scaler.fit_transform(A)
    col_scale = scaler.scale_
    y_std = float(np.std(y))
    y_scaled = y / y_std

    model = ARDRegression(
        threshold_lambda=threshold_lambda, fit_intercept=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(A, y_scaled)
    return (model.coef_ * y_std) / col_scale


def main() -> None:
    ce = load_ce(_resolve_ce_path("paircut9_5_5_ardr_n90.ce"))
    cs = ce._cluster_space
    prim = cs.primitive_structure

    data = json.load(open("training_set_refitB.json"))
    n = len(data)
    print(f"Computing cluster vectors for {n} training structures...")

    sc = StructureContainer(cluster_space=cs)
    for entry in data:
        atoms = decode(entry["relaxed geometry"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ideal, _ = map_structure_to_reference(
                atoms, reference=prim, inert_species=["Nb"],
                suppress_warnings=True,
            )
        sc.add_structure(
            structure=ideal,
            properties={"energy": entry["energy_per_atom"]},
        )
    A, y = sc.get_fit_data(key="energy")
    paths = [e["path"] for e in data]
    print(f"  cluster vectors cached, A shape {A.shape}")
    print()

    results = {"sparse": np.zeros(n), "dense": np.zeros(n)}
    print(f"Running LOOCV across all {n} structures...")
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        A_tr, y_tr = A[train_mask], y[train_mask]
        held_cv = A[i]
        held_y = y[i]

        for label, lam in [("sparse", 250), ("dense", 10000)]:
            params = fit_ardr_match_trainstation(A_tr, y_tr, lam)
            pred = held_cv @ params
            results[label][i] = (pred - held_y) * 1000

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} done")

    print()

    # Summary across all
    print("=== Full LOOCV across all 91 structures ===")
    for label in ("sparse", "dense"):
        arr = results[label]
        print(
            f"  {label:<8}: mean = {arr.mean():+.3f}, "
            f"RMS = {np.sqrt(np.mean(arr**2)):.3f}, "
            f"max abs = {np.max(np.abs(arr)):.3f} meV"
        )

    # Recent BUSS-round subset
    recent = ["config_82", "config_84", "config_85", "config_86", "config_87",
              "config_88", "config_89", "config_90", "config_91", "config_92",
              "config_93", "config_95"]
    recent_idx = [i for i, p in enumerate(paths) if any(c in p for c in recent)]
    print()
    print(f"=== Recent BUSS-round subset ({len(recent_idx)} structures) ===")
    for label in ("sparse", "dense"):
        arr = results[label][recent_idx]
        print(
            f"  {label:<8}: mean = {arr.mean():+.3f}, "
            f"RMS = {np.sqrt(np.mean(arr**2)):.3f}, "
            f"max abs = {np.max(np.abs(arr)):.3f} meV"
        )

    # Older subset
    other_idx = [i for i in range(n) if i not in recent_idx]
    print()
    print(f"=== Older subset ({len(other_idx)} structures) ===")
    for label in ("sparse", "dense"):
        arr = results[label][other_idx]
        print(
            f"  {label:<8}: mean = {arr.mean():+.3f}, "
            f"RMS = {np.sqrt(np.mean(arr**2)):.3f}, "
            f"max abs = {np.max(np.abs(arr)):.3f} meV"
        )

    # Per-structure print for largest errors
    print()
    print("=== Largest absolute errors ===")
    print(f"{'config':<28}  {'DFT_energy':>12}  {'sparse_err':>10}  {'dense_err':>10}")
    print("-" * 65)
    diff = np.abs(results["sparse"]) + np.abs(results["dense"])
    worst = np.argsort(diff)[::-1][:15]
    for i in worst:
        print(
            f"{paths[i][-26:]:<28}  {y[i]:>12.6f}  "
            f"{results['sparse'][i]:>+9.2f}  {results['dense'][i]:>+9.2f}"
        )

    # Save full results for plotting
    out = Path(__file__).parent / "loocv_results.npz"
    np.savez(
        out,
        sparse=results["sparse"], dense=results["dense"],
        dft_energies=y, paths=np.array(paths),
    )
    print(f"\nSaved per-structure results to {out}")


if __name__ == "__main__":
    main()
