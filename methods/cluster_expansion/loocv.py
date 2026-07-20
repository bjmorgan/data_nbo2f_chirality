"""Leave-one-out cross-validation of the cluster expansion.

Refits the model leaving out each training structure in turn and predicts the
held-out energy, at the production ARDR threshold (10000) and a sparser
alternative (250). Reports the cross-validation error in meV/atom.

    python loocv.py

Runnable to verify the model's predictive accuracy; not part of the `make`
figure pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ce_fit import THRESHOLD_LAMBDA, cluster_space, fit_ardr, fit_data

HERE = Path(__file__).resolve().parent
TRAINING_SET = HERE.parents[1] / "data" / "ce" / "training_set_refitB_n96.json"


def loocv(A: np.ndarray, y: np.ndarray, threshold_lambda: float) -> np.ndarray:
    """Per-structure held-out prediction error (meV/atom)."""
    n = len(y)
    err = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        params = fit_ardr(A[mask], y[mask], threshold_lambda)
        err[i] = (A[i] @ params - y[i]) * 1000.0
    return err


def main() -> None:
    cs = cluster_space()
    A, y = fit_data(cs, TRAINING_SET)
    print(f"{len(y)} structures, {A.shape[1]} orbits\n")
    for lam in (THRESHOLD_LAMBDA, 250):
        err = loocv(A, y, lam)
        tag = "   (production)" if lam == THRESHOLD_LAMBDA else ""
        print(f"lambda={lam:>6}: RMSE = {np.sqrt(np.mean(err ** 2)):.3f}   "
              f"MAE = {np.mean(np.abs(err)):.3f}   "
              f"max|e| = {np.max(np.abs(err)):.3f} meV/atom{tag}")


if __name__ == "__main__":
    main()
