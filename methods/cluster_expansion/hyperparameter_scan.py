"""Hyperparameter scan for the cluster-expansion fit.

Scans regression methods and their hyperparameters by cross-validation
(trainstation) on the [9, 5, 5] cluster space, reporting CV-RMSE, training RMSE,
and the number of non-zero ECIs. This is the model-selection step: it documents
the choice of ARDR at threshold_lambda = 10000 that ``fit_cluster_expansion.py``
then applies to produce the final model.

    python hyperparameter_scan.py

Runnable to reproduce the model-selection scan; not part of the `make` figure
pipeline. (Selection uses trainstation cross-validation; the final fit uses the
standardised scikit-learn ARDR in ``ce_fit.py``.)
"""
from __future__ import annotations

from pathlib import Path

from trainstation import CrossValidationEstimator

from ce_fit import THRESHOLD_LAMBDA, cluster_space, fit_data

HERE = Path(__file__).resolve().parent
TRAINING_SET = HERE.parents[1] / "data" / "ce" / "training_set_refitB_n96.json"
N_SPLITS = 10
ARDR_LAMBDAS = [250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]


def cv_row(A, y, **kwargs):
    cve = CrossValidationEstimator((A, y), n_splits=N_SPLITS, **kwargs)
    cve.validate()
    cve.train()
    return cve.rmse_validation, cve.rmse_train, cve.n_nonzero_parameters


def main() -> None:
    cs = cluster_space()
    A, y = fit_data(cs, TRAINING_SET)
    print(f"{len(y)} structures, {len(cs)} orbits, {N_SPLITS}-fold CV\n")

    print("ARDR (threshold_lambda scan)")
    print(f"  {'lambda':>8} {'CV-RMSE':>9} {'train':>8} {'n_nz':>5}")
    best = None
    for lam in ARDR_LAMBDAS:
        cv, tr, nz = cv_row(A, y, fit_method="ardr", threshold_lambda=lam)
        mark = "  <- production" if lam == THRESHOLD_LAMBDA else ""
        print(f"  {lam:>8} {cv * 1e3:>9.3f} {tr * 1e3:>8.3f} {nz:>5}{mark}")
        if best is None or cv < best[1]:
            best = (lam, cv)
    print(f"  best CV-RMSE at lambda = {best[0]}: {best[1] * 1e3:.3f} meV/atom\n")

    print("RFE (best over n_features, for comparison)")
    rfe = min(
        (cv_row(A, y, fit_method="rfe", n_features=nf) + (nf,) for nf in range(4, len(cs) + 1, 2)),
        key=lambda r: r[0],
    )
    print(f"  n_features = {rfe[3]}: CV-RMSE {rfe[0] * 1e3:.3f} meV/atom, {rfe[2]} non-zero")


if __name__ == "__main__":
    main()
