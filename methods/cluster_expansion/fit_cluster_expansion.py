"""Fit the cluster expansion used in the paper from the DFT training set.

Builds the [9, 5, 5] cluster space on the cubic ReO3-type NbF3 reference lattice
(Nb inert; the three anion sites mix O and F) and fits the effective cluster
interactions by automatic relevance determination (ARDR, threshold_lambda =
10000) on the 96-structure training set. The result matches the deposited
``data/ce/paircut9_5_5_ardr_n96.ce``.

The cutoffs and threshold_lambda are the values chosen by the hyperparameter
scan (``hyperparameter_scan.py``); this script applies that choice to produce the
final model.

    python fit_cluster_expansion.py

Runnable to verify or refit the cluster expansion; not part of the `make`
figure pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from icet import ClusterExpansion

from ce_fit import THRESHOLD_LAMBDA, cluster_space, fit_ardr, fit_data

HERE = Path(__file__).resolve().parent
TRAINING_SET = HERE.parents[1] / "data" / "ce" / "training_set_refitB_n96.json"
OUT = HERE.parents[1] / "data" / "ce" / "paircut9_5_5_ardr_n96.ce"


def main() -> None:
    cs = cluster_space()
    A, y = fit_data(cs, TRAINING_SET)
    parameters = fit_ardr(A, y, THRESHOLD_LAMBDA)

    ce = ClusterExpansion(cluster_space=cs, parameters=parameters)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ce.write(str(OUT))

    rmse = float(np.sqrt(np.mean((A @ parameters - y) ** 2)))
    n_nonzero = int((np.abs(parameters) > 1e-12).sum())
    print(f"wrote {OUT}")
    print(f"  {len(cs)} orbits, {n_nonzero} non-zero ECIs, "
          f"training RMSE {rmse * 1e3:.3f} meV/atom")


if __name__ == "__main__":
    main()
