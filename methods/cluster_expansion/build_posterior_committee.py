"""Build the ARDR-posterior uncertainty committee.

Refits ARDR on the training set with the same standardised scikit-learn routine
that produced the model, extracts the Gaussian posterior mean and covariance over
the active ECIs, and samples committee members from it. The posterior mean equals
the fitted model, so the committee is centred on the production cluster expansion.

Writes an .npz with a (n_samples, n_orbits) ``parameters`` array (the committee),
``mean`` (the model ECIs), and ``active_mask``. This is the uncertainty source for
the active-learning loop (``active_learning.py``).

    python build_posterior_committee.py

Runnable to rebuild the committee; not part of the `make` figure pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ce_fit import ardr_posterior, cluster_space, fit_data

HERE = Path(__file__).resolve().parent
TRAINING_SET = HERE.parents[1] / "data" / "ce" / "training_set_refitB_n96.json"
OUT = HERE.parents[1] / "build" / "ce" / "posterior_committee.npz"
N_SAMPLES = 100
SEED = 0


def main() -> None:
    cs = cluster_space()
    A_raw, y_raw = fit_data(cs, TRAINING_SET)
    n_orbits = A_raw.shape[1]
    coef, active_mask, sigma = ardr_posterior(A_raw, y_raw)
    n_active = int(active_mask.sum())

    rng = np.random.default_rng(SEED)
    samples = np.zeros((N_SAMPLES, n_orbits))
    samples[:, active_mask] = rng.multivariate_normal(coef[active_mask], sigma, size=N_SAMPLES)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, parameters=samples, mean=coef, active_mask=active_mask)

    diag = np.sqrt(np.diag(sigma)) * 1000.0
    print(f"wrote {OUT}")
    print(f"  {n_active} active ECIs, {N_SAMPLES} committee members")
    print(f"  posterior std (meV/atom): min={diag.min():.3f}, max={diag.max():.3f}, mean={diag.mean():.3f}")


if __name__ == "__main__":
    main()
