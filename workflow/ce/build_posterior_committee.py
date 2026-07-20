"""Build a Bayesian-posterior committee for an ARDR cluster expansion.

Refits ARDR directly with scikit-learn at a chosen ``threshold_lambda``,
extracts the posterior mean and covariance, and samples ECI vectors from
the resulting Gaussian. The output format matches the bootstrap-committee
``.npz`` files (a ``(n_samples, n_orbits)`` array under key ``parameters``).

Usage:
    python build_posterior_committee.py --ce paircut9_5_5_ardr_n90.ce \
        --training-json training_set_refitA.json \
        --threshold-lambda 10000 \
        --n-samples 100 \
        --out posterior_committee_paircut9_5_5_n90.npz
"""
from __future__ import annotations

import argparse
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


def build_fit_data(cs, json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sc = StructureContainer(cluster_space=cs)
    with json_path.open() as f:
        data = json.load(f)
    for entry in data:
        atoms = decode(entry["relaxed geometry"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ideal, _ = map_structure_to_reference(
                atoms, reference=cs.primitive_structure,
                inert_species=["Nb"], suppress_warnings=True,
            )
        sc.add_structure(
            structure=ideal,
            properties={"energy": entry["energy_per_atom"]},
        )
    A, y = sc.get_fit_data(key="energy")
    return A, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce", type=str, required=True)
    parser.add_argument("--training-json", type=Path, required=True)
    parser.add_argument("--threshold-lambda", type=float, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ce_path = _resolve_ce_path(args.ce)
    ce = load_ce(ce_path)
    cs = ce._cluster_space
    print(f"Cluster space: {len(cs.orbit_list)} orbits, cutoffs {cs.cutoffs}")

    A_raw, y_raw = build_fit_data(cs, args.training_json)
    n_orbits = A_raw.shape[1]  # matches CE.parameters length (includes zerolet)
    print(f"Training set: {len(y_raw)} structures, fit matrix {A_raw.shape}")

    # Match trainstation's standardize=True protocol: column-std X to 1,
    # scale y so std(y)=1, fit, then rescale parameters and posterior covariance
    # back to original units. Note that in trainstation the column-std unscaling
    # is done by an opaque in-place reshape trick on the parameters array; we
    # do the same operation explicitly here so that ``coef`` matches the icet
    # production model and ``sigma`` is the corresponding posterior covariance.
    A = A_raw.copy()
    scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
    scaler.fit_transform(A)
    col_scale = scaler.scale_  # std=1 columns where original std==0
    y_std = float(np.std(y_raw))
    y = y_raw / y_std

    print(f"Fitting ARDR (threshold_lambda={args.threshold_lambda})...")
    model = ARDRegression(
        threshold_lambda=args.threshold_lambda,
        fit_intercept=False,
    )
    model.fit(A, y)

    # Coefficients fitted in standardized space relate to real ECIs by
    # w_real = (w_fit * y_std) / col_scale.
    coef = (model.coef_ * y_std) / col_scale

    # Posterior covariance transforms by the same per-coefficient factor, but
    # only over active rows/columns. Build the active rescale vector first.
    active_mask = model.lambda_ < args.threshold_lambda
    n_active = int(active_mask.sum())
    active_scale = (y_std / col_scale[active_mask])
    sigma_std = np.asarray(model.sigma_)
    sigma = sigma_std * np.outer(active_scale, active_scale)

    print(f"  {n_active} active ECIs (threshold lambda={args.threshold_lambda})")
    print(f"  coef[:5] = {coef[:5]}")
    print(f"  sigma shape: {sigma.shape}")
    if sigma.shape[0] != n_active:
        raise RuntimeError(
            f"sigma_ size {sigma.shape} does not match active count {n_active}"
        )

    rng = np.random.default_rng(args.seed)
    mean_active = coef[active_mask]
    samples_active = rng.multivariate_normal(
        mean=mean_active, cov=sigma, size=args.n_samples,
    )

    samples = np.zeros((args.n_samples, n_orbits))
    samples[:, active_mask] = samples_active

    print(f"Sampled {args.n_samples} draws from posterior.")
    diag = np.sqrt(np.diag(sigma)) * 1000  # meV/atom
    print(
        f"  per-coef posterior std (meV/atom), active: "
        f"min={diag.min():.3f}, max={diag.max():.3f}, "
        f"mean={diag.mean():.3f}"
    )

    np.savez(args.out, parameters=samples, mean=coef, active_mask=active_mask)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
