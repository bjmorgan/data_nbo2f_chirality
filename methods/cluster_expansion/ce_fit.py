"""Shared cluster-expansion construction and ARDR fit.

Defines the reference lattice, the [9, 5, 5] cluster space, and the standardised
scikit-learn ARDR routine that produced the paper's model, and its posterior.
`fit_cluster_expansion`, `loocv`, `build_posterior_committee` and
`workflow/ce/ce_vs_dft.py` build on these, so the fit protocol is defined in one
place.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.jsonio import decode
from icet import ClusterSpace, StructureContainer
from icet.tools import map_structure_to_reference
from sklearn.linear_model import ARDRegression
from sklearn.preprocessing import StandardScaler

A_REF = 3.902                                          # cubic ReO3-type NbF3 lattice constant (Angstrom)
CUTOFFS = [9.0, 5.0, 5.0]                              # pair, triplet, quadruplet
CHEMICAL_SYMBOLS = [["Nb"], ["F", "O"], ["F", "O"], ["F", "O"]]   # Nb inert; anions mix O/F
THRESHOLD_LAMBDA = 10000                               # ARDR relevance threshold (from the hyperparameter scan)


def reference_lattice() -> Atoms:
    """Ideal cubic ReO3-type NbF3 primitive: Nb at the corner, anions at the edge centres."""
    return Atoms(
        "NbF3",
        scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        cell=[A_REF, A_REF, A_REF],
        pbc=True,
    )


def cluster_space() -> ClusterSpace:
    return ClusterSpace(reference_lattice(), cutoffs=CUTOFFS, chemical_symbols=CHEMICAL_SYMBOLS)


def fit_data(cs: ClusterSpace, training_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """Map each relaxed DFT structure onto the ideal lattice and return (A, y)."""
    data = json.loads(Path(training_json).read_text())
    sc = StructureContainer(cluster_space=cs)
    for entry in data:
        atoms = decode(entry["relaxed geometry"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ideal, _ = map_structure_to_reference(
                atoms, reference=cs.primitive_structure,
                inert_species=["Nb"], suppress_warnings=True,
            )
        sc.add_structure(structure=ideal, properties={"energy": entry["energy_per_atom"]})
    return sc.get_fit_data(key="energy")


def fit_ardr(A: np.ndarray, y: np.ndarray, threshold_lambda: float = THRESHOLD_LAMBDA) -> np.ndarray:
    """Standardised scikit-learn ARDR -- the routine that produced the deposited model.

    Columns are scaled to unit variance and the target to unit standard deviation
    before fitting, and the coefficients are scaled back afterwards.
    """
    A = A.copy()
    scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
    scaler.fit_transform(A)
    y_std = float(np.std(y))
    model = ARDRegression(threshold_lambda=threshold_lambda, fit_intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(A, y / y_std)
    return (model.coef_ * y_std) / scaler.scale_


def ardr_posterior(A: np.ndarray, y: np.ndarray,
                   threshold_lambda: float = THRESHOLD_LAMBDA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same standardised ARDR fit, with its Gaussian posterior in real units.

    Returns the ECIs (identical to `fit_ardr`), the mask of active ECIs, and the
    posterior covariance over the active ECIs, rescaled from the standardised fit.
    """
    A = A.copy()
    scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
    scaler.fit_transform(A)
    col_scale = scaler.scale_
    y_std = float(np.std(y))
    model = ARDRegression(threshold_lambda=threshold_lambda, fit_intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(A, y / y_std)
    coef = (model.coef_ * y_std) / col_scale
    active = model.lambda_ < threshold_lambda
    scale = y_std / col_scale[active]
    sigma = np.asarray(model.sigma_) * np.outer(scale, scale)
    if sigma.shape[0] != int(active.sum()):
        raise RuntimeError(f"posterior covariance {sigma.shape} does not match {int(active.sum())} active ECIs")
    return coef, active, sigma
