# Cluster-expansion workflow

How the cluster expansion the REWL runs sample was built: the fit itself, and
the active-learning loop that grew its DFT training set. This is methods
documentation for understanding and reuse — the scripts in
`methods/cluster_expansion/` are runnable to verify or refit, but they are not
part of the `make` figure pipeline, and the loop as a whole is not turn-key
(the DFT relaxations run externally on VASP).

## The model

The cluster expansion is defined on the cubic ReO3-type NbF3 reference lattice
with niobium held fixed and the three anion sites mixing oxygen and fluorine.
Cutoffs are 9 / 5 / 5 Angstrom (pair / triplet / quadruplet), giving 41 orbits.
The fitted model — `data/ce/paircut9_5_5_ardr_n96.ce`, 19 non-zero effective
cluster interactions — is trained on the 96-structure set
`data/ce/training_set_refitB_n96.json`.

## How it was fit

Two separate steps use two different tools:

1. **Hyperparameter selection.** The regression method (ARDR, versus RFE and
   LASSO variants), the cutoffs (9 / 5 / 5), and the ARDR relevance threshold
   (`threshold_lambda = 10000`) were chosen by cross-validation using
   **trainstation**.
2. **The final fit.** With those choices fixed, the effective cluster
   interactions were fit by automatic relevance determination — a standardised
   scikit-learn `ARDRegression` (columns scaled to unit variance, target scaled
   to unit standard deviation, no intercept) — **not** trainstation. This is
   what `methods/cluster_expansion/fit_cluster_expansion.py` does; it rebuilds
   the cluster space from scratch and reproduces the deposited model exactly.

The active-learning uncertainty committee (below) is drawn from the ARDR
posterior of that **same** final fit, so its posterior mean equals the fitted
model by construction. The fit and the committee share one routine.

The fit is ill-conditioned (the fit matrix condition number is about 2.85e16 —
the cluster vectors are nearly linearly dependent). Consequently the individual
ECIs are only weakly determined: trainstation and the scikit-learn routine, or
different linear-algebra backends, yield equivalent models whose ECIs differ by
up to ~25 meV/atom but whose energy predictions agree to ~0.01 meV/atom. Energy
predictions — what the Monte Carlo consumes — are robust; individual ECI values
are not uniquely pinned.

## The active-learning loop (BUSS)

The training set was grown iteratively. One round:

1. **Assemble the training set** from the completed DFT relaxations
   (`build_training_set.py`).
2. **Fit the cluster expansion** (`fit_cluster_expansion.py`).
3. **Build an uncertainty committee** by sampling ECIs from the Gaussian ARDR
   posterior of that fit (`build_posterior_committee.py`). An earlier round used
   a bootstrap committee instead; the ARDR posterior superseded it.
4. **Propose new structures** by running parallel tempering with a
   committee-disagreement observer, which stashes the highest-variance
   configurations and de-duplicates them against the current training set
   (`active_learning.py`).
5. **DFT-relax the proposed structures** (external: VASP), and fold the results
   back into the training set. Repeat.

Steps 1-4 are in `methods/cluster_expansion/`; step 5 is external. The
`data/ce/training_set_refitB_n96.json` set is the final round.

## Validation

`methods/cluster_expansion/loocv.py` runs leave-one-out cross-validation on the
final training set, reproducing the reported CV-RMSE.
