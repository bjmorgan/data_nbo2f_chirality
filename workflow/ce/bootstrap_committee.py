"""Bootstrap committee of CE models for uncertainty quantification.

Generates a committee of CE parameterisations by bootstrap resampling
of the training data. Includes an mchammer BaseObserver subclass that
evaluates committee disagreement during MC/PT runs, and stashes the
most uncertain structures for targeted DFT validation (BUSS approach).

Usage (generate committee):
    python bootstrap_committee.py [--ce CE] [--training-json PATH]

Import the observer for PT runs:
    from bootstrap_committee import CommitteeUncertaintyObserver, load_committee
"""
from __future__ import annotations

import argparse
import heapq
import json
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.jsonio import decode
from icet import ClusterSpace, StructureContainer
from icet.tools import map_structure_to_reference
from mchammer.observers.base_observer import BaseObserver
from trainstation import Optimizer

from ce_tools import _resolve_ce_path, load_ce


def _build_structure_container(
    cs: ClusterSpace, json_path: Path,
) -> StructureContainer:
    """Build a StructureContainer from a training JSON."""
    sc = StructureContainer(cluster_space=cs)
    with json_path.open() as f:
        data = json.load(f)
    for entry in data:
        atoms_raw = decode(entry["relaxed geometry"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ideal, _ = map_structure_to_reference(
                atoms_raw, reference=cs.primitive_structure,
                inert_species=["Nb"], suppress_warnings=True,
            )
        sc.add_structure(
            structure=ideal, user_tag=entry["path"],
            properties={"energy": entry["energy_per_atom"]},
        )
    return sc


def fit_committee(
    cs: ClusterSpace,
    training_json: Path,
    n_committee: int = 100,
    weight_scale: float = 20.0,
    random_seed: int = 0,
) -> np.ndarray:
    """Bootstrap-resample training data and fit weighted RFE to each resample.

    Each resample draws N training structures with replacement from the
    full set, applies energy-dependent weights w = 1/(1 + gap/scale),
    then scans ``n_features`` for RFE and selects the value that
    minimises weighted RMSE on the out-of-bag samples.

    Args:
        cs: Cluster space defining the orbit structure.
        training_json: Path to the training JSON file.
        n_committee: Number of bootstrap resamples.
        weight_scale: Energy weighting scale in meV. Structures with
            gap >> scale are downweighted.
        random_seed: RNG seed for reproducibility.

    Returns:
        Parameter matrix of shape ``(n_committee, n_params)``.
        Each row is an ECI-times-multiplicity vector compatible with
        ``cluster_vector @ params = energy_per_atom``.
    """
    sc = _build_structure_container(cs, training_json)
    A, y = sc.get_fit_data(key="energy")
    n_train, n_params = A.shape
    rng = np.random.default_rng(random_seed)

    # Fixed weights from full dataset
    gaps = np.maximum((y - y.min()) * 1000, 0)  # meV
    weights = 1.0 / (1.0 + gaps / weight_scale)

    best_nf_list = []
    committee = np.zeros((n_committee, n_params))
    for i in range(n_committee):
        indices = rng.integers(0, n_train, size=n_train)
        w_i = weights[indices]
        A_i = A[indices] * w_i[:, None]
        y_i = y[indices] * w_i

        # Out-of-bag indices for model selection
        oob_mask = np.ones(n_train, dtype=bool)
        oob_mask[np.unique(indices)] = False

        best_score = np.inf
        best_params = None
        best_nf = None
        for nf in range(5, n_params + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt = Optimizer(
                    (A_i, y_i),
                    fit_method="rfe",
                    n_features=nf,
                    train_size=1.0,
                    check_condition=False,
                )
                opt.train()
            if oob_mask.any():
                pred_oob = A[oob_mask] @ opt.parameters
                w_oob = weights[oob_mask]
                score = np.sqrt(np.mean(w_oob * (pred_oob - y[oob_mask]) ** 2))
            else:
                pred = A[indices] @ opt.parameters
                score = np.sqrt(np.mean(w_i * (pred - y[indices]) ** 2))
            if score < best_score:
                best_score = score
                best_params = opt.parameters.copy()
                best_nf = nf

        committee[i] = best_params
        best_nf_list.append(best_nf)
        if (i + 1) % 10 == 0:
            nf_arr = np.array(best_nf_list)
            print(f"  {i + 1}/{n_committee} resamples fitted "
                  f"(n_features: mean={nf_arr.mean():.0f}, "
                  f"range={nf_arr.min()}-{nf_arr.max()})")
    return committee


def save_committee(path: Path, committee: np.ndarray) -> None:
    """Save a committee parameter matrix."""
    np.savez_compressed(path, parameters=committee)


def load_committee(path: Path) -> np.ndarray:
    """Load a committee parameter matrix."""
    return np.load(path)["parameters"]


class CommitteeUncertaintyObserver(BaseObserver):
    """Evaluates CE committee disagreement on the current MC configuration.

    Computes the cluster vector once per observation, then evaluates all
    committee members via a single matrix multiply. Returns the per-atom
    mean and standard deviation across the committee.

    Optionally stashes the top-K most uncertain occupation vectors
    (sorted by committee std) for later DFT evaluation.

    The ClusterSpace is stored as a path and loaded lazily on first use,
    so the observer is fully picklable (required for ``get_observers``
    to retrieve the stash from ProcessPool workers).

    Args:
        ce_path: Path to a ``.ce`` file. A fresh ClusterSpace is loaded
            from this on first use, avoiding the mutated cluster space
            held by the MC calculator.
        committee_params: Parameter matrix, shape ``(n_committee, n_params)``.
        interval: Observation interval in MC trial steps.
        stash_top_k: Retain this many highest-uncertainty structures.
    """

    def __init__(
        self,
        ce_path: str | Path,
        committee_params: np.ndarray,
        interval: int,
        stash_top_k: int = 20,
    ) -> None:
        super().__init__(
            return_type=dict, interval=interval,
            tag="CommitteeUncertainty",
        )
        self._ce_path = str(ce_path)
        self._cs: ClusterSpace | None = None  # loaded lazily
        self._params = np.asarray(committee_params, dtype=np.float64)
        self._stash_top_k = stash_top_k
        self._stash: list[tuple[float, int, np.ndarray]] = []
        self._counter = 0

    def __getstate__(self) -> dict:
        """Exclude the live ClusterSpace from pickle; reload on demand."""
        state = self.__dict__.copy()
        state["_cs"] = None
        return state

    def _get_cs(self) -> ClusterSpace:
        if self._cs is None:
            from icet import ClusterExpansion as _CE
            self._cs = _CE.read(self._ce_path)._cluster_space
        return self._cs

    def get_observable(self, structure: Atoms) -> dict:
        """Evaluate committee disagreement on ``structure``.

        Returns:
            ``{"committee_mean": float, "committee_std": float}``
            both in eV/atom.
        """
        cv = np.asarray(
            self._get_cs().get_cluster_vector(structure), dtype=np.float64,
        )
        predictions = self._params @ cv
        mean = float(predictions.mean())
        std = float(predictions.std())

        if self._stash_top_k > 0:
            occ = structure.numbers.copy()
            entry = (std, self._counter, occ)
            self._counter += 1
            if len(self._stash) < self._stash_top_k:
                heapq.heappush(self._stash, entry)
            elif std > self._stash[0][0]:
                heapq.heapreplace(self._stash, entry)

        return {"committee_mean": mean, "committee_std": std}

    @property
    def stashed_structures(self) -> list[tuple[float, np.ndarray]]:
        """Stashed ``(std, occupation_vector)`` pairs, highest std first."""
        return [
            (std, occ)
            for std, _, occ in sorted(self._stash, reverse=True)
        ]


class CommitteeObserverFactory:
    """Factory for ``pool.attach_observer_factory``.

    Each worker calls ``factory(replica)`` to construct a
    :class:`CommitteeUncertaintyObserver`. The factory loads a fresh CE
    from ``replica.cluster_expansion_path`` (set automatically by
    ProcessPool) to get an unmutated ClusterSpace — icet's calculator
    mutates its cluster space on construction, so the replica's own
    ClusterSpace has the wrong dimensionality.

    Args:
        committee_path: Path to the committee ``.npz`` file.
        interval: Observation interval in MC trial steps.
        stash_top_k: Per-replica stash size.
        ce_path: Explicit path to the CE ``.ce`` file. If provided,
            the observer loads its ClusterSpace from this path instead
            of from ``replica.cluster_expansion_path``. Use this to
            avoid icet's internal caching, which can return a mutated
            ClusterSpace when reading from the same file the calculator
            already loaded.
    """

    def __init__(
        self,
        committee_path: str | Path,
        interval: int,
        stash_top_k: int = 5,
        ce_path: str | Path | None = None,
    ) -> None:
        self.committee_path = str(committee_path)
        self.interval = interval
        self.stash_top_k = stash_top_k
        self.ce_path = str(ce_path) if ce_path is not None else None

    def __call__(self, replica) -> "CommitteeUncertaintyObserver":
        ce_path = self.ce_path or replica.cluster_expansion_path
        committee_params = load_committee(Path(self.committee_path))
        return CommitteeUncertaintyObserver(
            ce_path=ce_path,
            committee_params=committee_params,
            interval=self.interval,
            stash_top_k=self.stash_top_k,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a bootstrap committee of CE models.",
    )
    parser.add_argument("--ce", type=str, default=None)
    parser.add_argument(
        "--training-json", type=str,
        default=str(Path(__file__).parent / "training_set.json"),
    )
    parser.add_argument("--n-committee", type=int, default=100)
    parser.add_argument("--weight-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ce_path = _resolve_ce_path(args.ce)
    ce = load_ce(ce_path)
    cs = ce._cluster_space
    print(f"CE: {ce_path.name}, {len(cs)} parameters, "
          f"cutoffs {cs.cutoffs}")
    print(f"Training data: {args.training_json}")

    print(f"Fitting {args.n_committee} bootstrap weighted RFE models "
          f"(OOB-scanned n_features, weight_scale={args.weight_scale})...")
    committee = fit_committee(
        cs, Path(args.training_json),
        n_committee=args.n_committee,
        weight_scale=args.weight_scale,
        random_seed=args.seed,
    )

    n_nz_per = np.count_nonzero(committee, axis=1)
    print(f"Non-zero params per model: mean={n_nz_per.mean():.1f}, "
          f"min={n_nz_per.min()}, max={n_nz_per.max()}")

    out_path = Path(__file__).parent / f"committee_{ce_path.stem}.npz"
    save_committee(out_path, committee)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
