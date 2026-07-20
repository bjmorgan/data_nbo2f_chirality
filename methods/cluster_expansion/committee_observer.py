"""Committee-uncertainty observer for active learning.

An mchammer observer that scores cluster-expansion committee disagreement on the
current Monte Carlo configuration and stashes the most uncertain structures for
DFT validation (the BUSS mechanism). Used by ``active_learning.py``; the
committee is built by ``build_posterior_committee.py``.
"""
from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
from ase import Atoms
from mchammer.observers.base_observer import BaseObserver


def load_committee(path: str | Path) -> np.ndarray:
    """Load a committee parameter matrix, shape ``(n_committee, n_params)``."""
    return np.load(path)["parameters"]


class CommitteeUncertaintyObserver(BaseObserver):
    """Evaluates CE committee disagreement on the current MC configuration.

    Computes the cluster vector once per observation, then evaluates all
    committee members via a single matrix multiply, returning the per-atom mean
    and standard deviation across the committee. Optionally stashes the top-K
    most uncertain occupation vectors (by committee std) for later DFT
    evaluation.

    The ClusterSpace is stored as a path and loaded lazily on first use, so the
    observer is picklable (required to retrieve the stash from ProcessPool
    workers).
    """

    def __init__(
        self,
        ce_path: str | Path,
        committee_params: np.ndarray,
        interval: int,
        stash_top_k: int = 20,
    ) -> None:
        super().__init__(return_type=dict, interval=interval, tag="CommitteeUncertainty")
        self._ce_path = str(ce_path)
        self._cs = None  # loaded lazily
        self._params = np.asarray(committee_params, dtype=np.float64)
        self._stash_top_k = stash_top_k
        self._stash: list[tuple[float, int, np.ndarray]] = []
        self._counter = 0

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cs"] = None
        return state

    def _get_cs(self):
        if self._cs is None:
            from icet import ClusterExpansion
            self._cs = ClusterExpansion.read(self._ce_path)._cluster_space
        return self._cs

    def get_observable(self, structure: Atoms) -> dict:
        cv = np.asarray(self._get_cs().get_cluster_vector(structure), dtype=np.float64)
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
        return [(std, occ) for std, _, occ in sorted(self._stash, reverse=True)]


class CommitteeObserverFactory:
    """Factory for ``pool.attach_observer_factory``.

    Each ProcessPool worker calls ``factory(replica)`` to construct a
    :class:`CommitteeUncertaintyObserver`, loading a fresh cluster space from
    ``ce_path`` (or the replica's own path) to avoid the mutated cluster space
    the calculator holds.
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

    def __call__(self, replica) -> CommitteeUncertaintyObserver:
        ce_path = self.ce_path or replica.cluster_expansion_path
        return CommitteeUncertaintyObserver(
            ce_path=ce_path,
            committee_params=load_committee(Path(self.committee_path)),
            interval=self.interval,
            stash_top_k=self.stash_top_k,
        )
