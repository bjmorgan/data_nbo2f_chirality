"""PT run with BUSS-style committee uncertainty tracking.

Runs parallel tempering on NbO2F at a configurable cell size with
per-replica CommitteeUncertaintyObservers attached via ProcessPool's
``attach_observer_factory``. After the run, retrieves worker-side
stashes via ``get_observers``, deduplicates against each other and
the existing training set, and writes diverse candidates for DFT
validation with chainorder diagnostics. Also writes the lowest-energy
final structure for active learning.

Usage:
    python pt_buss.py [--ce CE] [--n-sc 6] [--start random]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write
from ase.io.jsonio import decode
from icet.tools import map_structure_to_reference

from mchammer_pt import CanonicalParallelTempering
from mchammer_moves import CustomCanonicalEnsemble, PairSwap
from mchammer.calculators import ClusterExpansionCalculator

from chain_geometry import RowShift, MotifShift
from bootstrap_committee import (
    CommitteeObserverFactory,
    CommitteeUncertaintyObserver,
    load_committee,
)
from ce_tools import (
    TEMPERATURES_K,
    _TRANS_PAIRS,
    _f_mask_from_atoms,
    _resolve_ce_path,
    atoms_from_f_mask,
    atoms_from_f_mask_stable,
    load_ce,
    load_orbit_rep,
    make_random,
    nb_anion_neighbours,
    tile_f_mask,
)
from chainorder import SublatticeOccupation, order_params

MIN_HAMMING = 20
MAX_CANDIDATES = 8


# ---------------------------------------------------------------------------
# Helpers parameterised by n_sc
# ---------------------------------------------------------------------------

def _block_size(n_sc: int) -> int:
    """Two sweeps over the anion sublattice."""
    return 2 * 3 * n_sc**3


def _build_ordered_start(n_sc: int) -> Atoms | None:
    """Tiled P3_121 ground state, or None if n_sc is not a multiple of 3."""
    if n_sc % 3 != 0:
        return None
    atoms_3 = load_orbit_rep(11)
    if n_sc == 3:
        return atoms_from_f_mask_stable(3, _f_mask_from_atoms(atoms_3, 3))
    tile = n_sc // 3
    f_mask_3 = _f_mask_from_atoms(atoms_3, 3)
    f_mask_big = tile_f_mask(f_mask_3, 3, tile)
    return atoms_from_f_mask_stable(n_sc, f_mask_big)


def _build_random_start(n_sc: int, seed: int = 42) -> Atoms:
    rng = np.random.default_rng(seed)
    atoms_rand = make_random(n_sc, rng)
    f_mask = _f_mask_from_atoms(atoms_rand, n_sc)
    return atoms_from_f_mask_stable(n_sc, f_mask)


def _occ_to_f_mask(occ: np.ndarray, n_sc: int) -> np.ndarray:
    return occ[n_sc**3:] == 9


def _chainorder_diagnostics(f_mask: np.ndarray, n_sc: int) -> dict[str, float]:
    """OOF amplitude, OOF motif fraction, and cis fraction."""
    atoms = atoms_from_f_mask_stable(n_sc, f_mask)
    so = SublatticeOccupation.from_atoms(atoms, N=n_sc, species="F")

    fft_idx = n_sc // 3
    oof_amps = []
    for direction in [so.x, so.y, so.z]:
        fft = order_params.chain_fft(direction)
        oof_amps.append(np.abs(fft[:, :, fft_idx]).mean())
    oof_amp = float(np.mean(oof_amps))

    oof_fracs = []
    for direction in [so.x, so.y, so.z]:
        mf = order_params.motif_frequencies(direction, window_length=3)
        frac = sum(
            mf[k].mean() for k in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
            if k in mf
        )
        oof_fracs.append(frac)
    oof_frac = float(np.mean(oof_fracs))

    n_cation = n_sc**3
    neighbours = nb_anion_neighbours(n_sc)
    f_per_nb = f_mask[neighbours].sum(axis=1)
    n_cis = 0
    for b in range(n_cation):
        if f_per_nb[b] != 2:
            continue
        positions = np.where(f_mask[neighbours[b]])[0]
        pair = frozenset({int(positions[0]), int(positions[1])})
        if pair not in _TRANS_PAIRS:
            n_cis += 1
    cis_frac = n_cis / n_cation

    return {"oof_amp": oof_amp, "oof_frac": oof_frac, "cis": cis_frac}


# ---------------------------------------------------------------------------
# Training-set deduplication
# ---------------------------------------------------------------------------

def _load_training_f_masks(
    training_json: Path, cs, n_sc: int,
) -> list[np.ndarray]:
    """Extract F-masks from training structures at the target cell size."""
    n_atoms_target = 4 * n_sc**3
    with training_json.open() as f:
        data = json.load(f)
    masks: list[np.ndarray] = []
    for entry in data:
        atoms_raw = decode(entry["relaxed geometry"])
        if len(atoms_raw) != n_atoms_target:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ideal, _ = map_structure_to_reference(
                atoms_raw, reference=cs.primitive_structure,
                inert_species=["Nb"], suppress_warnings=True,
            )
        masks.append(_f_mask_from_atoms(ideal, n_sc))
    return masks


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------

def _greedy_diverse_selection(
    candidates: list[tuple[float, float, np.ndarray]],
    exclude_masks: list[np.ndarray],
    min_hamming: int,
    max_select: int,
) -> list[tuple[float, float, np.ndarray]]:
    """Greedy selection from (std, temperature, f_mask) tuples.

    Candidates must be sorted by std descending. Skips candidates
    too close to ``exclude_masks`` (training set) or to already-
    selected candidates.
    """
    selected: list[tuple[float, float, np.ndarray]] = []
    for std, T, mask in candidates:
        too_close = any(
            np.count_nonzero(mask != ex) < min_hamming
            for ex in exclude_masks
        )
        if too_close:
            continue
        if any(
            np.count_nonzero(mask != s_mask) < min_hamming
            for _, _, s_mask in selected
        ):
            continue
        selected.append((std, T, mask))
        if len(selected) >= max_select:
            break
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce", type=str, default=None)
    parser.add_argument("--n-sc", type=int, default=6)
    parser.add_argument(
        "--start", choices=["ordered", "random"], default="random",
    )
    parser.add_argument(
        "--committee", type=str, default=None,
        help="Path to committee .npz (default: committee_<ce_stem>.npz)",
    )
    parser.add_argument(
        "--training-json", type=str,
        default=str(Path(__file__).parent / "training_set_refitB.json"),
    )
    parser.add_argument(
        "--moves", choices=["pairswap", "combined"], default="combined",
        help="Move set: pairswap only or combined PairSwap+RowShift+MotifShift.",
    )
    parser.add_argument(
        "--row-shift-weight", type=float, default=0.2,
        help="Relative weight for RowShift (PairSwap weight is 1.0).",
    )
    parser.add_argument(
        "--motif-shift-weight", type=float, default=0.5,
        help="Relative weight for MotifShift (PairSwap weight is 1.0).",
    )
    parser.add_argument(
        "--temperatures", type=str, default=None,
        help="Comma-separated temperature ladder (K). Defaults to "
             "the 17-rung ladder in ce_tools.",
    )
    parser.add_argument(
        "--observe-interval", type=int, default=None,
        help="Observer interval in MC steps (default: block_size).",
    )
    parser.add_argument("--n-cycles", type=int, default=150)
    parser.add_argument("--stash-per-replica", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--min-hamming", type=int, default=MIN_HAMMING)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Master RNG seed for PT and (when --start=random) for the "
        "initial configuration.",
    )
    args = parser.parse_args()

    n_sc = args.n_sc
    n_cation = n_sc**3
    block_size = _block_size(n_sc)

    if args.temperatures is not None:
        temperatures = [float(t) for t in args.temperatures.split(",")]
    else:
        temperatures = list(TEMPERATURES_K)

    observe_interval = args.observe_interval or block_size

    ce_path = _resolve_ce_path(args.ce)
    ce = load_ce(ce_path)

    committee_path = Path(args.committee) if args.committee else (
        Path(__file__).parent / f"committee_{ce_path.stem}.npz"
    )
    committee_params = load_committee(committee_path)
    n_committee = committee_params.shape[0]
    print(f"Committee: {n_committee} models from {committee_path.name}")

    # Training-set F-masks at this cell size (for deduplication).
    training_json = Path(args.training_json)
    cs_dedup = load_ce(ce_path)._cluster_space
    training_masks = _load_training_f_masks(training_json, cs_dedup, n_sc)
    print(f"Training set: {len(training_masks)} structures at "
          f"{n_sc}x{n_sc}x{n_sc} (for deduplication)")

    # Starting structure.
    if args.start == "ordered":
        atoms0 = _build_ordered_start(n_sc)
        if atoms0 is None:
            print(f"n_sc={n_sc} not divisible by 3; falling back to random")
            atoms0 = _build_random_start(n_sc, seed=args.seed)
            label = f"random (ordered unavailable, seed={args.seed})"
        else:
            label = "tiled P3_121"
    else:
        atoms0 = _build_random_start(n_sc, seed=args.seed)
        label = f"random (seed={args.seed})"

    n_atoms = len(atoms0)
    e0 = float(ce.predict(atoms0))
    print(f"Start ({label}): {n_atoms} atoms ({n_sc}x{n_sc}x{n_sc}), "
          f"E/atom = {e0:+.6f}")
    print(f"Block size: {block_size} steps/cycle, "
          f"{args.n_cycles} cycles, {len(temperatures)} replicas")

    # Observer factory — each worker constructs its own observer using
    # the replica's ClusterSpace (avoids pickling icet objects).
    factory = CommitteeObserverFactory(
        committee_path=committee_path,
        interval=observe_interval,
        stash_top_k=args.stash_per_replica,
    )
    print(f"Observer interval: every {observe_interval} steps "
          f"({block_size / observe_interval:.0f}x per cycle)")

    # Move set and ensemble configuration.
    if args.moves == "combined" and n_sc % 3 == 0:
        # Need sublattice index for PairSwap. Use a throwaway CE
        # to avoid mutating the one passed to process_pool.
        ce_probe = load_ce(ce_path)
        probe_calc = ClusterExpansionCalculator(atoms0, ce_probe)
        for i, sl in enumerate(probe_calc.sublattices):
            if set(sl.chemical_symbols) == {"O", "F"}:
                sl_idx = i
                break
        del ce_probe, probe_calc
        moves = [
            (PairSwap(sublattice_index=sl_idx), 1.0),
            (RowShift(n_sc), float(args.row_shift_weight)),
            (MotifShift(n_sc), float(args.motif_shift_weight)),
        ]
        ensemble_kwargs = {
            "moves": moves,
            "ensemble_data_write_interval": block_size,
        }
        ensemble_cls = CustomCanonicalEnsemble
        print(f"Moves: PairSwap(1.0) + RowShift({args.row_shift_weight}) "
              f"+ MotifShift({args.motif_shift_weight})")
    else:
        ensemble_kwargs = None
        ensemble_cls = None
        if args.moves == "combined" and n_sc % 3 != 0:
            print(f"n_sc={n_sc} not divisible by 3; falling back to pairswap")
        print("Moves: PairSwap only")

    pt_kwargs = dict(
        cluster_expansion=ce,
        atoms=atoms0,
        temperatures=temperatures,
        block_size=block_size,
        random_seed=args.seed,
    )
    if ensemble_cls is not None:
        pt_kwargs["ensemble_cls"] = ensemble_cls
        pt_kwargs["ensemble_kwargs"] = ensemble_kwargs

    with CanonicalParallelTempering.process_pool(**pt_kwargs) as pt:
        pt.pool.attach_observer_factory(factory)

        t0 = time.perf_counter()
        history = pt.run(n_cycles=args.n_cycles)
        wall = time.perf_counter() - t0

        containers = pt.pool.data_containers()
        # Final occupations for lowest-energy structure.
        final_occs = [
            pt.pool.current_occupations(r)
            for r in range(len(temperatures))
        ]
        # Retrieve worker-side observer stashes via get_observers.
        stashes_per_replica: list[list[tuple[float, np.ndarray]]] = []
        for r in range(len(temperatures)):
            obs_dict = pt.pool.get_observers(r)
            obs = obs_dict.get("CommitteeUncertainty")
            if obs is not None:
                stashes_per_replica.append(obs.stashed_structures)
            else:
                stashes_per_replica.append([])

    print(f"Ran {args.n_cycles} cycles in {wall:.1f}s (ProcessPool)")

    # Energy summary + per-temperature uncertainty.
    e_trace = history.energies_per_cycle / n_atoms

    print()
    print(f"{'T (K)':>8}  {'E_mean_2h':>10}  "
          f"{'unc_mean (meV)':>15}  {'unc_max (meV)':>14}")
    unc_per_T: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    long_rows: list[pd.DataFrame] = []
    for r, T in enumerate(temperatures):
        mean_2h = e_trace[args.n_cycles // 2:, r].mean()
        dc = containers[r]
        valid = dc.data["committee_std"].notna()
        stds = dc.data.loc[valid, "committee_std"].values
        steps = dc.data.loc[valid, "mctrial"].values
        unc_per_T[T] = (steps, stds)
        print(f"{T:>8.0f}  {mean_2h:>10.5f}  "
              f"{stds.mean() * 1000:>15.2f}  {stds.max() * 1000:>14.2f}")
        # Long-format snapshot record for downstream analysis.
        df_r = dc.data.loc[valid, ["mctrial", "committee_std"]].copy()
        df_r["T_K"] = float(T)
        df_r["replica_index"] = r
        long_rows.append(df_r)

    # Persist per-snapshot committee-std observations.
    obs_df = pd.concat(long_rows, ignore_index=True)
    obs_csv = (
        Path.cwd()
        / f"pt_buss_observations_{ce_path.stem}_N{n_sc}_{args.start}_seed{args.seed}.csv"
    )
    obs_df.to_csv(obs_csv, index=False)
    print(f"Wrote per-snapshot committee-std observations to {obs_csv.name}")

    # Write lowest-energy final structure.
    final_energies = e_trace[-1, :]
    best_r = int(np.argmin(final_energies))
    best_T = temperatures[best_r]
    best_E = final_energies[best_r]
    best_mask = _occ_to_f_mask(final_occs[best_r], n_sc)
    best_atoms = atoms_from_f_mask(n_sc, best_mask)
    best_path = Path.cwd() / f"buss_N{n_sc}_{args.start}_seed{args.seed}_lowest_energy.vasp"
    write(best_path, best_atoms, format="vasp", direct=True)
    diag = _chainorder_diagnostics(best_mask, n_sc)
    print(f"\nLowest-energy structure: T={best_T:.0f} K, "
          f"E/atom={best_E:.6f}")
    print(f"  cis={diag['cis']:.3f}  oof_amp={diag['oof_amp']:.3f}  "
          f"oof_frac={diag['oof_frac']:.3f}")
    print(f"  Saved to {best_path.name}")

    # Collect stashed structures from worker-side observers.
    all_candidates: list[tuple[float, float, np.ndarray]] = []
    for r, T in enumerate(temperatures):
        for std, occ in stashes_per_replica[r]:
            all_candidates.append((std, T, _occ_to_f_mask(occ, n_sc)))
    all_candidates.sort(key=lambda x: -x[0])
    print(f"\nCollected {len(all_candidates)} stashed structures")

    # Persist the full per-replica stash for offline inspection.
    stash_dir = (
        Path.cwd()
        / f"buss_stash_{ce_path.stem}_N{n_sc}_{args.start}_seed{args.seed}"
    )
    stash_dir.mkdir(exist_ok=True)
    stash_records: list[dict] = []
    for r, T in enumerate(temperatures):
        for k, (std, occ) in enumerate(stashes_per_replica[r]):
            f_mask = _occ_to_f_mask(occ, n_sc)
            atoms_out = atoms_from_f_mask(n_sc, f_mask)
            fname = f"T{int(round(T)):04d}K_rank{k:02d}_std{std * 1000:05.2f}meV.vasp"
            write(stash_dir / fname, atoms_out, format="vasp", direct=True)
            diag = _chainorder_diagnostics(f_mask, n_sc)
            stash_records.append(
                {
                    "T_K": float(T),
                    "replica_index": r,
                    "stash_rank": k,
                    "committee_std_meV": float(std * 1000),
                    "filename": fname,
                    "cis": diag["cis"],
                    "oof_amp": diag["oof_amp"],
                    "oof_frac": diag["oof_frac"],
                }
            )
    pd.DataFrame(stash_records).to_csv(stash_dir / "stash_index.csv", index=False)
    print(f"Wrote per-replica stash ({len(stash_records)} structures) to {stash_dir}")

    # Greedy diversity selection, excluding training-set duplicates.
    selected = _greedy_diverse_selection(
        all_candidates, training_masks,
        args.min_hamming, args.max_candidates,
    )
    n_excluded = sum(
        1 for _, _, mask in all_candidates
        if any(np.count_nonzero(mask != ex) < args.min_hamming
               for ex in training_masks)
    )
    print(f"Excluded {n_excluded} candidates too close to training set")
    print(f"Selected {len(selected)} diverse candidates")

    # Write candidates with chainorder diagnostics.
    out_dir = Path.cwd()
    if selected:
        print()
        print(f"{'#':>2}  {'std (meV)':>10}  {'T (K)':>7}  "
              f"{'cis':>6}  {'oof_amp':>8}  {'oof_frac':>9}")
        for i, (std, T, f_mask) in enumerate(selected):
            diag = _chainorder_diagnostics(f_mask, n_sc)
            print(f"{i:>2}  {std * 1000:>10.2f}  {T:>7.0f}  "
                  f"{diag['cis']:>6.3f}  {diag['oof_amp']:>8.3f}  "
                  f"{diag['oof_frac']:>9.3f}")
            atoms_out = atoms_from_f_mask(n_sc, f_mask)
            tag = f"buss_N{n_sc}_{args.start}_seed{args.seed}_{i:02d}"
            write(out_dir / f"{tag}.vasp", atoms_out,
                  format="vasp", direct=True)
        print(f"Written to {out_dir}/buss_N{n_sc}_{args.start}_seed{args.seed}_*.vasp")
    else:
        print("\nNo candidates survived deduplication.")

    # Plot.
    fig, (ax_e, ax_u) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    cmap = plt.get_cmap("coolwarm")
    for r, T in enumerate(temperatures):
        colour = cmap(r / max(1, len(temperatures) - 1))
        ax_e.plot(
            np.arange(args.n_cycles + 1), e_trace[:, r],
            color=colour, label=f"{T:.0f} K", linewidth=1, alpha=0.9,
        )
        steps_r, stds_raw = unc_per_T[T]
        ax_u.plot(
            steps_r / block_size, stds_raw * 1000,
            color=colour, linewidth=1, alpha=0.9,
        )

    ax_e.set_ylabel("CE energy per atom (eV)")
    ax_e.legend(loc="best", fontsize=7, ncol=3)
    ax_e.grid(True, alpha=0.3)

    ax_u.set_xlabel("Cycle")
    ax_u.set_ylabel("Committee std (meV/atom)")
    ax_u.grid(True, alpha=0.3)

    fig.suptitle(
        f"BUSS PT on {n_sc}x{n_sc}x{n_sc} NbO2F, "
        f"start={label}, {n_committee} committee models",
    )
    fig.tight_layout()
    plot_path = (
        out_dir / f"pt_buss_{ce_path.stem}_N{n_sc}_{args.start}_seed{args.seed}.png"
    )
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    main()
