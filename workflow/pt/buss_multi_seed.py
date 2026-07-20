"""Run BUSS with multiple seeds to find CV-distinct high-uncertainty structures."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.io.jsonio import decode
from icet import ClusterExpansion
from icet.tools import map_structure_to_reference
from mchammer_pt import CanonicalParallelTempering
from mchammer_moves import CustomCanonicalEnsemble, PairSwap
from mchammer.calculators import ClusterExpansionCalculator

from bootstrap_committee import CommitteeObserverFactory, load_committee
from ce_tools import (
    _resolve_ce_path,
    atoms_from_f_mask,
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
    load_ce,
)
from chain_geometry import RowShift, MotifShift

MIN_CV_DIST = 0.05
MIN_STD = 0.5  # meV


def main() -> None:
    ce_path = str(_resolve_ce_path(None))
    ce = load_ce(ce_path)
    n_sc = 6
    block_size = 2 * 3 * n_sc**3

    # Fresh CS for CV computation
    cs = ClusterExpansion.read(ce_path)._cluster_space
    prim = cs.primitive_structure

    # Collect existing CVs (training + running)
    existing_cvs: list[np.ndarray] = []
    with open(Path(__file__).parent / "training_set.json") as f:
        data = json.load(f)
    for entry in data:
        if "6x6x6" not in entry["path"]:
            continue
        atoms = decode(entry["relaxed geometry"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ideal, _ = map_structure_to_reference(
                atoms, reference=prim, inert_species=["Nb"],
                suppress_warnings=True,
            )
        cv = cs.get_cluster_vector(ideal)
        if len(cv) == len(cs):
            existing_cvs.append(np.array(cv))

    DFT = Path("../dft_calculations/6x6x6")
    for i in range(18, 28):
        p = DFT / f"config_{i:02d}" / "POSCAR.init"
        if p.exists():
            atoms = read(str(p), format="vasp")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ideal, _ = map_structure_to_reference(
                    atoms, reference=prim, inert_species=["Nb"],
                    suppress_warnings=True,
                )
            existing_cvs.append(np.array(cs.get_cluster_vector(ideal)))

    print(f"Existing CVs: {len(existing_cvs)}")

    # Committee + production GS reference
    committee = load_committee(Path("committee_paircut9_5_5.npz"))
    cv_gs = np.array(cs.get_cluster_vector(build_tiled_groundstate_atoms()))
    e_gs_prod = cv_gs @ ce.parameters

    # Sublattice index from throwaway CE
    ce_probe = load_ce(ce_path)
    probe_calc = ClusterExpansionCalculator(
        build_tiled_groundstate_atoms(), ce_probe,
    )
    sl_idx = next(
        i for i, sl in enumerate(probe_calc.sublattices)
        if set(sl.chemical_symbols) == {"O", "F"}
    )
    del ce_probe, probe_calc

    temperatures = [
        100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
        600, 650, 700, 800, 1000, 1500, 3000,
    ]

    moves = [
        (PairSwap(sublattice_index=sl_idx), 1.0),
        (RowShift(n_sc), 0.2),
        (MotifShift(n_sc), 0.5),
    ]

    all_novel: list[tuple[float, float, int, np.ndarray, np.ndarray, float]] = []

    for seed in [1, 2, 3, 4, 5]:
        print(f"\n=== Seed {seed} ===")
        atoms0 = build_tiled_groundstate_atoms()

        factory = CommitteeObserverFactory(
            committee_path="committee_paircut9_5_5.npz",
            interval=200,
            stash_top_k=10,
        )

        with CanonicalParallelTempering.process_pool(
            cluster_expansion=ce,
            atoms=atoms0,
            temperatures=temperatures,
            block_size=block_size,
            random_seed=seed,
            ensemble_cls=CustomCanonicalEnsemble,
            ensemble_kwargs={
                "moves": moves,
                "ensemble_data_write_interval": block_size,
            },
        ) as pt:
            pt.pool.attach_observer_factory(factory)
            pt.run(n_cycles=300)

            for r, T in enumerate(temperatures):
                obs_dict = pt.pool.get_observers(r)
                obs = obs_dict.get("CommitteeUncertainty")
                if obs is None:
                    continue
                for std_val, occ in obs.stashed_structures:
                    f_mask_s = occ[n_sc**3:] == 9
                    atoms_s = atoms_from_f_mask_stable(n_sc, f_mask_s)
                    cv = np.array(cs.get_cluster_vector(atoms_s))
                    preds = committee @ cv
                    std_mev = preds.std() * 1000
                    gap = (cv @ ce.parameters - e_gs_prod) * 1000

                    all_ref = existing_cvs + [c[3] for c in all_novel]
                    min_dist = min(np.linalg.norm(cv - ref) for ref in all_ref)

                    if min_dist >= MIN_CV_DIST and std_mev > MIN_STD:
                        all_novel.append(
                            (std_mev, T, seed, cv, f_mask_s, gap)
                        )
                        print(
                            f"  T={T:>5.0f}  std={std_mev:.2f}  "
                            f"gap={gap:+.1f}  cv_dist={min_dist:.3f}"
                        )

    # Per-temperature summary: top candidate at each T
    by_temp: dict[float, list] = {}
    for entry in all_novel:
        by_temp.setdefault(entry[1], []).append(entry)

    print(f"\n=== Top candidate per temperature ===")
    print(f"{'T':>6}  {'std':>7}  {'gap':>8}  {'n_found':>7}")
    for T in sorted(by_temp):
        candidates = sorted(by_temp[T], key=lambda x: -x[0])
        top = candidates[0]
        print(f"{T:>6.0f}  {top[0]:>6.2f}  {top[5]:>+7.1f}  {len(candidates):>7}")

    # Save top candidate per temperature
    out_dir = Path(__file__).parent
    saved = 0
    for T in sorted(by_temp):
        top = max(by_temp[T], key=lambda x: x[0])
        std, _, seed, cv, mask, gap = top
        atoms_out = atoms_from_f_mask(n_sc, mask)
        write(
            str(out_dir / f"buss_novel_{saved:02d}.vasp"),
            atoms_out, format="vasp", direct=True,
        )
        saved += 1

    # Also global top-10 for reference
    all_novel.sort(key=lambda x: -x[0])
    print(f"\n=== Global top 10 by committee std: {len(all_novel)} total ===")
    print(f"{'#':>3}  {'std':>7}  {'T':>6}  {'seed':>4}  {'gap':>8}")
    for i, (std, T, seed, cv, mask, gap) in enumerate(all_novel[:10]):
        print(f"{i:>3}  {std:>6.2f}  {T:>6.0f}  {seed:>4}  {gap:>+7.1f}")

    if all_novel:
        print(f"\nWrote {saved} structures (1 per temperature) to "
              f"buss_novel_*.vasp")
    else:
        print("\nNo novel candidates found above threshold.")


if __name__ == "__main__":
    main()
