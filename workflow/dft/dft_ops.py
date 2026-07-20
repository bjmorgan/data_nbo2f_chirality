"""Per-config DFT order parameters + energy for the 3x3x3 demonstration (sec:dft).

Pipeline stage between the DFT outputs and the energy-vs-coherence figure:
for each 3x3x3 config, read its ideal input structure (POSCAR.init) and DFT
energy (OUTCAR), compute the order parameters via the validated
ChainOrderObserver, and classify by the order parameters themselves --
  random            : oof below the period-3 threshold (no chain ordering)
  partial           : period-3 present but incomplete (OOF_MIN < oof < OOF_FULL);
                      not a clean ordering, so excluded from the figure
  OOF random-phase  : full period-3 per chain (oof ~ 1/3) but cis incomplete
  cis+OOF orbit     : full period-3 AND cis-NbO4F2 everywhere (cis ~ 1)

Writes dft_ops.csv. Nothing is hard-coded; the classification comes from the
data. Only 3x3x3 is used: it is the theorem cell, 2x2x2 cannot host period-3
ordering, and the 6x6x6 CE-training set uses a different k-mesh (Gamma-only vs
2x2x2 Monkhorst) so its energies are not comparable on the same axis.

The structure is read from POSCAR.init (ideal, undistorted) and re-expressed in
stable Nb-first ordering via the recovered F-mask, so the observer's cis/NbO4F2
branch is well defined.

    python dft_ops.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read
from chainorder import SublatticeOccupation
from nbo2f_analysis.ce_tools import _f_mask_from_atoms, atoms_from_f_mask_stable
from nbo2f_analysis.chain_order_observer import build_chain_order_observer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "dft"       # deposited DFT configs (read-only)
BUILD = REPO / "build" / "dft"     # regenerated products
N_SC = 3
OOF_MIN = 0.30     # random ceiling: at/below this, no period-3 chain ordering
OOF_FULL = 0.32    # full period-3 OOF (per-chain 1/3); OOF_MIN < oof < OOF_FULL is partial
CIS_FULL = 0.98    # cis-NbO4F2 at (essentially) every cation


def energy_per_atom(config_dir: Path) -> float:
    """Relaxed DFT energy per atom, read directly from the config's OUTCAR.

    Sourcing energy from the kept OUTCAR (not a derived collation) keeps the
    table in step with the calculation directory: any config with a completed
    OUTCAR is included, without a separate re-collation step.
    """
    relaxed = read(config_dir / "OUTCAR")
    return relaxed.get_potential_energy() / len(relaxed)


def match_orbit(atoms, obs) -> int:
    """Symmetry-inequivalent orbit index (0-11) this config belongs to, via the
    observer's per-orbit similarity (proper or improper match -> 1.0); -1 if none."""
    occ = SublatticeOccupation.from_atoms(atoms, N=N_SC, species="F")
    c1 = obs._per_cell_counts(occ.occupation)
    c0 = obs._tile**3 - c1
    best, which = -1.0, -1
    for label, ref in obs._orbit_refs.items():
        s = max(obs._best_similarity(c1, c0, ref.proper),
                obs._best_similarity(c1, c0, ref.improper))
        if s > best:
            best, which = s, int(label)
    return which if best > 0.999 else -1


def classify(oof: float, cis: float, nbo4f2: float) -> str:
    if cis < 0.05:
        return "trans"      # deliberate cis-free (all-trans) construction -- not a random config
    if oof <= OOF_MIN:
        return "random"
    if oof < OOF_FULL:
        return "partial"    # incomplete period-3 OOF -- neither random nor a clean ordering
    return "orbit" if cis > CIS_FULL else "OOF-random-phase"


def main() -> None:
    obs = build_chain_order_observer(
        n_sc=N_SC, interval=1,
        ops=("chi_11", "icoh_global", "oof_amp", "cis_frac", "nbo4f2_frac"))
    rows = []
    for d in sorted((DATA / "3x3x3").glob("config_*")):
        cfg = d.name.split("config_")[1]
        if not (d / "OUTCAR").exists():
            continue
        f_mask = _f_mask_from_atoms(read(d / "POSCAR.init"), N_SC)
        stable = atoms_from_f_mask_stable(N_SC, f_mask)
        o = obs.get_observable(stable)
        cls = classify(o["oof_amp"], o["cis_frac"], o["nbo4f2_frac"])
        rows.append(dict(
            config=cfg, energy_per_atom=energy_per_atom(d),
            oof=o["oof_amp"], cis=o["cis_frac"], nbo4f2=o["nbo4f2_frac"],
            icoh=o["icoh_global"], chi=abs(o["chi_11"]),
            orbit=match_orbit(stable, obs) if cls == "orbit" else -1, cls=cls))
    df = pd.DataFrame(rows).sort_values("energy_per_atom").reset_index(drop=True)
    out = BUILD / "dft_ops.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} configs)\n")
    print(df.groupby("cls").agg(
        n=("config", "count"), epa_min=("energy_per_atom", "min"),
        epa_max=("energy_per_atom", "max")).to_string(float_format=lambda v: f"{v:.4f}"))
    gs = df.iloc[0]
    print(f"\nground state: config_{gs.config}  epa={gs.energy_per_atom:.4f}  "
          f"cis={gs.cis:.2f} oof={gs.oof:.2f} icoh={gs.icoh:.2f} |chi|={gs.chi:.2f}")


if __name__ == "__main__":
    main()
