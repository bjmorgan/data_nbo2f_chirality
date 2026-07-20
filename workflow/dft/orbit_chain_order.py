"""Per-orbit ChainOrder chiral invariants for the twelve cis+OOF orbits (App. B).

Downstream companion to dft_ops.py. For each of the twelve enumerated orbit
representatives (``structures/orbit_representatives/orbit_*.vasp`` -- the same
set the SI gallery draws), computes the ChainOrder chiral diagnostics via the
validated observer:

  icoh         : inter-chain phase coherence (``icoh_global``)
  icoh_nn      : nearest-neighbour inter-chain coherence (``icoh_nn``)
  helical_amp  : helical amplitude ``|E+|^2 + |E-|^2`` (``circ_coherence``)
  chi          : chirality pseudoscalar ``|E+|^2 - |E-|^2`` (``chirality``)
  chi_11       : signed template similarity to orbit 11 (cross-ref dft_ops.csv)

and writes orbit_chain_order.csv (one row per orbit, ascending DFT energy).

Result (Appendix B): icoh = 1 for exactly one orbit -- P3_121 (orbit 11) --
and every other orbit has icoh <= 1/2. Since the twelve orbits exhaust the
cis+OOF configurations on the commensurate 3x3x3 cell, this shows the helix is
the only fully coherent cis+OOF state; the helical amplitude tells the same
story (uniquely 1/4 at orbit 11).

Caveat: on the 3x3x3 cell icoh averages the inter-chain correlation over the
3x3 chain-plane lag set (eight off-origin lags), so the sub-unity values are
quantised and cell-size-specific. Only icoh = 1 (all lags phase-locked) is a
size-robust statement; the ranking below 1 is particular to this cell.

    python orbit_chain_order.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ase.io import read
from nbo2f_analysis.ce_tools import _f_mask_from_atoms, atoms_from_f_mask_stable
from nbo2f_analysis.chain_order_observer import build_chain_order_observer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REP = REPO / "data" / "structures" / "orbit_representatives"   # deposited orbit reps
BUILD = REPO / "build" / "dft"                                 # regenerated products
DFT_OPS = BUILD / "dft_ops.csv"
N_SC = 3
OPS = ("icoh_global", "icoh_nn", "circ_coherence", "chirality", "chi_11")


def dE_by_orbit() -> dict[int, float]:
    """Per-orbit DFT energy above the ground state (meV/atom), from dft_ops.csv."""
    df = pd.read_csv(DFT_OPS)
    o = df[df.cls == "orbit"]
    return ((o.groupby("orbit")["energy_per_atom"].mean() - df.energy_per_atom.min())
            * 1000.0).to_dict()


def main() -> None:
    obs = build_chain_order_observer(n_sc=N_SC, interval=1, ops=OPS)
    dE = dE_by_orbit()
    rows = []
    for f in sorted(REP.glob("orbit_*.vasp")):
        tok = f.stem.split("_")            # orbit_<idx>_sg<sg>_<HM...>_stab<n>
        idx, hm = int(tok[1]), "_".join(tok[3:-1])
        stable = atoms_from_f_mask_stable(N_SC, _f_mask_from_atoms(read(f), N_SC))
        o = obs.get_observable(stable)
        rows.append(dict(orbit=idx, hm=hm, dE_meV=round(dE[idx], 3),
                         icoh=o["icoh_global"], icoh_nn=o["icoh_nn"],
                         helical_amp=o["circ_coherence"], chi=o["chirality"],
                         chi_11=o["chi_11"]))
    df = pd.DataFrame(rows).sort_values(["dE_meV", "orbit"]).reset_index(drop=True)
    out = BUILD / "orbit_chain_order.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:7.4f}")
    print(df.to_string(index=False))
    fully = sorted(df[df.icoh > 0.99].orbit)
    print(f"\nwrote {out}")
    print(f"fully coherent (icoh = 1): orbits {fully}  (expect only 11);  "
          f"max icoh below that: {df[df.orbit != 11].icoh.max():.4f}")


if __name__ == "__main__":
    main()
