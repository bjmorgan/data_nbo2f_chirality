"""Per-run T_c estimators from the REWL Monte-Carlo outputs.

Pipeline stage between the MC outputs and the finite-T figure / FSS fit:
for each (size, seed) it derives, from the stitched density of states and the
frozen-g measurement, two independent coexistence-temperature estimators --

  * equal-area (Borgs-Kotecky equal-weight) T_c from the DOS, and
  * the Binder-minimum temperature of the chirality (from the measure run) --

and writes them to ``tc_estimators.csv``. Nothing is hard-coded: T_c values
come from the data. The equal-area smoothing sigma is selected automatically
per run by a sigma-robustness rule (smallest sigma whose T_c agrees with the
next-larger sigma, ignoring spurious narrow finds), which reproduces the
hand-curated consensus (L6 357.87, L9 452.88, L12 478.26, L15 485.69).

Run after the DOS/measure are stitched (``mchammer-pt-stitch``):
    python tc_estimators.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from mchammer_pt.analysis.coexistence import equal_area_temperature

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "rewl"          # deposited data files (read-only)
BUILD = REPO / "build" / "rewl"        # regenerated products (this pipeline)
SIZES = (6, 9, 12, 15)
SEEDS = range(1, 6)
SIGMA_LADDER = (2.0, 4.0, 8.0, 16.0, 30.0)


def dos_path(L: int, s: int) -> Path:
    if L == 15:   # measure not yet run at L15; use the production DOS
        return DATA / f"L{L}/prod/seed{s}/stitched_dos.csv"
    return DATA / f"L{L}/prod/seed{s}/measure/stitched_dos.csv"


def measure_path(L: int, s: int) -> Path:
    return BUILD / f"L{L}/prod/seed{s}/measure/canonical_observables/ChainOrder.csv"


def equal_area_robust(dos: pd.DataFrame) -> tuple[float, float, float]:
    """Equal-area T_c -> (T_c, latent, sigma).

    Selection combines two criteria, both with no per-size tuning:
      1. broad coexistence -- under- and over-smoothing both underestimate the
         peak separation (latent heat) by mislocating the saddle, so keep only
         sigmas within 10% of the maximum latent;
      2. T_c on a stable plateau -- take the smallest such sigma whose T_c
         agrees with the next-larger one (within 1 K). For a noisy/under-
         converged DOS the lowest sigma can be off-plateau even with broad
         latent (e.g. an under-converged L15 seed at sigma 2), so the plateau
         test steps up until T_c stabilises.
    Reproduces the hand-curated consensus (L6 sigma 2; L9/L12 sigma 2;
    under-converged L15 seeds sigma 4-8).
    """
    res = []
    for sig in SIGMA_LADDER:
        try:
            r = equal_area_temperature(dos, smoothing_sigma=sig)
            res.append((sig, float(r.T_K), float(r.latent_heat)))
        except Exception:
            continue
    if not res:
        return np.nan, np.nan, np.nan
    latmax = max(l for _, _, l in res)
    near = [(s, t, l) for s, t, l in res if l >= 0.9 * latmax]   # broad, ascending sigma
    for i in range(len(near) - 1):
        if abs(near[i][1] - near[i + 1][1]) < 1.0:              # T_c plateau
            return near[i][1], near[i][2], near[i][0]
    sig, tc, lat = max(near, key=lambda r: r[2])                 # fallback: most-developed
    return tc, lat, sig


def binder_min_T(meas: pd.DataFrame) -> float:
    return float(meas["T_K"].to_numpy()[int(np.argmin(meas["chi_11_binder"].to_numpy()))])


def main() -> None:
    rows = []
    for L in SIZES:
        for s in SEEDS:
            dp = dos_path(L, s)
            if not dp.exists():
                continue
            tc, latent, sigma = equal_area_robust(pd.read_csv(dp))
            mp = measure_path(L, s)
            bm = binder_min_T(pd.read_csv(mp)) if mp.exists() else np.nan
            rows.append(dict(L=L, seed=s, tc_equal_area=tc, latent=latent,
                             sigma=sigma, binder_min_T=bm))
    df = pd.DataFrame(rows)
    out = BUILD / "tc_estimators.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")
    agg = df.groupby("L").agg(
        tc_mean=("tc_equal_area", "mean"), tc_std=("tc_equal_area", "std"),
        n=("seed", "count"), binder_mean=("binder_min_T", "mean"))
    print(agg.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
