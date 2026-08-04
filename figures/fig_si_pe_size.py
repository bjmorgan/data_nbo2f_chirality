"""SI: system-size dependence of P(E|T_c) and the latent heat (NbO2F chirality).

(A) P(E|T_c(L)) for L = 6, 9, 12, 15, 18, each at that size's equal-weight
    temperature, centred on its free-energy minimum E*(L) (same convention as
    the main-text P(E) panel) and plotted per atom so the four distributions
    are directly comparable; each normalised to unit area. The trough between
    the two peaks deepens with L (first claim).
(B) latent heat per atom, dE/N, vs L. The per-atom latent heat rises with L
    with decelerating increments toward a bulk constant (~3.5 meV/atom): the
    latent heat is extensive, approached from below by finite-size rounding.
    Total dE therefore grows slightly faster than proportionally to N at these
    sizes -- strict proportionality is only asymptotic.

Everything is computed from the production stitched ln g(E) (seed-averaged);
no numbers are hard-coded. The per-L equal-weight T_c reproduces
tc_estimators.csv to 0.1 K.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from ase.units import kB
from figure_formatting.figure_formatting import set_formatting, tableau
from mchammer_pt.analysis.coexistence import equal_area_temperature

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow" / "rewl"))
from tc_estimators import equal_area_robust   # noqa: E402  (same rule as the T_c pipeline)

set_formatting()
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "dejavusans"
T = tableau
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "rewl"      # deposited data files (read-only)

Lseq = [6, 9, 12, 15, 18]
NAT = {L: 4 * L**3 for L in Lseq}          # atoms in the L^3 supercell
LW, RLW, FS = 0.7, 0.7, 7                   # match the main finite-T figure
# same L-colouring as the main finite-temperature figure (reversed viridis)
Lcol = {L: cm.viridis(s) for L, s in zip(Lseq, [0.80, 0.56, 0.34, 0.17, 0.0])}


def seedavg_dos(L: int) -> pd.DataFrame:
    """Seed-averaged production stitched ln g on the common energy overlap.

    The grid follows each size's own sampled bin width, so no resolution is
    discarded (L=6 was run on a finer energy grid than the larger sizes).
    """
    seeds = [pd.read_csv(DATA / f"L{L}/prod/seed{s}/stitched_dos.csv") for s in range(1, 6)]
    elo = max(d["energy"].min() for d in seeds)
    ehi = min(d["energy"].max() for d in seeds)
    # rounded: the energies are ~1e4 eV, so differencing them leaves the bin
    # width a few parts in 1e11 short of its nominal value
    step = round(min(float(np.median(np.diff(d["energy"].to_numpy()))) for d in seeds), 6)
    E = np.arange(elo, ehi + 1e-9, step)
    S = np.mean([np.interp(E, d["energy"], d["entropy"]) for d in seeds], axis=0)
    return pd.DataFrame({"energy": E, "entropy": S})


def coexistence(L: int):
    """(dos, T_c, E*, latent_heat, trough=ln[Pmax/Pmin]) at the equal-weight T_c.

    The detection smoothing is chosen per size by the same sigma-robustness rule
    the T_c pipeline uses, rather than fixed: the broader coexistence region at
    the largest size needs more smoothing for the saddle to be located reliably.
    """
    dos = seedavg_dos(L)
    _, _, sigma = equal_area_robust(dos)
    cp = equal_area_temperature(dos, smoothing_sigma=sigma)
    trough = cp.barrier_height / (kB * cp.T_K)      # ln[P_peak / P_saddle]
    return dos, cp.T_K, cp.split.E_star, cp.latent_heat, trough


fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.6, 2.2))   # half the 4-panel width -> single column, same per-panel size
for ax in (axA, axB):
    ax.set_box_aspect(1)
    ax.locator_params(nbins=5)
    ax.yaxis.labelpad += 2

# (A) P(E|T_c(L)) centred per atom, unit area
latN = {}
for L in Lseq:
    dos, Tc, Estar, latent, trough = coexistence(L)
    latN[L] = latent / NAT[L] * 1000.0
    S = gaussian_filter1d(dos["entropy"].to_numpy(), 1.0)
    lp = S - dos["energy"].to_numpy() / (kB * Tc); lp -= lp.max()
    P = np.exp(lp)
    xe = (dos["energy"].to_numpy() - Estar) / NAT[L] * 1000.0   # meV/atom, centred on E*
    P /= np.trapezoid(P, xe)                                    # unit area
    axA.plot(xe, P, color=Lcol[L], lw=LW, label=f"$L={L}$")
axA.set_xlim(-3.5, 3.5)
axA.set_ylim(0, 1.6)        # headroom so the legend clears the tallest (L=18) peaks
axA.set_xlabel(r"$(E-E^*)/N$  (meV/atom)")
axA.set_ylabel(r"$P(E\,|\,T_c)$  (unit area)")
axA.set_title("(a)")
axA.legend(fontsize=FS, loc="upper left", handlelength=1.0, labelspacing=0.25,
           borderaxespad=0.3)

# (B) latent heat per atom vs L -- rises with decelerating increments toward a bulk constant
for L in Lseq:
    axB.plot(L, latN[L], "o", ms=4.5, color=Lcol[L], markeredgewidth=0, zorder=3)
axB.plot(Lseq, [latN[L] for L in Lseq], "-", color=T["grey"], lw=RLW, zorder=1)
axB.set_xlim(3, 21)
axB.set_xticks(Lseq)
axB.set_ylim(0, 4)
axB.set_xlabel(r"$L$")
axB.set_ylabel(r"$\Delta E / N$  (meV/atom)")
axB.set_title("(b)")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_si_pe_size.{ext}", dpi=300)
print("wrote", HERE / "fig_si_pe_size.png")

# --- numbers for the SI text ---
print(f"\n{'L':>3} {'N':>6} {'Tc(K)':>7} {'dE(eV)':>8} {'dE/N(meV)':>10} {'trough':>7}")
for L in Lseq:
    _, Tc, _, latent, trough = coexistence(L)
    print(f"{L:>3} {NAT[L]:>6} {Tc:>7.1f} {latent:>8.2f} {latent/NAT[L]*1000:>10.3f} {trough:>7.2f}")
