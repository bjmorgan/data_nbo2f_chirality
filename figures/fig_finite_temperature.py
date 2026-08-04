"""Finite-temperature 4-panel figure (NbO2F chirality manuscript).

Panels (left -> right):
  a  <O>(T) order-parameter hierarchy   -- chirality/icoh collapse; cis/OOF persist
  b  P(E|T_c) at the largest L          -- first-order coexistence (two phases)
  c  chirality Binder cumulant vs T     -- negative dip deepening with L
  d  T_c(L) vs 1/L^3                     -- equal-area + Binder-min estimators -> same T_c(inf)

Built on figure_formatting. Reads the deposited densities of states from
../data/rewl and the reweighted observables from ../build/rewl.

Pipeline: MC outputs -> workflow/rewl/tc_estimators.py (per-run equal-weight and
Binder-min estimators) -> build/rewl/tc_estimators.csv -> this figure. No T_c
values are hard-coded, so the fitted T_c(inf) in panel (d) follows the data.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from figure_formatting.figure_formatting import set_formatting, tableau

set_formatting()
mpl.rcParams["mathtext.fontset"] = "dejavusans"   # italic math variables (custom Helvetica falls back to upright)
T = tableau
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "rewl"      # deposited data files (read-only)
BUILD = HERE.parent / "build" / "rewl"    # regenerated products (make)
kB = 8.617333e-5

# Per-run T_c estimators are computed by workflow/rewl/tc_estimators.py and
# aggregated here -- no hard-coded T_c values.
_EST = BUILD / "tc_estimators.csv"
if not _EST.exists():
    raise SystemExit(f"missing {_EST}\nrun  make  first")
_est = pd.read_csv(_EST)
TC      = _est.groupby("L")["tc_equal_area"].mean().to_dict()
TC_STD  = _est.groupby("L")["tc_equal_area"].std().to_dict()
BINDERT = _est.dropna(subset=["binder_min_T"]).groupby("L")["binder_min_T"].mean().to_dict()
NAT     = {L: 4 * L**3 for L in (6, 9, 12, 15, 18)}
Lseq = [6, 9, 12, 15, 18]
# viridis reversed: L=6,9,12 get well-separated green/teal/blue-purple for the
# Binder curves in (c); L=15,18 (panel d markers only) take the dark-purple end.
# Non-uniform stops for L = 6, 9, 12, 15, 18. Shared by (c) and (d).
_vir = [0.80, 0.56, 0.34, 0.17, 0.0]
Lcol = {L: cm.viridis(s) for L, s in zip(Lseq, _vir)}


def dosfiles(L):
    return [DATA / f"L{L}/prod/seed{s}/stitched_dos.csv" for s in range(1, 6)]


def seedavg(L, col):
    fs = [BUILD / f"L{L}/prod/seed{s}/measure/canonical_observables/ChainOrder.csv" for s in range(1, 6)]
    return np.vstack([pd.read_csv(f)[col].to_numpy() for f in fs]).mean(0)


def Tgrid(L):
    f = BUILD / f"L{L}/prod/seed1/measure/canonical_observables/ChainOrder.csv"
    return pd.read_csv(f)["T_K"].to_numpy()


def pe_and_saddle(L):
    """Seed-averaged consensus P(E|T_c); returns (E, P, E_saddle)."""
    seeds = [pd.read_csv(f) for f in dosfiles(L)]
    elo = max(d["energy"].min() for d in seeds)
    ehi = min(d["energy"].max() for d in seeds)
    E = np.arange(elo, ehi, 0.1)
    S = gaussian_filter1d(
        np.mean([np.interp(E, d["energy"], d["entropy"]) for d in seeds], axis=0), 1.0)
    lp = S - E / (kB * TC[L]); lp -= lp.max(); p = np.exp(lp); p /= np.trapezoid(p, E)
    pk, _ = find_peaks(p, height=p.max() * 0.05)
    if len(pk) >= 2:
        iL, iH = sorted(pk[np.argsort(p[pk])[-2:]])
        iS = iL + int(np.argmin(p[iL:iH + 1]))
        return E, p, E[iS]
    return E, p, E[int(np.argmax(p))]


# double-column 1x4 banner: ~4 cm square panels at 8 pt text
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
axOP, axPE, axBinder, axFSS = axes   # L->R panels: (a) order params, (b) P(E), (c) Binder, (d) FSS
for ax in axes.flat:
    ax.set_box_aspect(1)
    ax.locator_params(nbins=5)
    ax.yaxis.labelpad += 2

LW, RLW, FS = 0.7, 0.7, 7   # all lines 0.7 pt (match axes); legend/in-plot fontsize

# (a) <O>(T) order-parameter hierarchy at the largest L
L = 18
OPS = [("chirality_sq_mean", r"$|\chi|$", T["red"], True),
       ("icoh_global_mean", "icoh", T["orange"], False),
       ("oof_amp_mean", "OOF", T["blue"], False),
       ("cis_frac_mean", "cis", T["green"], False)]
for col, lab, c, rms in OPS:
    y = np.sqrt(seedavg(L, col)) if rms else seedavg(L, col)
    axOP.plot(Tgrid(L), y, color=c, lw=LW, label=lab)
axOP.axvline(TC[L], color=T["grey"], ls="--", lw=RLW)
axOP.set_xlim(300, 700); axOP.set_ylim(0, 1)
axOP.set_xlabel(r"$T$  (K)")
axOP.set_ylabel(r"$\langle O\rangle(T)$")
axOP.set_title("(a)")
axOP.legend(fontsize=FS, loc="upper right", handlelength=1.2)

# (b) coexistence P(E|T_c) at the largest L
Lhi = 18
E, p, Estar = pe_and_saddle(Lhi)
epa = (E - Estar) / NAT[Lhi] * 1000.0
pn = p / p.max()
axPE.fill_between(epa, pn, color="0.85", lw=0)
axPE.plot(epa, pn, color="0.25", lw=LW)
pk, _ = find_peaks(pn, height=0.05)
top2 = pk[np.argsort(pn[pk])[-2:]]
pts = sorted((epa[i], pn[i]) for i in top2)   # (x, height) left peak then right peak
for (xc, yc), lab in zip(pts, ("chiral", "achiral")):
    axPE.text(xc, yc + 0.08, lab, ha="center", fontsize=FS, color="black")
axPE.set_xlim(-5, 4); axPE.set_ylim(0, 1.28)
axPE.set_xlabel(r"$(E-E^*)/N$  (meV/atom)")
axPE.set_ylabel(r"$P(E\,|\,T_c)$  (norm.)")
axPE.set_title("(b)")

# (c) chirality Binder vs T -- L=6,9,12 only; the deeper dips at L=15,18 would
# compress the axis and hide the smaller sizes, and the trend is already clear here.
for L in (6, 9, 12):
    axBinder.plot(Tgrid(L), seedavg(L, "chirality_binder"), color=Lcol[L], lw=LW, label=f"$L={L}$")
axBinder.axhline(0, color=T["grey"], lw=RLW)
axBinder.set_yscale("symlog", linthresh=1.0)
axBinder.set_xlim(300, 600)
axBinder.set_ylim(-100, 2)
axBinder.set_xlabel(r"$T$  (K)")
axBinder.set_ylabel(r"$|\chi|$ Binder  $U_L$")
axBinder.set_title("(c)")
axBinder.legend(fontsize=FS, loc="lower left", handlelength=1.2, labelspacing=0.25,
                borderaxespad=0.3)

# (d) T_c(L) vs 1/L^3 -- two estimators each with a 1/L^3 fit to a common intercept
x = np.array([1 / L**3 for L in Lseq])
for L in Lseq:
    # seed-SE bars (~0.1-0.4 K) are far smaller than the marker on this axis
    axFSS.plot(1 / L**3, TC[L], "o", ms=4.5, color=Lcol[L], markeredgewidth=0, zorder=3)
xx = np.linspace(0, x.max() * 1.08, 50)
bea = np.polyfit([1 / L**3 for L in Lseq], [TC[L] for L in Lseq], 1,
                 w=[np.sqrt(5) / TC_STD[L] for L in Lseq])
axFSS.plot(xx, np.polyval(bea, xx), "-", color=T["grey"], lw=RLW, zorder=1)
Lbind = [L for L in Lseq if L in BINDERT]
for L in Lbind:
    axFSS.plot(1 / L**3, BINDERT[L], "D", ms=3.8, color=Lcol[L], markeredgewidth=0, zorder=3)
bbm = np.polyfit([1 / L**3 for L in Lbind], [BINDERT[L] for L in Lbind], 1)
axFSS.plot(xx, np.polyval(bbm, xx), "-", color=T["grey"], lw=RLW, zorder=1)
# thermodynamic T_c: horizontal reference the two estimators extrapolate to
axFSS.axhline(bea[1], ls="--", color="black", lw=RLW, zorder=1)
axFSS.text(0.5 * xx.max(), bea[1] + 3, rf"$T_c(\infty) = {bea[1]:.0f}$ K",
           ha="center", va="bottom", fontsize=FS, color="black")
axFSS.set_xlim(0, xx.max())
axFSS.set_ylim(350, 525)
axFSS.set_xlabel(r"$1/L^3$")
axFSS.set_ylabel(r"$T_c(L)$  (K)")
axFSS.set_title("(d)")
axFSS.legend(handles=[Line2D([], [], marker="o", ls="none", color=T["grey"], ms=4.5, markeredgewidth=0, label="equal weight"),
                      Line2D([], [], marker="D", ls="none", color=T["grey"], ms=3.8, markeredgewidth=0, label="Binder min")],
             fontsize=FS, loc="lower left", handlelength=1.0)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_finite_temperature.{ext}", dpi=300)
print("wrote", HERE / "fig_finite_temperature.png")
