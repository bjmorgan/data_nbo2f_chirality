"""SI: DFT energy of the twelve cis + OOF orbits against their side-by-side fluoride pairs (NbO2F).

DFT energy in meV/atom above P3_121 (one point per orbit, the lower-energy setting where
an orbit was computed in two, as in fig_dft_coherence) against the number of fluoride
pairs side by side on neighbouring parallel chains in the 3x3x3 cell, which fixes the
point-charge energy. The line is the least-squares fit, with its R^2 on the panel; the
axes are square, with limits that put the fit on the diagonal. The panel is the size of
each panel of fig_dft_coherence, with the same fonts, so at the same scaling the two match.

Reads build/dft/orbit_electrostatics.csv. Fonts are embedded as TrueType (Type 42); the
italic R is DejaVu Sans Oblique.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_formatting.figure_formatting import set_formatting, tableau

set_formatting()
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica Neue"
plt.rcParams["mathtext.it"] = "DejaVu Sans:italic"
plt.rcParams["pdf.fonttype"] = 42
LW, FS, SYM = 0.7, 8.5, 12   # line width, fontsize, symbol size
plt.rcParams.update({k: FS for k in ("font.size", "axes.labelsize", "axes.titlesize",
                                     "xtick.labelsize", "ytick.labelsize")})
LABELPAD = 8
BOX = 1.24                    # panel side (inches), as in fig_dft_coherence
MARGIN = 2                    # pairs beyond the data at each end of the x axis
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "figures"
ELECTROSTATICS = REPO / "build" / "dft" / "orbit_electrostatics.csv"
if not ELECTROSTATICS.exists():
    raise SystemExit(f"missing {ELECTROSTATICS}\nrun  make  first")

df = pd.read_csv(ELECTROSTATICS)
assert len(df) == 12
x, y = df.side_by_side_pairs, df.dft_meV_per_atom
slope, intercept = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2

# a fixed panel size; the saved page is trimmed to the content
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.9 / 3, 0.9 / 3, BOX / 3, BOX / 3])
xs = np.array([-MARGIN, x.max() + MARGIN])
ax.plot(xs, intercept + slope * xs, color="0.6", lw=LW, ls="--", zorder=1)
ax.scatter(x, y, color=tableau["red"], s=SYM, edgecolors="none", zorder=3)
gs = df[df.orbit == 11].iloc[0]
ax.annotate(r"$\mathrm{P3_121}$", (gs.side_by_side_pairs, gs.dft_meV_per_atom),
            textcoords="offset points", xytext=(5, -1), ha="left", va="top", fontsize=FS)   # below the fit
ax.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FS)
ax.set_xlim(xs)
ax.set_ylim(intercept + slope * xs)                # the fit runs corner to corner
ax.set_xticks(range(0, int(x.max()) + 1, 6))
ax.set_yticks(np.arange(0, 2.01, 0.5))           # as in fig_dft_coherence (b)
ax.set_xlabel("side-by-side F pairs\nper 3×3×3 cell", labelpad=LABELPAD)
ax.set_ylabel("energy  (meV/atom)", labelpad=LABELPAD)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_si_electrostatics.{ext}", dpi=300)
print(f"fit: slope {slope:.4f} meV/atom per pair, intercept {intercept:+.3f} meV/atom, R^2 {r2:.3f}")
print("wrote", OUT / "fig_si_electrostatics.pdf")
