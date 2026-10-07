"""SI: DFT energy of the twelve cis + OOF orbits against their side-by-side fluoride pairs (NbO2F).

DFT energy in meV/atom above P3_121 (one point per orbit, the lower-energy setting where
an orbit was computed in two, as in fig_dft_coherence) against the number of fluoride
pairs side by side on neighbouring parallel chains in the 3x3x3 cell, which fixes the
point-charge energy. The line is the least-squares fit, with its R^2 on the panel.
Points are coloured by inter-chain coherence.

Reads build/dft/orbit_electrostatics.csv and build/dft/dft_ops.csv. Fonts are embedded
as TrueType (Type 42).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_formatting.figure_formatting import set_formatting

set_formatting()
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica Neue"
plt.rcParams["pdf.fonttype"] = 42
LW, FS, SYM = 0.7, 8.5, 12   # line width, fontsize, symbol size
plt.rcParams.update({k: FS for k in ("font.size", "axes.labelsize", "axes.titlesize",
                                     "xtick.labelsize", "ytick.labelsize")})
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "figures"
ELECTROSTATICS = REPO / "build" / "dft" / "orbit_electrostatics.csv"
DFT_OPS = REPO / "build" / "dft" / "dft_ops.csv"
for f in (ELECTROSTATICS, DFT_OPS):
    if not f.exists():
        raise SystemExit(f"missing {f}\nrun  make  first")

ops = pd.read_csv(DFT_OPS)
orbits = ops[ops.cls == "orbit"].sort_values("energy_per_atom").drop_duplicates("orbit")
df = pd.read_csv(ELECTROSTATICS).merge(orbits[["orbit", "icoh"]], on="orbit", validate="one_to_one")
assert len(df) == 12
x, y = df.side_by_side_pairs, df.dft_meV_per_atom
slope, intercept = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2

fig, ax = plt.subplots(figsize=(3.7, 2.6))
xs = np.array([-1, x.max() + 1])
ax.plot(xs, intercept + slope * xs, color="0.6", lw=LW, ls="--", zorder=1)
sc = ax.scatter(x, y, c=df.icoh, cmap="viridis", vmin=0, vmax=1, s=SYM * 1.5, edgecolors="none", zorder=3)
gs = df[df.orbit == 11].iloc[0]
ax.annotate(r"$\mathrm{P3_121}$", (gs.side_by_side_pairs, gs.dft_meV_per_atom),
            textcoords="offset points", xytext=(4, 2), ha="left", fontsize=FS)
ax.text(0.04, 0.96, f"R² = {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FS)
ax.set_xlim(xs)
ax.set_xticks(range(0, int(x.max()) + 1, 3))
ax.set_xlabel("side-by-side F pairs per 3×3×3 cell")
ax.set_ylabel("energy  (meV/atom)")
cb = fig.colorbar(sc, ax=ax, pad=0.03, aspect=25)
cb.set_label("inter-chain coherence", labelpad=4)
cb.outline.set_linewidth(LW)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_si_electrostatics.{ext}", dpi=300)
print(f"fit: slope {slope:.4f} meV/atom per pair, intercept {intercept:+.3f} meV/atom, R^2 {r2:.3f}")
print("wrote", OUT / "fig_si_electrostatics.pdf")
