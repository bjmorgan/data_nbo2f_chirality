"""SI: the cluster expansion against DFT on the low-energy manifold.

CE energy against DFT energy, in meV/atom above P3_121, for three sets, each
referenced to P3_121 within its own set of DFT calculations:
  - the eleven orbit representatives other than P3_121 (3x3x3 cells), none of
    which is in the training set;
  - the 6x6x6 training structures within 2.5 meV/atom of the tiled P3_121, with
    their held-out (leave-one-out) predictions;
  - the fifteen achiral 3x3x6 classes, relative to the helix in the same cell;
    one (stack_010101) is in the training set as a 6x6x6 cell.
Data from build/ce/ce_vs_dft_low_energy.csv, which also holds the posterior
standard deviation of each CE energy difference; it is not plotted, because the
CE energy of each structure is a single deterministic value.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from figure_formatting.figure_formatting import set_formatting, tableau

set_formatting()
import matplotlib as mpl  # noqa: E402

mpl.rcParams["mathtext.fontset"] = "dejavusans"
T = tableau

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "build" / "ce" / "ce_vs_dft_low_energy.csv"
OUT = REPO / "figures"
LW, FS, MS = 0.7, 7, 3.5
LIM = (-0.05, 2.5)

df = pd.read_csv(DATA)
df = df[~((df.set == "orbits") & (df.orbit == 11))]          # the reference, zero by definition

fig, ax = plt.subplots(figsize=(3.3, 3.1))
ax.plot(LIM, LIM, color="0.6", lw=LW, ls="--", zorder=0)

train = df[df.set == "training"]
ax.plot(train.dft_meV_per_atom, train.ce_meV_per_atom, "s", ms=MS, mfc="white", mec=T["grey"], mew=LW,
        label="6×6×6 training structures", zorder=2)
ax.plot(train.dft_meV_per_atom, train.ce_held_out_meV_per_atom, "x", ms=MS * 0.9, mew=LW, color=T["grey"],
        label="   held-out predictions", zorder=2)

s36 = df[df.set == "3x3x6"]
for inside, mfc, lab in ((False, T["green"], "3×3×6 achiral classes"), (True, "white", "   in training (010101)")):
    g = s36[s36.in_training == inside]
    ax.plot(g.dft_meV_per_atom, g.ce_meV_per_atom, "^", ms=MS + 0.5, mfc=mfc, mec=T["green"], mew=LW,
            label=lab, zorder=2)

orb = df[df.set == "orbits"]
ax.plot(orb.dft_meV_per_atom, orb.ce_meV_per_atom, "o", ms=MS + 0.5, mfc=T["blue"], mec=T["blue"], mew=LW,
        label="orbits 0–10 (not in training)", zorder=3)

ax.set_xlim(LIM)
ax.set_ylim(LIM)
ax.set_aspect("equal")
ax.set_xlabel(r"$E_\mathrm{DFT}$ above P3$_1$21 (meV/atom)", fontsize=FS)
ax.set_ylabel(r"$E_\mathrm{CE}$ above P3$_1$21 (meV/atom)", fontsize=FS)
ax.tick_params(labelsize=FS)
handles, labels = ax.get_legend_handles_labels()
order = ["orbits 0–10 (not in training)", "3×3×6 achiral classes", "   in training (010101)",
         "6×6×6 training structures", "   held-out predictions"]
ax.legend([handles[labels.index(o)] for o in order], order, fontsize=FS - 1, frameon=False,
          loc="upper left", handletextpad=0.3, borderaxespad=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_si_ce_vs_dft.{ext}", dpi=300)
print("wrote", OUT / "fig_si_ce_vs_dft.pdf")
