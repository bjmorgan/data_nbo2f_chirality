"""DFT energetics of the anion orderings (NbO2F, 3x3x3 demonstration; sec:dft).

(a) DFT energy by ensemble -- the two ordering energy scales: a large gain on
    period-3 OOF ordering, a small further gain on cis-locking.
(b) DFT energy vs inter-chain coherence among the cis+OOF orbits -- energy
    falls with coherence, selecting the maximally-coherent P3_121 ground state.

Reads dft_calculations/dft_ops.csv (built by dft_ops.py); no hard-coded values.
Built on figure_formatting.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_formatting.figure_formatting import set_formatting, tableau

set_formatting()
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica Neue"
T = tableau
HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "build" / "dft" / "dft_ops.csv"
if not CSV.exists():
    raise SystemExit(f"missing {CSV}\nrun  make  first")
df = pd.read_csv(CSV)
df["dE"] = (df["energy_per_atom"] - df["energy_per_atom"].min()) * 1000.0  # meV/atom above the GS
LW, RLW, FS, SYM = 0.7, 0.7, 8, 12   # line width, fontsize (matches panel titles), symbol size

CLS = [("random", "random", T["grey"]),
       ("OOF-random-phase", "OOF only", T["blue"]),
       ("orbit", "cis + OOF", T["red"])]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.7, 2.05))
for ax in (axA, axB):
    ax.set_box_aspect(1)
    ax.locator_params(nbins=5)

# (a) energy by ensemble + the two energy scales
rng = np.random.default_rng(0)
med = {}
for xi, (cls, lab, c) in enumerate(CLS):
    e = df[df.cls == cls]["dE"].to_numpy()
    axA.scatter(xi + rng.uniform(-0.16, 0.16, len(e)), e, s=SYM, color=c, edgecolors="none")
    med[cls] = float(np.median(e))
    axA.plot([xi - 0.28, xi + 0.28], [med[cls], med[cls]], color="black", lw=RLW, zorder=4)
dE_oof = med["random"] - med["OOF-random-phase"]
dE_cis = med["OOF-random-phase"] - med["orbit"]
axA.text(0.97, 0.97, f"OOF ≈ {dE_oof:.0f} meV\ncis ≈ {dE_cis:.0f} meV",
         transform=axA.transAxes, ha="right", va="top", fontsize=FS)
axA.set_xticks([0, 1, 2])
axA.set_xticklabels([c[1] for c in CLS], fontsize=FS, rotation=20, ha="right")
axA.set_xlim(-0.5, 2.5)
axA.set_ylabel("energy  (meV/atom)")
axA.set_title("(a)   energy by ensemble")

# (b) energy vs inter-chain coherence among the orbits
o = df[df.cls == "orbit"].drop_duplicates("orbit")   # one point per symmetry orbit (drops enantiomer duplicates)
axB.scatter(o["icoh"], o["dE"], s=SYM, color=T["red"], edgecolors="none", zorder=3)
gs = o.loc[o["dE"].idxmin()]   # P3_121: lowest energy, coherence = 1 (same symbol)
axB.annotate(r"$\mathrm{P3_121}$", (gs["icoh"], gs["dE"]),
             textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=FS)
axB.set_xlabel("inter-chain coherence")
axB.set_ylabel("energy  (meV/atom)")
axB.set_title("(b)   energy vs coherence")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_dft_coherence.{ext}", dpi=300)
print("wrote", HERE / "fig_dft_coherence.png")
