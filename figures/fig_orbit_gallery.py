"""Anion-ordering patterns of the twelve chiral orbits (NbO2F, 3x3x3; SI gallery
+ main-text ground-state panel).

Each cation in an all-cis-all-OOF ordering has exactly one fluorine-free
octahedral axis (the "missing axis"); colouring every cation by that axis
(blue = x, green = y, gold = z) renders the whole ordering as three stacked
3x3 layers. The ground state P3_121 is the maximally coherent such pattern.

Two outputs from one primitive:
  fig_orbit_p3121      -- orbit 11 alone (main-text Fig. 2b).
  fig_orbit_gallery    -- all twelve orbits in ascending DFT energy, so the
                          ground state's stacking variants sit next to it.

Reads structures/orbit_representatives/*.vasp (the enumerated orbit reps) and
dft_calculations/dft_ops.csv (per-orbit DFT energy, for ordering and labels).
Nothing geometric is hard-coded: the missing axis is recovered from each
structure. Built on figure_formatting.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from ase.io import read
from chainorder import SublatticeOccupation
from figure_formatting.figure_formatting import set_formatting

set_formatting()
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica Neue"

HERE = Path(__file__).resolve().parent
REP = HERE.parent / "data" / "structures" / "orbit_representatives"
CSV = HERE.parent / "build" / "dft" / "dft_ops.csv"
COCSV = HERE.parent / "build" / "dft" / "orbit_chain_order.csv"
N = 3

# --- Mockup geometry (all adjustable). Lengths in mm, strokes in pt. ---
CELL_COLOUR = {0: "#a2d8e6", 1: "#dcb96f", 2: "#58b787"}   # 0=x blue, 1=y gold, 2=z green
EDGE = "#333333"                      # cell outline colour
DASH = "#333333"                      # inter-layer connector colour
STROKE_PT = 2.0                       # stroke weight (cells and connectors)
DASH_PT = 16.0                         # dash period (on + off), pt
CELL_W_MM = 58.5                      # one cell's width after skew (bounding-box extent)
CELL_H_MM = 14.6                      # one cell's height (foreshortened depth)
GAP_MM = 14.0                         # empty vertical gap between adjacent sheets
ANGLE = np.deg2rad(32.9)             # receding-edge angle above horizontal

# Oblique projection basis (mm): horizontal edge V, foreshortened receding edge U.
# The receding edge is both skewed AND shortened, so depth reads as foreshortened
# rather than full-length (cabinet- rather than cavalier-style). CELL_W/H are per
# cell; a sheet is N cells, so its vertical extent is N*CELL_H_MM.
U = np.array([CELL_H_MM / np.tan(ANGLE), CELL_H_MM])  # +i rows: receding, foreshortened
V = np.array([CELL_W_MM - U[0], 0.0])                 # +j cols: horizontal
DZ = N * CELL_H_MM + GAP_MM                            # sheet-to-sheet vertical rise

# Panel framing (mm), shared by both figures so every panel is the same physical
# size (1 mm = 1 data unit) -- the stroke weight then reads identically in the
# single panel and in each gallery cell.
MARGIN_MM, LABEL_MM = 6.0, 24.0       # surround; label strip (room for two lines)
LABEL_GAP_MM = 12.0                   # clear gap between a stack and its label
LABEL_PT = 30.0                       # main-panel label + legend font size (pt)
PAD_MM = 30.0                         # gallery inter-panel padding
# Gallery panels carry a third line (icoh) with slightly larger, more open text.
GAL_LABEL_MM = 50.0                   # taller strip: symmetry label + energy + icoh
GAL_LABEL_PT = 36.0
GAL_LINESPACING = 1.5
STACK_W = N * CELL_W_MM
STACK_H = N * CELL_H_MM + (N - 1) * DZ
PANEL_W = STACK_W + 2 * MARGIN_MM


def panel_h(label_mm: float) -> float:
    """Panel height (mm) for a given label-strip height; the stack size is fixed,
    so a taller strip changes only the space below the stack (strokes unchanged)."""
    return STACK_H + 2 * MARGIN_MM + LABEL_GAP_MM + label_mm


@dataclass(frozen=True)
class Orbit:
    """One enumerated orbit representative."""
    idx: int          # orbit index 0-11
    sg: int           # space-group number
    hm: str           # Hermann-Mauguin symbol, e.g. "P3_121"
    path: Path        # the .vasp representative


def parse_reps() -> list[Orbit]:
    """All orbit representatives, parsed from their filenames.

    Filenames are ``orbit_<idx>_sg<sg>_<HM>_stab<n>.vasp``; the HM symbol may
    itself contain underscores (``P3_121``), so it is everything between the
    ``sg`` and ``stab`` tokens.
    """
    orbits = []
    for f in sorted(REP.glob("orbit_*.vasp")):
        tok = f.stem.split("_")
        orbits.append(Orbit(idx=int(tok[1]), sg=int(tok[2][2:]),
                            hm="_".join(tok[3:-1]), path=f))
    return orbits


def missing_axis_field(path: Path) -> np.ndarray:
    """Per-cation fluorine-free axis of an all-cis-all-OOF structure.

    Returns an ``(N, N, N)`` int array indexed ``[i, j, k]`` whose values are
    the F-free axis (0=x, 1=y, 2=z) of the cation at corner ``(i, j, k)``. A
    cation has an F neighbour along axis ``d`` when either flanking edge
    midpoint carries F; the one axis with neither is its F-free axis. Raises if
    any cation does not have exactly one F-free axis (i.e. the structure is not
    cleanly all-cis), which would make the colouring ill-defined.
    """
    occ = SublatticeOccupation.from_atoms(read(path), N=N, species="F").occupation
    has_F = np.stack([occ[d] | np.roll(occ[d], 1, axis=d) for d in range(3)])
    free = has_F == 0
    if not np.all(free.sum(axis=0) == 1):
        raise ValueError(f"{path.name}: not cleanly all-cis (cation without a "
                         f"unique F-free axis)")
    return np.argmax(free, axis=0)


def draw_orbit_stack(ax, m: np.ndarray) -> None:
    """Draw one orbit as three stacked oblique 3x3 layers on ``ax``.

    ``m`` is the ``(N, N, N)`` missing-axis field from `missing_axis_field`;
    layer ``k`` shows ``m[:, :, k]``. The front-right corner's connector is
    drawn in front of the sheets (it is the one that passes over them); the
    other three sit behind. Callers set the view limits and add the label.
    """
    def corner(i, j, k):
        return i * U + j * V + np.array([0.0, k * DZ])

    dashes = (DASH_PT / 2.0, DASH_PT / 2.0)    # equal on/off -> DASH_PT period
    front = (0, N)                             # front-right corner -> in front
    for ci, cj in ((0, 0), (N, 0), (0, N), (N, N)):
        p0, p1 = corner(ci, cj, 0), corner(ci, cj, N - 1)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=DASH, lw=STROKE_PT,
                dashes=dashes, zorder=(N + 2) if (ci, cj) == front else 1)

    for k in range(N):                       # bottom (k=0) to top
        for i in range(N):
            for j in range(N):
                quad = [corner(i, j, k), corner(i + 1, j, k),
                        corner(i + 1, j + 1, k), corner(i, j + 1, k)]
                ax.add_patch(Polygon(quad, closed=True, facecolor=CELL_COLOUR[int(m[i, j, k])],
                                     edgecolor=EDGE, lw=STROKE_PT, joinstyle="round",
                                     zorder=2 + k))

    ax.set_aspect("equal")
    ax.axis("off")


def draw_panel(ax, m: np.ndarray, label: str, *, label_mm: float = LABEL_MM,
               fontsize: float = LABEL_PT, linespacing: float = 1.2) -> None:
    """Draw one orbit stack in the shared mm frame, with a label beneath it.

    The stack view limits are the same in every panel, so it is the same
    physical size (1 mm = 1 data unit) whether it is the lone main-text panel or
    one gallery cell -- keeping the stroke weight visually consistent. Only the
    label strip below the stack varies (``label_mm``/``fontsize``/``linespacing``),
    which the gallery enlarges to carry the extra icoh line.
    """
    draw_orbit_stack(ax, m)
    ax.set_xlim(-MARGIN_MM, STACK_W + MARGIN_MM)
    ax.set_ylim(-(MARGIN_MM + LABEL_GAP_MM + label_mm), STACK_H + MARGIN_MM)
    ax.text(STACK_W / 2, -(LABEL_GAP_MM + label_mm * 0.5), label, ha="center",
            va="center", fontsize=fontsize, linespacing=linespacing)


def dE_by_orbit() -> dict[int, float]:
    """Per-orbit DFT energy above the ground state, meV/atom, from dft_ops.csv."""
    df = pd.read_csv(CSV)
    o = df[df.cls == "orbit"]
    dE = (o.groupby("orbit")["energy_per_atom"].mean()
          - df.energy_per_atom.min()) * 1000.0
    return dE.to_dict()


def icoh_by_orbit() -> dict[int, float]:
    """Per-orbit inter-chain coherence (icoh_global) from orbit_chain_order.csv."""
    df = pd.read_csv(COCSV)
    return dict(zip(df["orbit"], df["icoh"]))


def hm_label(hm: str, dE: float, icoh: float) -> str:
    """Gallery panel label: Hermann-Mauguin symbol + energy + inter-chain coherence."""
    return (rf"$\mathrm{{{hm}}}$" + "\n"
            + rf"$\mathrm{{\Delta E}}$ = {dE:.1f} meV" + "\n"
            + f"icoh = {icoh:.2f}")


def main() -> None:
    orbits = parse_reps()
    fields = {ob.idx: missing_axis_field(ob.path) for ob in orbits}
    dE = dE_by_orbit()
    icoh = icoh_by_orbit()

    # --- main-text panel: P3_121 (highest space group) alone ---
    gs = max(orbits, key=lambda ob: ob.sg)
    figm = plt.figure(figsize=(PANEL_W / 25.4, panel_h(LABEL_MM) / 25.4))
    draw_panel(figm.add_axes((0, 0, 1, 1)), fields[gs.idx], rf"$\mathrm{{{gs.hm}}}$")
    for ext in ("png", "pdf"):
        figm.savefig(HERE / f"fig_orbit_p3121.{ext}", dpi=300)

    # --- SI gallery: all twelve in ascending DFT energy ---
    # Energy order (idx breaks ties) so the near-degenerate stacking variants of
    # the P3_121 ground state land adjacent to it. Same stack (and stroke) size as
    # the main-text panel; a taller label strip carries the per-orbit icoh line.
    # Tiled 3-wide x 4-tall with a legend strip below.
    PH = panel_h(GAL_LABEL_MM)
    order = sorted(orbits, key=lambda ob: (dE[ob.idx], ob.idx))
    nrow, ncol, legend_mm = 4, 3, 30.0
    fig_w = ncol * PANEL_W + (ncol - 1) * PAD_MM
    fig_h = nrow * PH + (nrow - 1) * PAD_MM + legend_mm
    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4))
    for idx, ob in enumerate(order):
        row, col = divmod(idx, ncol)
        ax = fig.add_axes((col * (PANEL_W + PAD_MM) / fig_w,
                           (legend_mm + (nrow - 1 - row) * (PH + PAD_MM)) / fig_h,
                           PANEL_W / fig_w, PH / fig_h))
        draw_panel(ax, fields[ob.idx], hm_label(ob.hm, dE[ob.idx], icoh[ob.idx]),
                   label_mm=GAL_LABEL_MM, fontsize=GAL_LABEL_PT,
                   linespacing=GAL_LINESPACING)
    handles = [Patch(facecolor=CELL_COLOUR[d], edgecolor=EDGE, lw=STROKE_PT, label=lab)
               for d, lab in ((0, "x"), (1, "y"), (2, "z"))]
    fig.legend(handles=handles, title="F-free axis", ncol=3, frameon=False,
               loc="center", bbox_to_anchor=(0.5, 0.5 * legend_mm / fig_h),
               fontsize=GAL_LABEL_PT, title_fontsize=GAL_LABEL_PT)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_orbit_gallery.{ext}", dpi=300)
    print("wrote", HERE / "fig_orbit_gallery.png", "and fig_orbit_p3121.png")


if __name__ == "__main__":
    main()
