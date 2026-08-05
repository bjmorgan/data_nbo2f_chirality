"""cis-NbO4F2 labelling-rule legend -- main-text Fig. 2(a).

One cis-NbO4F2 octahedron per fluoride-free axis (x, y, z), all in the same
orientation so only the fluoride positions move between them. The cation is
coloured by its fluoride-free axis (blue = x, gold = y, green = z); the two cis
fluorides are black, the four oxides white. The rule reads off directly: the
coloured axis is the one whose two neighbours are both white (oxide).

A clean base for hand annotation (the axis arrows / labels are added by hand).
Rendered with hofmann via a forced oblique projection (x horizontal, z vertical,
y a foreshortened diagonal) -- see `_oblique_rotation`. CELL_COLOUR must match
fig_orbit_gallery.CELL_COLOUR.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from hofmann import StructureScene, BondSpec, AtomStyle, ViewState

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                     # repository root
OUT = REPO / "figures"                    # rendered figures
A = 3.9
CELL_COLOUR = {0: "#a2d8e6", 1: "#dcb96f", 2: "#58b787"}   # x blue, y gold, z green
F_COLOUR, O_COLOUR = (0.1, 0.1, 0.1), "white"
NB_RADIUS, ANION_RADIUS = 0.62, 0.42
ATOM_SCALE = 0.85
OUTLINE_PT = 0.8
OBLIQUE_DEG = 35.0                   # depth axis (y) drawn this far above horizontal
OBLIQUE_F = 0.6                      # depth foreshortening (cavalier 1.0, cabinet 0.5)
BOND_MAX = 2.3                        # Nb-X cutoff (A); nearest Nb-X = a/2 = 1.95
_E = np.eye(3)


def _oblique_rotation() -> np.ndarray:
    """Shear matrix used as ``ViewState.rotation`` to get an oblique projection
    that hofmann lacks natively: x -> horizontal, z -> vertical, y -> a
    foreshortened diagonal. ``ViewState.project`` does ``coords @ R.T`` then
    takes columns 0-1, so a non-orthonormal R yields a cabinet/cavalier view;
    the third row is the depth, used only for sphere/bond ordering.
    """
    th = np.deg2rad(OBLIQUE_DEG)
    return np.array([[1.0, OBLIQUE_F * np.cos(th), 0.0],
                     [0.0, OBLIQUE_F * np.sin(th), 1.0],
                     [0.0, -1.0, 0.0]])


def octahedron(a: int) -> Atoms:
    """cis-NbO4F2 whose fluoride-free axis is `a`: the two cis F on the other
    two +axes, oxide on the remaining four octahedral positions."""
    others = [b for b in range(3) if b != a]
    f_dirs = [_E[others[0]], _E[others[1]]]                    # cis fluorides
    o_dirs = [-_E[others[0]], -_E[others[1]], _E[a], -_E[a]]   # oxides
    pos = [np.zeros(3)] + [d * (A / 2) for d in f_dirs + o_dirs]
    return Atoms(symbols=["Nb", "F", "F", "O", "O", "O", "O"], positions=pos)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6, 2.2))
    for ax, a in zip(axes, (0, 1, 2)):
        scene = StructureScene.from_ase(
            octahedron(a),
            [BondSpec(species=("Nb", "F"), max_length=BOND_MAX),
             BondSpec(species=("Nb", "O"), max_length=BOND_MAX)],
            atom_styles={"Nb": AtomStyle(radius=NB_RADIUS, colour=CELL_COLOUR[a]),
                         "F": AtomStyle(radius=ANION_RADIUS, colour=F_COLOUR),
                         "O": AtomStyle(radius=ANION_RADIUS, colour=O_COLOUR)})
        view = ViewState()
        view.rotation = _oblique_rotation()
        view.centre = np.zeros(3)
        scene.view = view
        scene.render_mpl(ax=ax, atom_scale=ATOM_SCALE, bond_colour="white",
                         half_bonds=False, atom_outline_width=OUTLINE_PT,
                         bond_outline_width=OUTLINE_PT, show_axes=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_p3121_legend.{ext}", dpi=300)
    print("wrote", HERE / "fig_p3121_legend.pdf")


if __name__ == "__main__":
    main()
