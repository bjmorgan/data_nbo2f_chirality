"""ReO3-type parent of NbO2F -- main-text Fig. 1.

The achiral Pm-3m parent: a cubic framework of corner-sharing MX6 octahedra,
the two anion species sharing the single crystallographic anion site. Rendered
as a finite 2x2x2 block of NbX6 octahedra (exactly eight, no periodic images)
with hofmann. The octahedra are a neutral grey to mark the unordered parent;
the fluoride-free-axis colours (blue/gold/green) are reserved for the ordered
structures (Fig. 2, SI gallery), so colour only ever appears where there is
ordering.

No input file -- the ReO3 cluster is constructed here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ase import Atoms
from hofmann import StructureScene, BondSpec, PolyhedronSpec, AtomStyle, ViewState

HERE = Path(__file__).resolve().parent
A = 3.9                               # cubic lattice parameter (A)
OCTA_COLOUR = "#9bb8d3"               # neutral grey octahedra (unordered parent)
ANION_COLOUR = "#d98b8b"             # anion (mixed O/F site, single colour)
NB_COLOUR = "#888888"
ANION_RADIUS, NB_RADIUS = 0.34, 0.6
ATOM_SCALE = 0.8
EDGE_PT = 1.0                         # shared weight: polyhedra edges == atom outlines
VIEW_DIR = [0.19, 0.13, 1.0]         # near face-on, slightly off the c axis
BOND_MAX = 2.3                        # Nb-X cutoff (A); nearest Nb-X = a/2 = 1.95


def reo3_block() -> Atoms:
    """Finite 2x2x2 ReO3 cluster: 8 cations plus the anions completing each MX6.

    A roomy cell with ``pbc=True`` gives the a/b/c axes widget a lattice without
    moving any atom (all sit inside the box); rendering with ``pbc=False`` then
    suppresses periodic image expansion, so exactly eight octahedra are drawn.
    """
    nb = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    offs = [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0),
            (0, -0.5, 0), (0, 0, 0.5), (0, 0, -0.5)]
    an = sorted({(i + dx, j + dy, k + dz)
                 for (i, j, k) in nb for (dx, dy, dz) in offs})
    pos = (np.array([*nb, *an], dtype=float) * A) + 5.0     # shift inside the box
    return Atoms(symbols=["Nb"] * len(nb) + ["O"] * len(an), positions=pos,
                 cell=[6 * A, 6 * A, 6 * A], pbc=True)


def main() -> None:
    scene = StructureScene.from_ase(
        reo3_block(), [BondSpec(species=("Nb", "O"), max_length=BOND_MAX)],
        polyhedra=[PolyhedronSpec(centre="Nb", colour=OCTA_COLOUR, alpha=1.0,
                                  edge_colour=(0.2, 0.2, 0.2), edge_width=EDGE_PT,
                                  hide_centre=True, hide_bonds=True)],
        atom_styles={"O": AtomStyle(radius=ANION_RADIUS, colour=ANION_COLOUR),
                     "Nb": AtomStyle(radius=NB_RADIUS, colour=NB_COLOUR)})
    scene.view = ViewState().look_along(VIEW_DIR)
    fig = scene.render_mpl(
        None, figsize=(5, 5), show=False, atom_scale=ATOM_SCALE,
        atom_outline_width=EDGE_PT, pbc=False, show_cell=False, show_axes=True)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_reo3_parent.{ext}", dpi=300)
    print("wrote", HERE / "fig_reo3_parent.pdf")


if __name__ == "__main__":
    main()
