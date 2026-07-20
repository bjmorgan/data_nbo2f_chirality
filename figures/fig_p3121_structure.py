"""P3_121 ground-state structure -- main-text Fig. 2(c).

The cis + OOF P3_121 orbit representative (orbit 11) as the unrelaxed crystal,
viewed down <111> with only the Nb-F bonds drawn. Each cation is coloured by
its fluoride-free octahedral axis (blue = x, gold = y, green = z), matching the
fluoride-free-axis colouring of Fig. 2(a,b) and the SI gallery; the x->y->z
cycle from cell to cell out of the page traces the 3_1 screw about <111>.

Reads structures/orbit_representatives/orbit_11_*.vasp and renders with hofmann.
CELL_COLOUR must match fig_orbit_gallery.CELL_COLOUR.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors
from ase.io import read
from chainorder import SublatticeOccupation
from hofmann import StructureScene, BondSpec, AtomStyle, ViewState

HERE = Path(__file__).resolve().parent
REP = HERE.parent / "data" / "structures" / "orbit_representatives"
VASP = next(REP.glob("orbit_11_*.vasp"))
N = 3

CELL_COLOUR = {0: "#a2d8e6", 1: "#dcb96f", 2: "#58b787"}   # x blue, y gold, z green
NB_RADIUS, F_RADIUS = 0.62, 0.38
NB_PLACEHOLDER = (0.5, 0.5, 0.5)     # overwritten by the fluoride-free-axis colouring
F_COLOUR = "#cccccc"                 # light grey fluoride
ATOM_SCALE, BOND_SCALE = 0.95, 0.95
OUTLINE_PT = 0.7
VIEW_ROT_DEG = 30.0                  # in-plane anticlockwise rotation about <111>
BOND_MAX = 2.3                       # Nb-F cutoff (A); nearest Nb-X = a/2 = 1.95


def nb_free_axis_field(atoms) -> np.ndarray:
    """(N, N, N) fluoride-free axis (0=x, 1=y, 2=z) per cation, from the F sublattice."""
    occ = SublatticeOccupation.from_atoms(atoms, N=N, species="F").occupation
    has_F = np.stack([occ[d] | np.roll(occ[d], 1, axis=d) for d in range(3)])
    return np.argmax(has_F == 0, axis=0)


def main() -> None:
    orb = read(VASP)
    m = nb_free_axis_field(orb)
    sym = np.array(orb.get_chemical_symbols())
    nb_idx = list(np.where(sym == "Nb")[0])
    f_idx = list(np.where(sym == "F")[0])
    sub = orb[nb_idx + f_idx]                       # Nb then F; oxide dropped

    # per-cation fluoride-free axis from its lattice site; fluorides carry NaN
    frac = orb.get_scaled_positions()
    ff = [float(m[tuple(np.round(frac[i] * N).astype(int) % N)]) for i in nb_idx]
    ff += [np.nan] * len(f_idx)

    def ffcmap(t):                                  # 0 / 0.5 / 1 -> x / y / z colour
        return mcolors.to_rgb(CELL_COLOUR[int(round(t * 2))])

    scene = StructureScene.from_ase(
        sub, [BondSpec(species=("F", "Nb"), max_length=BOND_MAX)],
        atom_styles={"F": AtomStyle(radius=F_RADIUS, colour=F_COLOUR),
                     "Nb": AtomStyle(radius=NB_RADIUS, colour=NB_PLACEHOLDER)},
        atom_data={"ffree": ff})

    theta = np.deg2rad(VIEW_ROT_DEG)
    cs, sn = np.cos(theta), np.sin(theta)
    view = ViewState().look_along([1, 1, 1])
    view.rotation = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]) @ view.rotation
    scene.view = view

    fig = scene.render_mpl(
        None, figsize=(5, 5), show=False, colour_by="ffree", cmap=ffcmap,
        colour_range=(0, 2), atom_scale=ATOM_SCALE, bond_scale=BOND_SCALE,
        half_bonds=False, bond_colour="white",
        atom_outline_width=OUTLINE_PT, bond_outline_width=OUTLINE_PT)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_p3121_structure.{ext}", dpi=300)
    print("wrote", HERE / "fig_p3121_structure.pdf")


if __name__ == "__main__":
    main()
