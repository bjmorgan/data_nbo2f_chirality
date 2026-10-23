"""Identify the oxide/vacancy order of BaZnO2 and BaCoO2 with one of the twelve orbits.

BaZnO2 and BaCoO2 are oxygen-deficient perovskites ABO2: Ba on the A site, Zn or Co on
the B site, and one third of the anion (X) sites vacant. Their B sites and anion sites
form the same network as the Nb and anion sites of ReO3-type NbO2F, with O and
vacancies in place of O and F in the same 2:1 ratio. This script maps each published
refinement onto the NbO2F labelling and compares it with the twelve cis + OOF orbit
representatives. The refinements are BaZnO2 from single-crystal data (von Schnering, Hoppe and Zemann,
Z. Anorg. Allg. Chem. 305, 241 (1960); COD 1540010, in data/structures/bazno2_type) and
BaCoO2 from neutron powder data at 400 K (Diatta et al., J. Am. Chem. Soc. 146, 15027
(2024); CCDC 2313333), which is built here from its published space group, cell and
coordinates.

It uses the routine of fe2seo_orbit_match.py, with the simple-cubic network fitted to
the Ba sublattice: the ideal B sites are the cube centres, and each of their six
vertices (ideal X sites, midway between neighbouring B sites) is tested for O within a
quarter of the pseudo-cubic lattice constant. A vertex with no O is vacant. The B
atoms are displaced well off the cube centres, towards their four O, so the network is
fitted to Ba rather than to B.

Writes build/structures/bazno2_orbit_match.csv, one row per refinement: compound,
source, database id and published space group; the pseudo-cubic lattice constants and the
angles between the cube axes; the largest B offset from its ideal site, the largest O
offset from its ideal X site, and the shortest distance from any O to a vacant X site;
whether every B has two vacancies in cis positions, whether the vacancies repeat every
three cubic cells and whether every chain holds one vacancy per three sites; and the
matched orbit and its space group.

    python bazno2_orbit_match.py
"""
from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.spacegroup import crystal

from fe2seo_orbit_match import ORBITS, canonical, labels, minority_sites

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CIFS = REPO / "data" / "structures" / "bazno2_type"
OUT = REPO / "build" / "structures" / "bazno2_orbit_match.csv"


def cif_field(text: str, tag: str) -> str:
    m = re.search(rf"^{tag}\s+(.+)$", text, re.M)
    return m.group(1).strip().strip("'") if m else ""


def first_author(text: str) -> str:
    m = re.search(r"^_publ_author_name\s*\n\s*'([^',]+)", text, re.M)
    return m.group(1) if m else ""


def bacoo2_diatta() -> Atoms:
    """BaCoO2 at 400 K (Diatta et al. 2024, CCDC 2313333): P3_121, Ba on 3b, Co on 3a, O on 6c."""
    a = crystal(["Ba", "Co", "O"], basis=[(0.6584, 0.0, 5 / 6), (0.584, 0.0, 1 / 3), (0.4322, 0.1295, 0.1259)],
                spacegroup=152, cellpar=[5.8778, 5.8778, 6.7508, 90, 90, 120])
    assert a.get_chemical_formula() == "Ba3Co3O6"
    return a


def refinements() -> list[dict]:
    """Each refinement: compound, source, database id, published space group and structure."""
    out = []
    for f in sorted(CIFS.glob("COD*.cif")):
        text = f.read_text()
        atoms = read(f)
        b = next(s for s in ("Zn", "Co") if s in atoms.get_chemical_symbols())
        out.append({"compound": f"Ba{b}O2", "source": f"{first_author(text)} {cif_field(text, '_journal_year')}",
                    "database_id": f"COD {cif_field(text, '_cod_database_code')}",
                    "published_space_group": cif_field(text, "_symmetry_space_group_name_H-M").replace(" ", ""),
                    "atoms": atoms, "B": b})
    out.append({"compound": "BaCoO2", "source": "Diatta 2024", "database_id": "CCDC 2313333",
                "published_space_group": "P3121", "atoms": bacoo2_diatta(), "B": "Co"})
    return out


def main() -> None:
    orbits = {}
    for f in sorted(ORBITS.glob("orbit_*.vasp")):
        tok = f.stem.split("_")
        m = minority_sites(read(f, format="vasp"), centre="Nb", majority="O")
        assert m["periodic"] and m["cis"] and m["chains"], f"{f.name} fails the ordering rules"
        orbits[canonical(labels(m["minority"]))] = (int(tok[1]), "_".join(tok[3:-1]))
    assert len(orbits) == 12

    rows = []
    for ref in refinements():
        b = ref["B"]
        m = minority_sites(ref["atoms"], centre=b, majority="O", frame="Ba")
        ok = m["periodic"] and m["cis"] and m["chains"]
        idx, hm = orbits.get(canonical(labels(m["minority"])), (None, "")) if ok else (None, "")
        lat = m["lattice"]
        length = np.linalg.norm(lat, axis=1)
        angles = [np.degrees(np.arccos(lat[i] @ lat[j] / (length[i] * length[j]))) for i, j in ((1, 2), (0, 2), (0, 1))]
        rows.append({
            **{k: ref[k] for k in ("compound", "source", "database_id", "published_space_group")},
            "a1_A": round(length[0], 3), "a2_A": round(length[1], 3), "a3_A": round(length[2], 3),
            "angle_23_deg": round(angles[0], 2), "angle_13_deg": round(angles[1], 2), "angle_12_deg": round(angles[2], 2),
            "max_B_offset_A": round(m["max_centre_offset"], 3),
            "max_O_offset_A": round(m["max_occupied_offset"], 3),
            "min_O_to_vacancy_A": round(m["min_vacant_clearance"], 3),
            "cis_everywhere": m["cis"],
            "period_three": m["periodic"],
            "one_vacancy_per_chain_period": m["chains"],
            "orbit": idx,
            "orbit_space_group": hm,
        })
        r = rows[-1]
        print(f"{r['compound']} ({r['source']}, {r['database_id']}): a' {length.round(3)} A, angles "
              f"{np.round(angles, 1)} deg; cis {m['cis']}, period three {m['periodic']}, chain rule {m['chains']} "
              f"-> orbit {idx} ({hm}); B within {r['max_B_offset_A']:.2f} A and O within {r['max_O_offset_A']:.2f} A "
              f"of their ideal sites, nearest O to a vacancy {r['min_O_to_vacancy_A']:.2f} A")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
