# Structures

This directory contains the structure files the analysis reads: the enumerated
anion orderings, reference settings of the ground state, and the published
structures of &beta;-Fe<sub>2</sub>SeO, BaZnO<sub>2</sub> and BaCoO<sub>2</sub>.

- `orbit_representatives/`: One representative of each of the twelve
  symmetry-distinct anion orderings that satisfy both ordering rules (cis
  coordination everywhere and period-three ordering along every chain) in the
  3&times;3&times;3 supercell. These twelve exhaust the possibilities on this
  cell, so they are the complete candidate set from which the ground state is
  identified.

  Filenames encode `orbit_<index>_sg<number>_<Hermann-Mauguin symbol>_stab<n>`,
  where the space group is that of the anion ordering and `stab` is the size of
  its stabiliser under the symmetry of the parent cell. `orbit_11`, of space
  group *P*3<sub>1</sub>21, is the ground state and the maximally symmetric of the
  twelve. `MANIFEST.txt` lists all twelve with their orbit sizes, stabilisers,
  space groups and whether each is chiral.

- `reference_cifs/`: The *P*3<sub>1</sub>21 ground state as a primitive cell
  (`p3121_primitive.cif`) and in the cubic 3&times;3&times;3 setting used
  throughout the analysis (`p3121_cubic_3x3x3.cif`).

- `beta_fe2seo/`: The single-crystal structure of &beta;-Fe<sub>2</sub>SeO
  refined at 60, 110 and 293 K, transcribed from Shamova et al. \[1\]: cell
  parameters from Supporting Information Table S1a, atomic coordinates from
  Table S1b. &beta;-Fe<sub>2</sub>SeO is an antiperovskite in which one third of
  the iron sites are vacant. Its iron sites are the vertices of corner-sharing
  O-centred octahedra, the same network as the anion sites of
  NbO<sub>2</sub>F, so its iron/vacancy order can be compared directly with the
  twelve orbits (`workflow/structures/fe2seo_orbit_match.py`).

- `bazno2_type/`: The single-crystal structure of BaZnO<sub>2</sub> from von
  Schnering et al. \[2\], from the Crystallography Open Database (COD 1540010).
  BaZnO<sub>2</sub> and BaCoO<sub>2</sub> are oxygen-deficient perovskites in which
  one third of the anion sites are vacant. Their B (Zn or Co) and anion sites form
  the same network as the Nb and anion sites of NbO<sub>2</sub>F, so their
  oxide/vacancy order can be compared directly with the twelve orbits
  (`workflow/structures/bazno2_orbit_match.py`). The BaCoO<sub>2</sub> structure,
  refined from neutron powder data at 400 K by Diatta et al. \[3\] (CCDC 2313333),
  is built in that script from its published space group, cell and coordinates.

## References

1. I. K. Shamova et al., &ldquo;Two Ordered Magnetic States of
   &beta;-Fe<sub>2</sub>SeO Characterized by Spin Exchange Analysis and by X-ray
   Diffraction, Torque and Magnetization Measurements&rdquo;, *Chem. Mater.*
   **38**, 3727 (2026). doi:[10.1021/acs.chemmater.6c00310](https://doi.org/10.1021/acs.chemmater.6c00310).
2. H. G. von Schnering, R. Hoppe and J. Zemann, *Z. Anorg. Allg. Chem.* **305**,
   241 (1960).
3. Diatta et al., *J. Am. Chem. Soc.* **146**, 15027 (2024).
   doi:[10.1021/jacs.3c14047](https://doi.org/10.1021/jacs.3c14047).
