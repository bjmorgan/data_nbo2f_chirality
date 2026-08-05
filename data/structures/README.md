# Structures

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
