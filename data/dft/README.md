# Density-functional calculations

This directory contains the VASP calculations of anion orderings in
NbO<sub>2</sub>F, in two sets that serve different purposes.

- `3x3x3/`: 92 configurations in the 3&times;3&times;3 supercell, spanning random
  anion placements, period-three orderings with random inter-chain phase, and the
  twelve enumerated cis + OOF orbit representatives. These establish that the two
  ordering rules are energetically active, and are the set from which the
  order-parameter analysis is built.
- `6x6x6/`: 96 configurations in the 6&times;6&times;6 supercell, the training set
  for the cluster expansion. Directory names record how each was obtained:
  `P3121_tiled` is the tiled ground state, `buss_*` and `config_*` were proposed
  by the active-learning search described in
  `docs/cluster-expansion-workflow.md`.

The two sets use different *k*-point meshes and their energies are not directly
comparable with each other.

Each configuration directory holds:

- `INCAR`, `KPOINTS`: The calculation parameters.
- `POSCAR.init`: The ideal, undistorted starting structure &mdash; the anion
  ordering itself, and the structure the order parameters are computed from.
- `POSCAR`: The relaxed structure.
- `OUTCAR`: The calculation output, from which the energies are read.
- `vaspmeta.yaml`: Provenance notes for the calculation.
- `POTCAR.spec`: The pseudopotentials used, one per species, with their hashes.
  The `POTCAR` files themselves are licensed by VASP and are not redistributed;
  this records which to obtain in order to repeat the calculations.
