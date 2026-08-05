# Cluster expansion

This directory contains the cluster expansion &mdash; the lattice model of the
configurational energy that the Wang&ndash;Landau simulations sample &mdash; and
the training set it was fitted to.

- `paircut9_5_5_ardr_n96.ce`: The fitted cluster expansion, in
  [icet](https://icet.materialsmodeling.org)'s format; read it with
  `icet.ClusterExpansion.read`. The model is defined on the cubic ReO<sub>3</sub>-type
  NbF<sub>3</sub> reference lattice with niobium held fixed and the three anion
  sites mixing oxygen and fluorine, with pair, triplet and quadruplet cutoffs of
  9, 5 and 5 &#8491; (41 orbits, of which 19 have non-zero effective cluster
  interactions).
- `training_set_refitB_n96.json`: The 96 density-functional calculations the
  model was fitted to. A JSON list, one entry per structure, with `path` (the
  calculation directory under `data/dft/6x6x6/`), `energy_per_atom` (eV) and
  `relaxed geometry` (the relaxed structure, ASE-JSON encoded).

`methods/cluster_expansion/` contains the code that produced both: how the
training set was assembled, how the hyperparameters were chosen, and the fit
itself. Refitting from this training set reproduces the deposited model.
