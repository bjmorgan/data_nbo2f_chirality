# Cluster expansion

This directory contains the density-functional training set for the cluster
expansion, and the resulting model fitted to these data. The model is the lattice
description of the configurational energy that the Wang&ndash;Landau simulations
sample.

- `training_set_refitB_n96.json`: The 96 density-functional calculations the
  model was fitted to. A JSON list, one entry per structure, with `path` (the
  calculation directory under `data/dft/6x6x6/`), `energy_per_atom` (eV) and
  `relaxed geometry` (the relaxed structure, ASE-JSON encoded).
- `paircut9_5_5_ardr_n96.ce`: The fitted cluster expansion, in
  [icet](https://icet.materialsmodeling.org)'s format; read it with
  `icet.ClusterExpansion.read`. The model is defined on the cubic ReO<sub>3</sub>-type
  NbF<sub>3</sub> reference lattice with niobium held fixed and the three anion
  sites mixing oxygen and fluorine, with pair, triplet and quadruplet cutoffs of
  9, 5 and 5 &#8491; (41 orbits, of which 19 have non-zero effective cluster
  interactions).

Both files are committed, so that this directory holds the model the simulations
actually sampled, and both are also rebuilt by `make` from the calculations in
`data/dft/6x6x6/`: the training set is assembled from their `OUTCAR`s, and the
model is fitted to it. The rebuilt training set is byte-identical to the one
here. The rebuilt model has identical effective cluster interactions, but is not
byte-identical, because the file format records when it was written.

`methods/cluster_expansion/` contains the code that does this, together with the
hyperparameter selection and the active-learning loop that grew the training set
in the first place.
