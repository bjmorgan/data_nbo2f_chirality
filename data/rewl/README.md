# Replica-exchange Wang&ndash;Landau simulations

This directory contains the densities of states and structural observables from
the replica-exchange Wang&ndash;Landau simulations, for supercells of side
*L* = 6, 9, 12, 15 and 18 (containing 4*L*<sup>3</sup> atoms), with five
independent random seeds at each size.

Each run is one directory, `L<L>/prod/seed<n>/`, containing:

- `config.yaml`: The run input &mdash; the cluster expansion used, the energy
  windows, the Wang&ndash;Landau schedule, the trial-move set and the random seed.
  `docs/running-a-simulation.md` describes how to run from one of these.
- `stitched_dos.csv`: The density of states. Two columns, `energy` (eV, total for
  the cell) and `entropy` (ln *g*(*E*), to an arbitrary additive constant). The
  per-window estimates from the walkers have been merged on their overlaps into a
  single curve.
- `measure/`: A second, shorter simulation run at the converged density of
  states, sampling structural order parameters. `config.yaml` is its input and
  `stitched_dos.csv` is the density of states it was run at (identical to the one
  above).
- `measure/observables/ChainOrder.csv`: Microcanonical moments of the order
  parameters. One row per energy bin, with `energy`, the number of samples in
  that bin (`count`), and for each order parameter the running sums of its first,
  second and fourth powers (`<name>_sum`, `_sum2`, `_sum4`).

The moments are sums over samples, not averages: dividing by `count` gives
&lang;*O*&rang; within a bin. Thermal averages come from reweighting these against
the density of states, which is the first step of the analysis workflow.

The order parameters are `chirality` (the chiral pseudoscalar), `icoh_global` and
`icoh_nn` (inter-chain phase coherence, global and nearest-neighbour), `oof_amp`
(period-three ordering amplitude), `cis_frac` and `nbo4f2_frac` (local
coordination fractions), `collinear_ff`, `circ_coherence`, and `chi_11`,
`closest_chi` and `closest_sim` (similarity to the enumerated orbits). Their
definitions are in `chainorder` and `nbo2f_analysis`.
