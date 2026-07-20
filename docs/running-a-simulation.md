# Running a REWL simulation

How the REWL data was produced, and how to run or extend a run. This is the
compute-heavy upstream: a converged run's checkpoint reaches several GB and
takes many cluster-hours. Its outputs — the stitched density of states and the
observable moments — are the deposited data files, so the `make` figure pipeline
does not need this step. Run it to reproduce or redo the sampling itself.

## Inputs

- A run configuration, `data/rewl/L<n>/prod/seed<s>/config.yaml` (one per
  supercell size and seed). It sets the supercell size, the cluster expansion
  (`ce: paircut9_5_5_ardr_n96`), the energy windows, the Wang-Landau schedule,
  the move set, and a measurement block.
- The cluster-expansion model, `data/ce/paircut9_5_5_ardr_n96.ce`.

Beyond the figure-pipeline environment (`environment.yml`), running a simulation
also needs `nbo2f_analysis` (which provides the `rewl` driver) and
`mchammer-moves` (the custom moves the configuration references).

## Steps

Each run writes into its own directory. Working from a run directory such as
`data/rewl/L6/prod/seed1/` (or a copy of it):

1. **Production Wang-Landau run** — sample `ln g(E)` across the windows:

   ```sh
   rewl run config.yaml --out-dir .
   ```

   This writes the checkpoint `rewl_state.h5`. A single submission rarely
   converges; extend it (chained across cluster jobs) with:

   ```sh
   rewl resume config.yaml --out-dir .
   ```

2. **Stitch the density of states** — assemble the per-window `ln g` into one
   curve:

   ```sh
   mchammer-pt-stitch rewl_state.h5 -o stitched_dos.csv
   ```

   `stitched_dos.csv` is a deposited data file.

3. **Frozen-g measurement** — with the density of states converged, sample the
   structural observables at fixed `ln g`:

   ```sh
   rewl measure config.yaml --out-dir .
   mchammer-pt-stitch-observables meas_state.h5 -o observables/
   ```

   This produces `observables/ChainOrder.csv`, the deposited microcanonical
   observable moments. Reweighting these to canonical observables is the first
   `make` step (see the README).

## Cluster runs

The production runs were carried out on ARCHER2 via SLURM. The checkpoint is
written every `checkpoint.interval_cycles`, so a walltime hit leaves a resumable
state; the next submission continues it with `rewl resume`.

## Reproducibility

Wang-Landau sampling is stochastic. An independent run with a different
`random_seed` yields an *equivalent* density of states, not a bit-identical one —
the equal-weight coexistence temperature agrees to within the seed-to-seed
scatter (about 0.2 K). The five deposited seeds per size are independent samples;
their spread is that stochastic bound.
