# Configurational chirality in NbO2F — supporting data and code

Data and analysis code for the replica-exchange Wang-Landau (REWL) finite-size
scaling study of the chirality-emergence transition in NbO2F.

Given the deposited density-of-states primitives and the pinned environment,
`make` regenerates every downstream quantity — canonical observables, the
per-run coexistence-temperature estimators, the extrapolated `T_c(inf)`, and the
finite-temperature figures — and it does so bit-for-bit.

## Layout

```
data/       deposited primitives (read-only inputs; committed)
workflow/   analysis code (run by the Makefile)
figures/    figure scripts (PDFs render here)
build/      everything `make` produces (regenerated; not committed)
```

## Quickstart

```sh
conda env create -f environment.yml
conda activate nbo2f-chirality
make                 # canonical observables -> T_c estimators -> figures
```

`make clean` removes everything under `build/` and the rendered figures; the
deposited primitives in `data/` are never touched.

## What is deposited vs regenerated

The 6 GB REWL checkpoints are deliberately **not** deposited. The stitched
density of states and the microcanonical observable moments — a few MB — are the
deposited starting point, and everything downstream regenerates from them.

| Path | Status | Produced by |
| --- | --- | --- |
| `data/rewl/L*/prod/seed*/stitched_dos.csv` | deposited | `mchammer-pt-stitch` on the WL checkpoint |
| `data/rewl/L*/prod/seed*/measure/stitched_dos.csv` | deposited | `mchammer-pt-stitch` (frozen-g checkpoint) |
| `data/rewl/L*/prod/seed*/measure/observables/ChainOrder.csv` | deposited | `mchammer-pt-stitch-observables` (frozen-g checkpoint) |
| `data/rewl/L*/prod/seed*/**/config.yaml` | deposited | run configuration |
| `build/rewl/L*/prod/seed*/measure/canonical_observables/ChainOrder.csv` | regenerated | `mchammer-pt-reweight-observables` |
| `build/rewl/tc_estimators.csv` | regenerated | `workflow/rewl/tc_estimators.py` |
| `figures/fig_*.pdf` | regenerated | `figures/fig_*.py` |

## Reproducibility boundary

- **Configuration to checkpoint** (the Wang-Landau sampling itself) is
  stochastic. An independent run with a different random seed yields an
  *equivalent* density of states, not a bit-identical one: the equal-weight
  `T_c` agrees to within the seed-to-seed scatter (about 0.2 K). The five
  deposited seeds per size are that reproducibility demonstration.
- **Checkpoint to primitives** (`mchammer-pt-stitch`, `-stitch-observables`) is
  deterministic but needs the checkpoints, so it is documented rather than
  executable here; its outputs are the deposited primitives.
- **Primitives to results to figures** (`make`) is deterministic and reproduces
  the deposited-era outputs bit-for-bit in the pinned environment.

## Sizes

Supercells L = 6, 9, 12, 15 (four sizes, five seeds each) feed the finite-size
extrapolation. L = 18 is a separate cross-check whose primitives are added when
its runs complete.
