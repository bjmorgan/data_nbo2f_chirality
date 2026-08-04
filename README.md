# Configurational chirality in NbO2F — supporting data and code

Data and analysis code for the NbO2F configurational-chirality study. It has two
parts:

- **REWL finite-size scaling** — replica-exchange Wang-Landau runs across
  supercell sizes that locate the chirality-emergence transition and extrapolate
  its temperature to the bulk limit.
- **DFT demonstration and cluster expansion** — the density-functional
  calculations that motivate the model, the order-parameter analysis built from
  them, and the cluster expansion (with its training set) that the REWL runs
  sample.

Given the deposited data files and the pinned environment, `make` regenerates
every downstream quantity — canonical observables, coexistence-temperature
estimators, the extrapolated `T_c(inf)`, the DFT order parameters, and all
figures. The regeneration is deterministic: in the pinned environment `make`
produces the same bytes on every run, and these are the figures the paper uses.

## Layout

```
data/
  rewl/        stitched density of states + observable moments (REWL runs)
  dft/         VASP relaxations: 3x3x3 demonstration set + 6x6x6 CE training set
  structures/  orbit representatives + reference structures
  ce/          cluster-expansion training set + fitted model
workflow/      run by `make` to reproduce the results:
  rewl/        REWL post-processing (coexistence T_c, finite-size fit)
  dft/         DFT order-parameter analysis
figures/       figure scripts (PDFs render here)
methods/       cluster-expansion + active-learning workflow (documentation; not run by make)
build/         everything `make` produces (regenerated; not committed)
docs/          how-to guides
```

## Quickstart

```sh
conda env create -f environment.yml
conda activate nbo2f-chirality
make                 # regenerates every CSV and figure
```

`make clean` removes everything under `build/` and the rendered figures; the
deposited data files in `data/` are never touched.

## What is deposited vs regenerated

Two upstream steps are too large or too compute-heavy to ship, so their outputs
are the deposited starting point: the 6 GB REWL checkpoints (deposited as the
few-MB stitched density of states and observable moments) and the VASP
relaxations (deposited as their `OUTCAR`s). Everything downstream regenerates
from these with `make`.

| Path | Status | Produced by |
| --- | --- | --- |
| `data/rewl/**/stitched_dos.csv` | deposited | `mchammer-pt-stitch` on the WL checkpoint |
| `data/rewl/**/observables/ChainOrder.csv` | deposited | `mchammer-pt-stitch-observables` |
| `data/rewl/**/config.yaml` | deposited | REWL run configuration |
| `data/dft/{3x3x3,6x6x6}/*/` | deposited | VASP relaxations (INCAR/KPOINTS/POSCAR/OUTCAR) |
| `data/structures/` | deposited | orbit representatives + reference structures |
| `data/ce/training_set_refitB_n96.json` | deposited | CE training set (96 structures) |
| `data/ce/paircut9_5_5_ardr_n96.ce` | deposited | fitted cluster expansion |
| `build/rewl/**/canonical_observables/ChainOrder.csv` | regenerated | `mchammer-pt-reweight-observables` |
| `build/rewl/tc_estimators.csv` | regenerated | `workflow/rewl/tc_estimators.py` |
| `build/dft/dft_ops.csv` | regenerated | `workflow/dft/dft_ops.py` |
| `build/dft/orbit_chain_order.csv` | regenerated | `workflow/dft/orbit_chain_order.py` |
| `build/structural_ops/op_reference_values.csv` | regenerated | `workflow/structural_ops/generate_op_references.py` |
| `figures/fig_*.pdf` | regenerated | `figures/fig_*.py` |

The `Makefile` is the authoritative record of how each regenerated file is made:
every product lists its inputs and the command that builds it.

## Figures

`figures/` builds every plotted figure in the paper. Two of the paper's
structure figures are laid out by hand from these outputs rather than emitted
whole: the parent-structure figure uses `fig_reo3_parent`, and the
$P3_121$ figure combines `fig_p3121_structure`, `fig_p3121_legend` and
`fig_orbit_p3121`. The panels themselves are reproduced here; only their
arrangement and cropping was done in a drawing program. The paper also contains
one hand-drawn schematic, which is not an analysis output and has no source
here.

## Reproducibility boundary

- **REWL sampling** (configuration to WL checkpoint) is stochastic. An
  independent run with a different random seed yields an *equivalent* density of
  states, not a bit-identical one: the equal-weight `T_c` agrees to within the
  seed-to-seed scatter (about 0.2 K). The five seeds per size are independent
  samples, and their spread is that stochastic bound.
- **VASP relaxations** are the DFT upstream. The deposited `OUTCAR`s are the
  relaxation results; re-running them needs VASP and the pseudopotentials (the
  copyright `POTCAR`s are not redistributed — `POTCAR.spec` records which were
  used).
- **Everything `make` builds** — canonical observables, `T_c` estimators, the
  DFT order parameters, and the figures — is deterministic: in the pinned
  environment it produces the same bytes on every run.

## Sizes

Supercells L = 6, 9, 12, 15, 18 (five sizes, five seeds each) feed the
finite-size extrapolation.

## Guides

- `docs/running-a-simulation.md` — how the REWL data was produced, and how to
  run or extend a simulation from a `config.yaml` and the cluster expansion.
- `docs/cluster-expansion-workflow.md` — the active-learning protocol that built
  the cluster expansion and its DFT training set.
