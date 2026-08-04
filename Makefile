# Regenerate all downstream products from the deposited data files.
#
#   data/     deposited data files (read-only inputs; committed)
#   workflow/ analysis code
#   build/    everything this Makefile produces (git-ignored)
#   figures/  figure scripts (PDFs/PNGs land here, git-ignored)
#
# Two pipelines feed the figures:
#   REWL  data/rewl/...        -> canonical observables -> T_c estimators
#   DFT   data/dft/, structures -> per-config order params -> per-orbit invariants
#
# Deposited (NOT regenerable here): the stitched DOS / observable moments come
# from the 6 GB REWL checkpoints; the DFT OUTCARs come from VASP. Everything
# below regenerates deterministically from those. Requires the pinned
# environment (environment.yml).

PY ?= python

# ---- REWL finite-size-scaling spine ----
RWL_DATA  := data/rewl
RWL_BUILD := build/rewl
SIZES := 6 9 12 15 18
SEEDS := 1 2 3 4 5
MEAS  := $(foreach L,$(SIZES),$(foreach s,$(SEEDS),L$(L)/prod/seed$(s)/measure))
CANON := $(foreach m,$(MEAS),$(RWL_BUILD)/$(m)/canonical_observables/ChainOrder.csv)

# ---- DFT order parameters ----
DFT_DATA  := data/dft
DFT_BUILD := build/dft
DFT_OPS   := $(DFT_BUILD)/dft_ops.csv
ORBIT_CO  := $(DFT_BUILD)/orbit_chain_order.csv
ORBIT_REPS := $(wildcard data/structures/orbit_representatives/orbit_*.vasp)

# ---- structural order-parameter reference values (SI table) ----
OP_REFS := build/structural_ops/op_reference_values.csv

.PHONY: all rewl dft refs figures clean
all: figures refs

# ===== REWL =====
## canonical: reweight microcanonical moments -> canonical <O>(T) on the 0.5 K grid
$(RWL_BUILD)/%/canonical_observables/ChainOrder.csv: $(RWL_DATA)/%/observables/ChainOrder.csv $(RWL_DATA)/%/stitched_dos.csv
	@mkdir -p $(dir $@)
	mchammer-pt-reweight-observables $(RWL_DATA)/$*/observables/ $(RWL_DATA)/$*/stitched_dos.csv \
	    --T-min 100 --T-max 800 --T-step 0.5 -o $(RWL_BUILD)/$*/canonical_observables/

## rewl: canonical observables -> per-run coexistence T_c estimators
rewl: $(RWL_BUILD)/tc_estimators.csv

$(RWL_BUILD)/tc_estimators.csv: $(CANON) workflow/rewl/tc_estimators.py
	cd workflow/rewl && $(PY) tc_estimators.py

# ===== DFT =====
## dft: per-config DFT order params (3x3x3) then per-orbit chiral invariants
dft: $(DFT_OPS) $(ORBIT_CO)

$(DFT_OPS): $(wildcard $(DFT_DATA)/3x3x3/config_*/OUTCAR) workflow/dft/dft_ops.py
	cd workflow/dft && $(PY) dft_ops.py

$(ORBIT_CO): $(DFT_OPS) $(ORBIT_REPS) workflow/dft/orbit_chain_order.py
	cd workflow/dft && $(PY) orbit_chain_order.py

# ===== structural order-parameter references =====
## refs: ground-state and random-limit reference values for the SI table.
## The slowest target -- around five minutes, most of it Monte-Carlo sampling of
## random configurations at the largest size.
refs: $(OP_REFS)

$(OP_REFS): workflow/structural_ops/generate_op_references.py
	cd workflow/structural_ops && $(PY) generate_op_references.py

# ===== figures =====
figures: figures/fig_finite_temperature.pdf figures/fig_si_pe_size.pdf \
         figures/fig_dft_coherence.pdf figures/fig_orbit_gallery.pdf \
         figures/fig_p3121_structure.pdf figures/fig_p3121_legend.pdf \
         figures/fig_reo3_parent.pdf

figures/fig_finite_temperature.pdf: $(RWL_BUILD)/tc_estimators.csv $(CANON) figures/fig_finite_temperature.py
	cd figures && $(PY) fig_finite_temperature.py

figures/fig_si_pe_size.pdf: $(CANON) figures/fig_si_pe_size.py
	cd figures && $(PY) fig_si_pe_size.py

figures/fig_dft_coherence.pdf: $(DFT_OPS) figures/fig_dft_coherence.py
	cd figures && $(PY) fig_dft_coherence.py

figures/fig_orbit_gallery.pdf: $(DFT_OPS) $(ORBIT_CO) $(ORBIT_REPS) figures/fig_orbit_gallery.py
	cd figures && $(PY) fig_orbit_gallery.py

figures/fig_p3121_structure.pdf: $(ORBIT_REPS) figures/fig_p3121_structure.py
	cd figures && $(PY) fig_p3121_structure.py

figures/fig_p3121_legend.pdf: figures/fig_p3121_legend.py
	cd figures && $(PY) fig_p3121_legend.py

figures/fig_reo3_parent.pdf: figures/fig_reo3_parent.py
	cd figures && $(PY) fig_reo3_parent.py

## clean: remove everything regenerated (keeps deposited data files)
clean:
	rm -rf build figures/fig_*.pdf figures/fig_*.png
