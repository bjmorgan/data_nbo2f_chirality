# Regenerate all downstream REWL products from the deposited primitives.
#
# Layout:
#   data/     deposited primitives (read-only inputs; committed)
#   workflow/ analysis code
#   build/    everything this Makefile produces (git-ignored)
#   figures/  figure scripts (PDFs land here, git-ignored)
#
# Deposited (NOT regenerable here -- produced from the 6 GB checkpoints,
# which are not part of this bundle):
#   data/rewl/L*/prod/seed*/stitched_dos.csv                    density of states
#   data/rewl/L*/prod/seed*/measure/stitched_dos.csv            frozen-g DOS
#   data/rewl/L*/prod/seed*/measure/observables/ChainOrder.csv  microcanonical moments
#
# Regenerated (deterministic; bit-for-bit reproducible):
#   build/rewl/.../canonical_observables/ChainOrder.csv   canonical <O>(T), Binder(T)
#   build/rewl/tc_estimators.csv                          per-(size,seed) T_c estimators
#   figures/fig_*.pdf, figures/fig_*.png
#
# Requires mchammer-pt (>= 0.27.1) on PATH; see environment.yml.

PY    ?= python
DATA  := data/rewl
BUILD := build/rewl

SIZES := 6 9 12 15
SEEDS := 1 2 3 4 5

MEAS  := $(foreach L,$(SIZES),$(foreach s,$(SEEDS),L$(L)/prod/seed$(s)/measure))
CANON := $(foreach m,$(MEAS),$(BUILD)/$(m)/canonical_observables/ChainOrder.csv)

.PHONY: all canonical tc figures clean
all: figures

## canonical: reweight microcanonical moments -> canonical <O>(T) on the 0.5 K grid
canonical: $(CANON)

$(BUILD)/%/canonical_observables/ChainOrder.csv: $(DATA)/%/observables/ChainOrder.csv $(DATA)/%/stitched_dos.csv
	@mkdir -p $(dir $@)
	mchammer-pt-reweight-observables $(DATA)/$*/observables/ $(DATA)/$*/stitched_dos.csv \
	    --T-min 100 --T-max 800 --T-step 0.5 -o $(BUILD)/$*/canonical_observables/

## tc: equal-weight + Binder-minimum T_c estimators for every run
tc: $(BUILD)/tc_estimators.csv
$(BUILD)/tc_estimators.csv: $(CANON) workflow/rewl/tc_estimators.py
	cd workflow/rewl && $(PY) tc_estimators.py

## figures: render the REWL-data figures (others depend on DFT/structure data)
figures: figures/fig_finite_temperature.pdf figures/fig_si_pe_size.pdf

figures/fig_finite_temperature.pdf: $(BUILD)/tc_estimators.csv $(CANON) figures/fig_finite_temperature.py
	cd figures && $(PY) fig_finite_temperature.py

figures/fig_si_pe_size.pdf: $(CANON) figures/fig_si_pe_size.py
	cd figures && $(PY) fig_si_pe_size.py

## clean: remove everything regenerated (keeps deposited primitives)
clean:
	rm -rf $(BUILD) figures/fig_*.pdf figures/fig_*.png
