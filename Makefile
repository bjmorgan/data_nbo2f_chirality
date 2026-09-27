# Regenerate all downstream products from the deposited data files.
#
#   data/     the dataset (committed); the DFT and REWL outputs are read-only,
#             the cluster expansion and its training set are rebuilt from them
#   workflow/ analysis and plotting code
#   build/    intermediate products (git-ignored)
#   figures/  rendered manuscript figures (committed)
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

# matplotlib stamps a creation date into every PDF, which would make otherwise
# identical figures differ byte-for-byte between runs; fixing the epoch removes it
export SOURCE_DATE_EPOCH := 0

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
DFT_CLASS := $(DFT_BUILD)/class_summary.csv
ORBIT_REPS := $(wildcard data/structures/orbit_representatives/orbit_*.vasp)

# ---- r2SCAN relaxations of the twelve orbits ----
R2SCAN_OUTCARS := $(wildcard $(DFT_DATA)/3x3x3_r2scan/config_*/OUTCAR)
R2SCAN_LADDER  := $(DFT_BUILD)/r2scan_ladder.csv

# ---- TaO2F against NbO2F on paired configurations ----
TAO2F_OUTCARS := $(wildcard $(DFT_DATA)/TaO2F_3x3x3/config_*/OUTCAR)
TAO2F         := $(DFT_BUILD)/tao2f_comparison.csv

# ---- 3x3x6: stackings of the P3_121 helix and every achiral arrangement ----
STACK_OUTCARS  := $(wildcard $(DFT_DATA)/3x3x6/stack_*/OUTCAR)
TWINS          := $(DFT_BUILD)/twin_stackings.csv
ACHIRAL_INPUTS := $(wildcard $(DFT_DATA)/3x3x6/*/OUTCAR) $(wildcard $(DFT_DATA)/3x3x6/*/POSCAR.init)
ACHIRAL        := $(DFT_BUILD)/achiral_3x3x6.csv

# ---- structural order-parameter reference values (SI table) ----
OP_REFS := build/structural_ops/op_reference_values.csv

# ---- beta-Fe2SeO against the twelve orbits ----
FE2SEO_CIFS  := $(wildcard data/structures/beta_fe2seo/*.cif)
FE2SEO_MATCH := build/structures/fe2seo_orbit_match.csv

# ---- cluster expansion ----
# The training set and the fitted model are committed, so that the deposit holds
# the model the simulations actually sampled; they are also rebuilt from the DFT
# calculations, so the chain from those to the model is unbroken.
CE_TRAIN := data/ce/training_set_refitB_n96.json
CE_MODEL := data/ce/paircut9_5_5_ardr_n96.ce
CE_LOOCV := build/ce/loocv.csv
CE_VS_DFT := build/ce/ce_vs_dft_low_energy.csv

.PHONY: all rewl dft r2scan tao2f twins achiral refs fe2seo ce cedft figures clean
all: figures refs r2scan tao2f twins achiral fe2seo ce cedft

# ===== REWL =====
## canonical: reweight microcanonical moments -> canonical <O>(T) on the 0.5 K grid
$(RWL_BUILD)/%/canonical_observables/ChainOrder.csv: $(RWL_DATA)/%/observables/ChainOrder.csv $(RWL_DATA)/%/stitched_dos.csv
	@mkdir -p $(dir $@)
	mchammer-pt-reweight-observables $(RWL_DATA)/$*/observables/ $(RWL_DATA)/$*/stitched_dos.csv \
	    --T-min 100 --T-max 800 --T-step 0.5 -o $(RWL_BUILD)/$*/canonical_observables/

## rewl: canonical observables -> per-run T_c estimators -> extrapolation to L -> infinity
rewl: $(RWL_BUILD)/tc_estimators.csv $(RWL_BUILD)/fss_fit.txt

$(RWL_BUILD)/tc_estimators.csv: $(CANON) workflow/rewl/tc_estimators.py
	cd workflow/rewl && $(PY) tc_estimators.py

# the paper's quoted T_c(infinity); the fit prints its summary, which is kept as
# the record of the extrapolation
$(RWL_BUILD)/fss_fit.txt: $(RWL_BUILD)/tc_estimators.csv workflow/rewl/fss_fit.py
	cd workflow/rewl && $(PY) fss_fit.py

# ===== DFT =====
## dft: per-config DFT order params (3x3x3) then per-orbit chiral invariants
dft: $(DFT_OPS) $(DFT_CLASS) $(ORBIT_CO)

$(DFT_OPS): $(wildcard $(DFT_DATA)/3x3x3/config_*/OUTCAR) workflow/dft/dft_ops.py
	cd workflow/dft && $(PY) dft_ops.py

# dft_ops.py writes the class summary immediately after dft_ops.csv, so building
# the rule above satisfies this one; the recipe fires only if the summary alone
# is removed (grouped targets would say this directly, but need make 4.3)
$(DFT_CLASS): $(DFT_OPS) workflow/dft/dft_ops.py
	cd workflow/dft && $(PY) dft_ops.py

$(ORBIT_CO): $(DFT_OPS) $(ORBIT_REPS) workflow/dft/orbit_chain_order.py
	cd workflow/dft && $(PY) orbit_chain_order.py

## r2scan: the ordering of the twelve orbits in r2SCAN against PBEsol, from
## full relaxations in each functional
r2scan: $(R2SCAN_LADDER)

$(R2SCAN_LADDER): $(R2SCAN_OUTCARS) $(DFT_OPS) $(ORBIT_REPS) workflow/dft/r2scan_ladder.py
	cd workflow/dft && $(PY) r2scan_ladder.py

## tao2f: TaO2F against NbO2F on paired 3x3x3 configurations: the energies of
## the tiers and the ordering of the twelve orbits
tao2f: $(TAO2F)

$(TAO2F): $(TAO2F_OUTCARS) $(DFT_OPS) workflow/dft/tao2f_comparison.py
	cd workflow/dft && $(PY) tao2f_comparison.py

## twins: the 3x3x6 stackings of the P3_121 helix, relative to the homochiral one,
## and the energy per twin boundary
twins: $(TWINS)

$(TWINS): $(STACK_OUTCARS) workflow/dft/twin_stackings.py
	cd workflow/dft && $(PY) twin_stackings.py

## achiral: every achiral arrangement on the 3x3x6 cell that obeys both rules,
## relative to the homochiral helix, in DFT, in the cluster expansion, and as
## the mean of its 3x3x3 blocks
achiral: $(ACHIRAL)

$(ACHIRAL): $(ACHIRAL_INPUTS) $(DFT_OPS) $(CE_MODEL) methods/dft/enumerate_3x3x6.py methods/dft/build_stackings.py workflow/dft/achiral_3x3x6.py
	cd workflow/dft && $(PY) achiral_3x3x6.py

# ===== structural order-parameter references =====
## refs: ground-state and random-limit reference values for the SI table.
## The slowest target -- around five minutes, most of it Monte-Carlo sampling of
## random configurations at the largest size.
refs: $(OP_REFS)

$(OP_REFS): workflow/structural_ops/generate_op_references.py
	cd workflow/structural_ops && $(PY) generate_op_references.py

# ===== beta-Fe2SeO =====
## fe2seo: map the iron/vacancy order of beta-Fe2SeO onto the anion labelling and
## identify which of the twelve orbits it is
fe2seo: $(FE2SEO_MATCH)

$(FE2SEO_MATCH): $(FE2SEO_CIFS) $(ORBIT_REPS) workflow/structures/fe2seo_orbit_match.py
	cd workflow/structures && $(PY) fe2seo_orbit_match.py

# ===== cluster expansion =====
## ce: DFT energies -> training set -> fitted model -> cross-validated RMSE,
## the fit quality quoted in the Methods
ce: $(CE_TRAIN) $(CE_MODEL) $(CE_LOOCV)

$(CE_TRAIN): $(wildcard $(DFT_DATA)/6x6x6/*/OUTCAR) methods/cluster_expansion/build_training_set.py
	cd methods/cluster_expansion && $(PY) build_training_set.py

$(CE_MODEL): $(CE_TRAIN) methods/cluster_expansion/fit_cluster_expansion.py methods/cluster_expansion/ce_fit.py
	cd methods/cluster_expansion && $(PY) fit_cluster_expansion.py

$(CE_LOOCV): $(CE_TRAIN) methods/cluster_expansion/loocv.py methods/cluster_expansion/ce_fit.py
	@mkdir -p $(dir $@)
	cd methods/cluster_expansion && $(PY) loocv.py

## cedft: the cluster expansion against DFT on the low-energy manifold -- the
## orbits, the low-energy training structures (with held-out predictions) and
## the 3x3x6 classes -- with posterior uncertainties (SI figure)
cedft: $(CE_VS_DFT)

$(CE_VS_DFT): $(CE_TRAIN) $(CE_MODEL) $(DFT_OPS) $(ORBIT_REPS) $(wildcard $(DFT_DATA)/3x3x3/config_*/OUTCAR) $(ACHIRAL_INPUTS) methods/cluster_expansion/ce_fit.py workflow/ce/ce_vs_dft.py
	@mkdir -p $(dir $@)
	cd workflow/ce && $(PY) ce_vs_dft.py

# ===== figures =====
figures: figures/fig_finite_temperature.pdf figures/fig_si_pe_size.pdf \
         figures/fig_dft_coherence.pdf figures/fig_orbit_gallery.pdf \
         figures/fig_p3121_structure.pdf figures/fig_p3121_legend.pdf \
         figures/fig_reo3_parent.pdf figures/fig_si_ce_vs_dft.pdf

figures/fig_finite_temperature.pdf: $(RWL_BUILD)/tc_estimators.csv $(RWL_BUILD)/fss_fit.txt $(CANON) workflow/figures/fig_finite_temperature.py
	cd workflow/figures && $(PY) fig_finite_temperature.py

figures/fig_si_pe_size.pdf: $(CANON) workflow/figures/fig_si_pe_size.py
	cd workflow/figures && $(PY) fig_si_pe_size.py

figures/fig_si_ce_vs_dft.pdf: $(CE_VS_DFT) workflow/figures/fig_si_ce_vs_dft.py
	cd workflow/figures && $(PY) fig_si_ce_vs_dft.py

figures/fig_dft_coherence.pdf: $(DFT_OPS) workflow/figures/fig_dft_coherence.py
	cd workflow/figures && $(PY) fig_dft_coherence.py

figures/fig_orbit_gallery.pdf: $(DFT_OPS) $(ORBIT_CO) $(ORBIT_REPS) workflow/figures/fig_orbit_gallery.py
	cd workflow/figures && $(PY) fig_orbit_gallery.py

figures/fig_p3121_structure.pdf: $(ORBIT_REPS) workflow/figures/fig_p3121_structure.py
	cd workflow/figures && $(PY) fig_p3121_structure.py

figures/fig_p3121_legend.pdf: workflow/figures/fig_p3121_legend.py
	cd workflow/figures && $(PY) fig_p3121_legend.py

figures/fig_reo3_parent.pdf: workflow/figures/fig_reo3_parent.py
	cd workflow/figures && $(PY) fig_reo3_parent.py

## clean: remove everything the pipeline regenerates, including the committed
## figures and cluster expansion; the DFT and REWL data are never touched
clean:
	rm -rf build figures/fig_*.pdf figures/fig_*.png $(CE_TRAIN) $(CE_MODEL)
