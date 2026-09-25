# Tier 1 run plan: the fits that draft every figure in the paper

Current as of 2026-09-24. This is the companion to `Notes/paper_outline.md`. The outline says
what each figure argues; this document says which synthetic-data (Tier 1) fits produce each
figure, what they cost, and in what order they run.

**The rule this plan follows:** every run produces a panel of a named figure or a named SI
item. Every results figure is drafted at Tier 1, on truncated versions of the FAS network,
before anything is run on the full model. Section 5 covers how, once the drafts exist, we
choose which figures to recreate on the full model with real data (Tier 2).

---

## 0. Setup in brief

**Systems.** The truncated systems cap the maximum fatty-acid chain length and keep all nine
enzymes (outline 2.4). Two are used:
- **Main system: C14+unsat** (352 reactions). It is the smallest truncation that meets all
  three requirements:
  - it engages the unsaturated branch
  - its chain-length profile is spread across six species
  - it is where `d1` and `d2` begin to separate

  All figures built on the main three-parameter fit come from here.
- **Replicate system: C8** (160 reactions), for the figures that need many repeated fits
  (calibration, robustness, SI checks). If the first C8 fit (R0) does not recover both
  parameters with shrinkage > 0.5, C12 takes its place, at ~1.6x the cost.

**Data (identical design on every system; outline 2.4).** All data are generated at the
no-op scaling values, which is the ME1 solution, with per-point noise of sd = 10% of the
clean value + a floor.

| Dataset | Condition(s) | Points |
|---|---|---|
| Time series | baseline | C16 equivalents at 72, 144, …, 720 s (10 points) |
| Profile | baseline | every fatty-acid species at 720 s (C8: 3 species, C4-C8; C14+unsat: 8, C4-C14 plus unsaturated C12 and C14) |
| Initial rates | baseline; FabH 0.1 µM; FabB 0; TesA 0.5 µM; FabZ 0 | C16 equivalents at 150 s ÷ 150 s, µM C16/min (5 points) |

That is five unique initial conditions, so five ODE solves per gradient evaluation.

**Sampler settings (outline 2.6).**
- 4 chains, 300 warmup steps.
- Stop at r-hat ≤ 1.01 and bulk ESS ≥ 400, holding on two consecutive checks.
- At least 3 chains must remain after stranded-chain exclusion.
- `target_accept` 0.8.
- LogNormal [0.1, 10] priors with the median at 1; matched Normal priors for `d`-type
  parameters.
- No concentration clamp; `EQX_ON_ERROR=nan`.

---

## 1. Figure by figure

"Main fit" means R2 below. Run IDs refer to section 2.

| Outline item | What the panel shows | Source | New fits |
|---|---|---|---|
| 3.1 ladder table, observable-count result | `a1` recovery on 14 systems | **completed** single-parameter ladder | 0 |
| 3.1 two-parameter pilots | `a1`+`c3` converges; `a1`+`c2` does not | **completed** pilots | 0 |
| **Fig 2** | posteriors vs. truth; z, shrinkage, 50/90/95% coverage; r-hat/ESS inset | R2 (R1 as fallback panel) | R1, R2 |
| **Fig 3** | SBC rank histogram + nominal-vs-observed coverage | R4 | R4 |
| **Fig 4** | shrinkage / z / coverage vs. noise and vs. prior offset | R5 | R5 |
| 3.1 multimodality | symmetric toy (known answer) + ladder's stranded-chain cases | toy model, CPU only | 0 A100 |
| **Fig 5** | grouped vs. split `c3`; Δelpd ± SE; negative control | R1 + R6 | R6 |
| **Fig 6 / 6b** | shrinkage per parameter; correlation matrix + covariance eigenvalues | R2, with R3 as the known-answer case | 0 beyond R2/R3 |
| 3.3 `d1`/`d2` case | only `12*d1+d2` identifiable at ≤ C12 | R3 | R3 |
| **Fig 7** | posterior-integrated enzyme sensitivity vs. Morris at the point estimate | forward solves over R2 draws | 0 |
| **Fig 8** | chain-length response to FabF/FabB:TesA ratio across draws | forward solves over R2 draws | 0 |
| **Fig 9** | shrinkage per data point across the data-type grid | expected information for every cell; R7 samples 3 cells | R7 |
| SI convergence / divergences | per-run tables | every run | 0 |
| SI stranded-chain detection | lp-gap mechanism | **completed** (ladder) | 0 |
| SI chain-count diagnostic power | diagnostic spread vs. chain count | **completed** chain-count test | 0 |
| SI mass matrix | diagonal vs. dense on a known-answer pair | R3 | inside R3 |
| SI `target_accept`, ODE tolerance | posterior unchanged at 0.95 / tighter `rtol` | R8 | R8 |
| SI prior width | shrinkage + misspecification curve | ladder + R5 | 0 |

Tier 1 needs no held-out-data refits. Where the outline uses a held-out check to decide
between two answers (3.4, 3.5), the known truth decides at Tier 1. Held-out refits belong to
Tier 2.

---

## 2. The runs

Costs are in A100-hours, from the cost model in appendix B. The anchor measurement used seven
conditions solved to 720 s, and these runs use five, with four of them solved only to 150 s.
So the costs are **upper bounds**, probably by ~30%. Re-anchor them on R0 and R1.

Run folders are `Results/Tier1/<run name>/`, written by `build_tier1_configs.py --plan`.

| ID | Run | System | Free params | Fits | A100-h | Feeds |
|---|---|---|---|---|---|---|
| R0 | smoke test, production stopping rule | C8 | `a1`+`c3` | 1 | 2.5 | plumbing; chooses the replicate system; centre point of Fig 4 |
| R1 | main-system duo | C14+unsat | `a1`+`c3` | 1 | 6.4 | Fig 2 fallback; Fig 5 grouped model; Fig 9 full-data cell |
| R2 | **main fit** | C14+unsat | `a1`+`c3`+`a2` | 1 | 34 | Figs 2, 6, 6b, 7, 8 |
| R3 | `d1`+`d2` known-answer control | C8 diagonal, C8 dense, C14+unsat diagonal | `d1`+`d2` | 3 | ~11 (cap 23) | 3.3 case; Fig 6/6b; SI mass matrix |
| R4 | SBC replicates, truth drawn from the prior | C8 | `a1`+`c3` | 10 pilot, then 30 more | 25 + 75 | Fig 3 |
| R5 | robustness: noise 5/20/40% (10% = R0), prior median shifted +1/+2/+3/+4 prior sd | C8 | `a1`+`c3` | 7 | 17.5 | Fig 4 |
| R6 | `c3` split test | C14+unsat | see below | 3 | ~74 | Fig 5 |
| R7 | data-type check, 3 cells | C14+unsat | `a1`+`c3` | 3 | 19 | Fig 9 |
| R8 | `target_accept` 0.95; `rtol` 1e-5 | C8 | `a1`+`c3` | 2 | ~7.5 | SI |
| | **Total** | | | **~71** | **~272** | |

### Why these parameters

- **`a1`+`c3`+`a2` for the main fit.** `a1` leads total production, `a2` leads chain length
  and unsaturated fraction, and `c3` is the TesA handle. All three are well separated in the
  data, and this trio has already converged in pilots on C6 and C10. Of all the trios, it
  gives the broadest coverage of the three sensitivity objectives in Figs 7 and 8.
- **`a1`+`c3` for everything replicated.** It is the one pair that has converged on every
  system tried, and it's cheap.
- **No four-parameter fit at Tier 1.** No four-parameter cost has been measured, and R3
  already supplies the unidentified-parameter example that a weakly identified fourth
  parameter would otherwise provide.

### Notes on the less obvious runs

**R3 (`d1`+`d2`).** On C8 every TesA binding step shares coefficient 12, so only
`12*d1 + d2` is identifiable and the posterior is a ridge bounded by the prior. That gives
known answers for three things at once:
- Fig 6 must flag both parameters individually as weakly identified.
- Fig 6b's eigen-analysis must find one tight combination and one flat direction.
- Fitting the same data with a diagonal and a dense mass matrix answers the SI's mass-matrix
  question.

The pre-flight already shows the degeneracy. At the sampler's start on C8, the gradients
with respect to the two sampled coordinates (`12*d1` and `d2`) are identical (11.429 each).
On C14+unsat they differ slightly (26.2 vs. 27.3). C14+unsat adds the point where separation
begins. Ridge posteriors can hit the tree-depth ceiling, so R3 has a hard 2x compute cap;
hitting it under diagonal but not dense is itself the SI result. *Optional:* C20+unsat (~12
A100-h) shows clear separation. Add it only if the figure needs a third point.

**R6 (grouping, Fig 5).** `c3` ties six TesA hydrolysis constants with six distinct nominal
values. It is split into short-chain (`c3s`: C4-C8) and long-chain (`c3l`: C10-C14 plus the
unsaturated species) halves in a script-generated variant of the C14+unsat reaction set
(`Reactions/EC_FAS_ME1/C14+unsat+c3split`); the reaction files are never hand-edited. The
runs:
- (a) grouped `a1`+`c3` on the standard data. This is R1, reused.
- (b) split `a1`+`c3s`+`c3l` on the same data. New; three-parameter cost ≈ 34.
- (c) data generated at `c3s` = 1, `c3l` = 3, fit with both models. New; 6.4 + 34.

At `c3l` = 3 the C10-C14 species move by −15% to +144% while C4-C8 move about 6%, a pattern
a single `c3` cannot produce. The three-parameter cost is borrowed from the main fit, but
`c3s`/`c3l` may be more correlated than that. Measure (b) before running (c). Gate every
Δelpd on Pareto-k ≤ 0.7. `a1` is included so the posterior has enough spread for LOO to
work.

**R7 (Fig 9).** The expected information is computed for every grid cell on the main-fit
model, at zero sampling cost, and that draws the figure. Three cells are then sampled with
`a1`+`c3`:
- profile only (the cheapest data type)
- the best cell per data point
- the full dataset (R1, reused)

This checks that the information ranking matches real posterior shrinkage.

**R4/R5 on C8.** Calibration and robustness test whether stated uncertainty is honest, which
doesn't depend on network size. R4 starts with a 10-replicate pilot and continues to 40 only
if the pilot's coverage and rank histogram look sane. SBC ranks and coverage come from the
same replicates. The R5 noise-level datasets reuse the 10% dataset's random draws, rescaled,
so the noise axis carries no draw-to-draw scatter. Compare the prior-offset runs on log-scale
shrinkage (`shrinkage_log` in `recovery_report.py`), which doesn't change when the prior
median moves.

---

## 3. Order of execution, with gates

**Stage 0: build and data (no A100 time).**

Built 2026-09-24:
1. Tier-1 data for C8, C12 and C14+unsat in `Data/Tier1_rates/Chain_<system>/`
   (`make_tier1_rate_data.py`). Each folder holds the noisy files, a noise-free `clean/`
   copy, and a `ground_truth.json` recording every scaling value, the seed (0), noise
   settings, conditions and solver settings. The data are solved at `rtol` 1e-8 / `atol`
   1e-10 with the production PID coefficients. There are also C8 noise-level variants at
   5/20/40% (`Chain_C8_noise5` etc.) for R5, and the off-grouping dataset
   `Chain_C14+unsat+c3split_c3l3` for R6.
2. The model's initial-rate output (`FA_conc.py`) reports µM C16/min, matching the data and
   the ME1 kinetics measurements.
3. Configs for every run except R4 and R7 (`build_tier1_configs.py --plan`). Each config
   records its data's truth (`tier1_truth`).
4. `C14+unsat+c3split` reaction set (`make_c3_split_variant.py --check`). At `c3s` = `c3l`
   = 1 it reproduces the original model exactly (max difference 0).
Checked 2026-09-25, after regenerating everything above (all initial rates match the values
in appendix D):

5. Pre-flight (`check_model_vs_data.py --all --grad`) passed on all 18 configs. Each model
   reproduces its noise-free data to 10⁻⁶-10⁻⁵ at the working tolerance (10⁻⁷ for R8's
   `rtol` 1e-5 run), and the log posterior and gradient are finite at the sampler's start.
   The R6 configs behave as designed:
   - The split model reproduces both the standard and the off-grouping data. At the start,
     its `c3s` + `c3l` gradients sum to the grouped model's `c3` gradient, less the one extra
     prior term.
   - The grouped model on the off-grouping data skips the clean comparison. Its profile
     noise z has sd 2.3, the misfit a single `c3` cannot absorb.
6. Multimodality toy (`multimodality_toy.py`, 10 replicates, saved to
   `multimodality_toy.json`). Every replicate split its chains across the two equal-mass
   mirror modes, and r-hat flagged all 10 (1.30-1.74). The stranded-chain rule excluded
   neither mode in any replicate (largest gap 0.18 nats, threshold 20). No divergences; in 3
   of the 10 a chain crossed between modes.

   The replicates were run one per process. On the laptop, jaxlib 0.7.0 aborts
   intermittently (`recursive_mutex lock failed`, inside XLA:CPU's JIT) when a second model
   is compiled in the same process. Each replicate is deterministic in its seed and
   reproduced exactly across runs. The pre-flight also aborted at interpreter exit, after
   printing "18/18 passed".
7. Shifted priors fixed in `inference_runner.py` (`_fit_prior`). preliz 0.23's `maxent`
   mis-fit the LogNormal at +2, +3 and +4 prior sd (σ 0.16, 0.52 and 1.11 instead of 1.17;
   mass 1.00, 1.00 and 0.96 instead of 0.95), with only a UserWarning. With the median fixed,
   σ is set by the mass constraint alone, so a LogNormal with a fixed median is now solved
   exactly. It matches `recovery_report.py`'s prior to 10⁻¹⁵ on every config, and the
   unshifted prior's σ moves by 1.7×10⁻⁶. Any prior whose mass misses its target by more than
   10⁻³ now raises instead of warning. The configs themselves were already right and are
   unchanged. The +2/+3/+4 sd pre-flights were re-run with finite logp and gradient. At +4 sd
   the sampler starts where the data barely respond: `c3`'s gradient there is the prior's
   alone (−0.5).

Still to do before R0:
- Sync the new files to the cluster, including `Utilities/inference_runner.py`.

**Stage 1: R0 (~2.5 A100-h).** Checks plumbing: resume, stopping on r-hat + ESS, finalize,
figures, `recovery_report.py`. It also decides the replicate system. Nothing else is queued
until R0 has been inspected.

**Stage 2: everything cheap and independent, one batch (~70 A100-h).** R1, R3, R5, R8 and the
R4 pilot.
- Gate to R2: R1 converges and recovers both parameters.
- Gate to the full R4: the pilot looks sane.

**Stage 3: the main fit and its dependents (~170 A100-h).**
- R2: inspect warmup with `convergence_and_warmup.py` at the first segment boundary. If
  acceptance is well below 0.8, or the first block still drifts, restart with 600 warmup
  steps and record it.
- R6 (b), then (c) once (b)'s cost is known.
- R7, and the remaining R4 replicates.

**Stage 4: post-processing (forward solves).** Figs 6/6b from R2 and R3; Figs 7 and 8 once
their modules exist; Fig 9's information grid.

**Stage 5: draft every figure, then choose what goes to Tier 2** (section 5).

---

## 4. Code status

**Exists** (all in `job_files/tier1/` unless noted):
- `build_tier1_configs.py`: `--plan` writes every config in section 2 except R4 and R7
- `make_tier1_rate_data.py`: the data above, including off-grouping and noise-level variants
- `make_c3_split_variant.py`: R6's reaction variant, with an exact-equivalence check
- `check_model_vs_data.py`: pre-flight (model vs. clean data, noise z, logp/gradient)
- `multimodality_toy.py`: 3.1's known-answer toy
- `recovery_report.py`: per run and parameter, z, shrinkage (natural and log scale) and
  50/90/95% coverage, scored against each run's recorded truth
- `tier1.sbatch`, `convergence_and_warmup.py`
- initial-rate, C16-equivalent and mole-fraction observables in
  `Calculation Files/Full_FAS/FA_conc.py`

**To build, in the order they block figures:**
1. Posterior-integrated sensitivity: Morris over the nine enzyme concentrations, repeated
   across posterior draws (Fig 7). `morris_screen.py` does Morris at a single point.
2. Optimisation across posterior draws (Fig 8).
3. Expected-information grid over data types (Fig 9).
4. Acyl-ACP intermediate observables (Fig 9's intermediates cells): a copy of `FA_conc.py`
   with a different species list.
5. SBC harness: draw a truth from the prior → generate data → fit → record the rank (Fig 3).

---

## 5. After the drafts: which figures get recreated on the full model

This is decided with every Tier-1 draft in hand. The default:

| Figure | Tier 2? | Why |
|---|---|---|
| Fig 2 | **analog, yes**: is the published estimate inside the real-data posterior? | no truth to recover on real data |
| Figs 3, 4 | no | they need a known truth |
| `d1`/`d2` case, toy, SI mass matrix | no | known-answer controls |
| Fig 5 | only if the Tier-1 test discriminates (off-grouping data correctly flagged, Pareto-k passes) | otherwise it stays a Tier-1 methods result |
| Figs 6/6b | yes, free from the Tier-2 fit | real-data identifiability |
| Fig 7 | yes (primary evidence), forward solves only | |
| Fig 8 | yes (primary evidence) | Tier 1 gives an attenuated preview only |
| Fig 9 | only the cells the real data allow; if R7 shows the information ranking predicts sampled shrinkage, Tier 2 uses the information analysis only | |
| SI tolerance / `target_accept` | one check on the Tier-2 fit, if budget allows | |

The Tier-2 core is therefore **one three-parameter fit** of the full model to the real data,
plus grouping fits only if Fig 5 earns them. Its parameters come from the Tier-1 Fig 6
results and the sensitivity screen. Tier 2 is blocked until the FabH-knockout conditions
(outline 2.8) are resolved.

---

## Appendix A: evidence behind the sampler settings

- **4 chains.** In the chain-count test (C12, `a1`+`c3`, 4/8/16/32/64 chains, equal
  sampling budget), 4 chains converged in 4.00 A100-h against 7.10 at 8. Splitting the
  64-chain run into disjoint groups gives the SI's diagnostic-power result.
- **300 warmup steps at two parameters.** Acceptance over the last 50 warmup draws is
  0.777-0.780 against the 0.8 target. The first sampling block's mean is within 0.12 sd of
  the rest, r-hat moves ≤ 0.004 when that block is dropped, and there are zero divergences.
  Step sizes scatter 8-10x across chains; with per-chain adaptation that means each chain
  reaches the same acceptance by a different route. Not yet validated at three parameters,
  hence the R2 warmup check.
- **A minimum of 3 chains.** One stranded chain out of four must not block a run. Two or more
  triggers a rerun.

## Appendix B: cost model

Anchor: C12, `a1`+`c3`, 4 chains, 300 warmup steps, seven 720 s conditions = **4.00 A100-h**
(warmup 2.05, sampling 1.95, converged at draw 400).

- **System size:** cost per draw ∝ reactions^1.21 (fitted on C6/C10/C14). Relative to C12:
  C8 0.62x, C14 1.20x, C14+unsat 1.61x, C20+unsat ~2.98x.
- **Parameter count:** `a1`+`c3`+`a2` costs ~3.5x per draw and ~3.0x per warmup step
  relative to `a1`+`c3`, and needs 2.24x the draws (pilot medians).

| | C8 | C12 | C14 | C14+unsat |
|---|---|---|---|---|
| two parameters | 2.5 | 4.0 | 4.8 | 6.4 |
| three parameters | 13 | 21 | 26 | 34 |
| four parameters (extrapolated; nothing measured) | 62 | 100 | 121 | 161 |

All runs are resumable across the 12 h cluster limit; a longer run just means more segments.

## Appendix C: parameter-set constraints

Collinearity index gamma is measured on the Tier-1 design; 1 means independent directions in
the data, and > 10-15 means jointly unidentifiable (Brun et al.).

- `d1` with `d2`: identical direction in the data up to C12 (gamma infinite on C8/C12,
  167-174 on C14/C14+unsat). They are never paired in a recovery fit. R3 pairs them on
  purpose, as the non-identifiable control.
- `c3` with `d1`: gamma 4.5-7.2. The pilot hit the tree-depth ceiling on all three systems.
- `a1` with `c2`: gamma 1.21-1.58 (well separated), yet the pilot never converged, and a
  dense mass matrix didn't rescue it in 12 h. It stays out of the main fits and is SI
  evidence only.
- Best-separated pairs: `a1`+`d1` 1.10, `a1`+`c3` 1.12, `b3`+`c1` 1.27. The trios scoring
  ~2 all contain `c2`, which looks "independent" only because the data barely see it. The
  quads score 5-7.5 and the quints 15+.
- Scaling-group sizes on C14+unsat (constants / distinct nominal values): `a1` 4/2, `a2`
  13/1, `c2` 7/6, `c3` 6/6, `c4` 2/2, `e` 14/6, `b1` 10/3. Single-value groups (`a2`, `a3`,
  `b3`) are trivial as grouping tests.

## Appendix D: how the data design was chosen

- **Why these three data types.** They mirror the real ME1 measurements (a reference time
  course, GC/MS chain-length profiles, initial rates), so each Tier-1 figure previews an
  analysis that can be repeated on real data.
- **Condition selection.** `scan_rate_multipliers.py` swept each enzyme over a
  one-significant-figure grid from 0 to 30 µM, and `check_rate_design.py` applied four tests:
  - ≥ 0.2 decades of separation from baseline on C12, C14 and C14+unsat (FabB's knockout
    reaches 0.19 on C14+unsat, the one near miss, and is kept as the only handle on the
    unsaturated branch)
  - output ≥ 5% of baseline
  - ≤ 1.5x baseline solver steps
  - agreement with a near-exact re-solve

  Those four conditions touch 12 of the 14 live scaling parameters. Nothing moves `b2`
  (FabD) or `f` (FabA). FabF adds nothing beyond FabB and FabH, and FabB is the only handle on
  the unsaturated branch.
- **Measured initial rates (µM C16/min, relative to baseline).**
  - C8: 3.83; FabH 0.50, FabB 0.61, TesA 0.10, FabZ 0.49
  - C12: 6.60; FabH 0.44, FabB 0.57, TesA 0.32, FabZ 0.48
  - C14+unsat: 5.30; FabH 0.49, FabB 0.65, TesA 0.63, FabZ 0.20
- **Where the profile carries information.** On C8, 99% of the 720 s total is in one species,
  so the profile adds little. The share in the largest species is 89% at C12, 58% at C14 and
  41% at C14+unsat. That's one reason the main system is C14+unsat.
- **A candidate additional design:** paired ratio conditions (FabF/FabB up while TesA down).
  These are what the ratiometric strategy predicts on, and they would resemble the ME1
  GC/MS experiments. Add them only if 3.5 or 3.6 needs them.
