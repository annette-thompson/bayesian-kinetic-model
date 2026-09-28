# Tier 1 run plan: the fits that draft every figure in the paper

Current as of 2026-09-27 22:00. This is the companion to `Notes/paper_outline.md`. The outline says
what each figure argues; this document says which synthetic-data (Tier 1) fits produce each
figure, what they cost, and in what order they run.

**The rule this plan follows:** every run produces a panel of a named figure or a named SI
item. Every results figure is drafted at Tier 1, on truncated versions of the FAS network,
before anything is run on the full model. Section 6 covers how, once the drafts exist, we
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
  parameters with posterior contraction > 0.5, C12 takes its place, at ~1.6x the cost. R0
  recovered both at contraction 0.997-0.998 (log scale; section 4), so C8 stays.

**Why C8, and not C6, C10 or C12.** The reasons were not written down when C8 was chosen.
They were reconstructed on 2026-09-26 from the reaction files, the Morris screen and the cost
profiles. Below C14, only FabA, FabZ and TesA have chain-length-specific rate constants:

| Enzyme: constant (fitted group) | Value relative to C4 | Differences appear |
|---|---|---|
| FabA: binding of hydroxyacyl- and enoyl-ACP (not fitted) | C6 3.8x, C8 8.5x, C10 11.8x (peak), C12 8.9x, C14 1x, C16+ 0.44x | from C6; large by C8 |
| FabZ: binding of hydroxyacyl- and enoyl-ACP (not fitted) | C6 2.1x (peak), C8 0.63x, C10 0.79x, C12 0.43x, C14 0.12x | from C6; turns over at C8 |
| TesA: acyl-ACP binding (`d1`, `d2`) | C6 1.6x, C8 8.9x, C10 32x, rising further from C12 | small at C6; large from C8 |
| TesA: hydrolysis kcat (`c3`) | C6 0.90x, C8 1.8x, C10 0.28x, C12 4.5x, C14 8.1x | first real difference at C8 |

From C14 on, FabH's acyl-ACP koff (`1/e`), flat up to C12, drops about 5x. FabB's and FabF's
condensation kcat (`c2` for FabF) are flat to C14, then fall to 0.15-0.32x at C16 and 0.02x
at C18. FabD, FabG and FabI have one value at every chain length.
- **C6.** It has only the start of that specificity: FabA 3.8x, FabZ 2.1x, TesA binding 1.6x,
  and TesA kcat within 10%. So on C6, `c3` ties two nearly equal constants (0.033 and 0.030),
  leaving no internal spread for the grouping test (Fig 5). C8 is the smallest system where
  all three enzymes differ substantially. The Morris screen agrees: on C6, `a1` ranks 4th for
  total production (51% of the top parameter), behind `c3`, `d2` and `d1`. On C8 it ranks
  2nd on both objectives (82% and 74% of the top). Nor is C6 cheaper: per gradient, C8 costs
  0.94-0.99x C6 (`multiparam_tests/cost_profile.py`), because the solver's step counts are
  the same.
- **C10 and C12.** Neither adds anything qualitatively new; they extend the FabA, FabZ and
  TesA patterns C8 already has. Both are slower, because the sampler takes the same number of
  leapfrog steps per draw on every system (4.3-5.1), so cost follows ODE size. In the `a1`+`c3`
  pilots, C10 took 1.56x and C14 2.40x C6's time per draw, and draws to converge showed no
  trend with size (500, 600 and 300 on C6, C10 and C14). FabH's specificity first appears at
  C14, one more reason the main system is C14+unsat.
- **Caveat for the paper.** FabB/FabF's slowdown from C16, which is what ends elongation in
  the full system, is in no Tier-1 system. Every truncation stops elongating at its cap
  instead, which is why 99% of C8's product is C8.

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
- Stop at r-hat ≤ 1.01 and bulk ESS ≥ 400, holding on three consecutive checks 100 draws apart:
  two, then a 100-draw confirmation block whose check must pass too. Until 2026-09-27 a failed
  confirmation was logged and ignored; it happened in 3 of 54 finished runs (R8 target_accept
  0.95, c3 r-hat 1.0116; R5 noise 20%, 1.0107; sbc029, 1.0110). Those three were reopened from
  their checkpoints under the new rule (`tier1.sbatch "<run>" 11.5 1000`: the third argument is
  the extra-draw ceiling; an environment variable does not reach the job, which runs with
  --export=NONE, and the first attempt was a no-op for that reason), so every run finishes
  under the same rule. R2 was a fourth case: its last segment started at ~14:30 on the older
  code, and its confirmation check at 700 draws failed (`a1` r-hat 1.0102) and was ignored. It
  was reopened the same way at 22:30 and finalized at 1000 draws on 2026-09-28 (section 3). The rule lives in resumable_sampler.py (a failed confirmation
  revokes the stop); the configs are unchanged, since their stopping keys are in the resume
  signature.
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
| **Fig 2** | posteriors vs. truth; z, posterior contraction, 50/90/95% coverage; r-hat/ESS inset | R2 (R1 as fallback panel) | R1, R2 |
| **Fig 3** | SBC rank histogram + nominal-vs-observed coverage | R4 | R4 |
| **Fig 4** | posterior interval, contraction and z vs. noise and vs. prior offset, from the default start (`robustness.png`; the ME1-start comparison is an SI panel, `robustness_start_comparison.png`) | R5 | R5 |
| 3.1 multimodality | symmetric toy (known answer) + ladder's stranded-chain cases | toy model, CPU only | 0 A100 |
| **Fig 5** | grouped vs. split `c3`; Δelpd ± SE; negative control | R1 + R6 | R6 |
| **Fig 6 / 6b** | contraction per parameter; correlation matrix + covariance eigenvalues | R2, with R3 as the known-answer case | 0 beyond R2/R3 |
| 3.3 `d1`/`d2` case | only `12*d1+d2` identifiable at ≤ C12 | R3 | R3 |
| **Fig 7** | posterior-integrated enzyme sensitivity vs. Morris at the point estimate | forward solves over R2 draws | 0 |
| **Fig 8** | chain-length response to FabF/FabB:TesA ratio across draws | forward solves over R2 draws | 0 |
| **Fig 9** | contraction per data point across the data-type grid | expected information for every cell; R7 samples 3 cells | R7 |
| SI convergence / divergences | per-run tables | every run | 0 |
| SI stranded-chain detection | lp-gap mechanism | **completed** (ladder) | 0 |
| SI chain-count diagnostic power | diagnostic spread vs. chain count | **completed** chain-count test | 0 |
| SI mass matrix | diagonal vs. dense on a known-answer pair | R3 | inside R3 |
| SI `target_accept`, ODE tolerance | posterior unchanged at 0.95 / tighter `rtol` | R8 | R8 |
| SI prior width | contraction + misspecification curve | ladder + R5 | 0 |

Tier 1 needs no held-out-data refits. Where the outline uses a held-out check to decide
between two answers (3.4, 3.5), the known truth decides at Tier 1. Held-out refits belong to
Tier 2.

---

## 2. The runs

Costs are A100-equivalent hours for the whole run (compute, finalize and figures), estimated by
`tier1/cost_estimate.py` from the rates measured on the runs so far (appendix B). The sampler
solves every condition to the latest save time (720 s), including the rate conditions observed
at 150 s, and every estimate assumes that. A run that has started is estimated from its own
progress log. One that has not uses the median of measured runs with the same system and free
parameters. The three-parameter fits were scaled from R1 by the pilot factors until R2 and R6
measured them; both came in far under (appendix B).
The monitor refreshes the estimates every 30 minutes (`Results/Tier1/figures/cost_estimate.json`,
with the history in `job_files/tier1/cost_estimate_history.jsonl`), and this table follows them.
Last updated Sun 2026-09-27 22:00 (`tier1_status.py`). Every run has finished except R3's five
ridge runs.

Two-parameter runs cap compute at 24 A100-h (`max_total_hours`), a runaway guard about twice
their estimates. The three-parameter fits (R2, R6 b and c) run with no cap (decided 2026-09-26),
and so do R3's C14+unsat and C18 runs, diagonal and dense (2026-09-27).

Run folders are `Results/Tier1/<run name>/`, written by `build_tier1_configs.py --plan`.

| ID | Run | System | Free params | Fits | A100-h | Feeds |
|---|---|---|---|---|---|---|
| R0 | smoke test, production stopping rule | C8 | `a1`+`c3` | 1 | 5.7 (measured) | plumbing; chooses the replicate system; centre point of Fig 4 |
| R1 | main-system duo | C14+unsat | `a1`+`c3` | 1 | 7.2 (measured) | Fig 2 fallback; Fig 5 grouped model; Fig 9 full-data cell |
| R2 | **main fit** | C14+unsat | `a1`+`c3`+`a2` | 1 | 16.7 (measured; estimated ~40) | Figs 2, 6, 6b, 7, 8 |
| R3 | `d1`+`d2` known-answer control | C8, C14+unsat and C18, each diagonal and dense | `d1`+`d2` | 6 | ~256: C8 dense 19.3 (measured), C8 diagonal 24.6 (cap), C14+unsat ~63 + ~38, C18 ~60 + ~52 (uncapped; running) | 3.3 case; Fig 6/6b; SI mass matrix |
| R4 | SBC replicates, truth drawn from the prior | C8 | `a1`+`c3` | 10 pilot, then 30 more | 284 (measured: pilot 69, the other 30 215) | Fig 3 |
| R5 | robustness: noise 5/20/40% (10% = R0), prior median shifted +1/+2/+3/+4 prior sd, each shift run from the default start and from the ME1 values | C8 | `a1`+`c3` | 11 | 51 (measured) | Fig 4 |
| R6 | `c3` split test | C14+unsat | see below | 3 | 21.4 (measured; estimated ~87) | Fig 5 |
| R7 | data-type check, 3 cells (R1 is the third) | C14+unsat | `a1`+`c3` | 2 new | 16 (measured) | Fig 9 |
| R8 | `target_accept` 0.95; `rtol` 1e-5; the prior +4 sd fit at `max_steps` 1000 | C8 | `a1`+`c3` | 3 | 19.0 (measured) | SI |
| | **Total** | | | **68 new** | **~680** | |

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
known answers for three things at once (as planned; see the 2026-09-27 decision below):
- Fig 6 must flag both parameters individually as weakly identified.
- Fig 6b's eigen-analysis must find one tight combination and one flat direction.
- Fitting the same data with a diagonal and a dense mass matrix answers the SI's mass-matrix
  question.

The pre-flight already shows the degeneracy. At the sampler's start on C8, the gradients
with respect to the two sampled coordinates (`12*d1` and `d2`) are identical (11.429 each).
On C14+unsat they differ slightly (26.2 vs. 27.3). C14+unsat adds the point where separation
begins. Ridge posteriors can hit the tree-depth ceiling. R3's C8 runs keep the standard
24 A100-h cap, about 4x the C8 estimate (the C14+unsat and C18 runs have none, from
2026-09-27); hitting it under diagonal but not dense is itself the SI result. *Optional:* `d1`+`d2` on C20+unsat, the full model, shows clear
separation. Add it only if the figure needs a third point. It would cost more like ~100 A100-h
than the ~20 first estimated: that figure scaled R1's `a1`+`c3` rate, while `d1`+`d2` on
C14+unsat is running at ~63 A100-h (diagonal, estimate at 22:00) against R1's 7.

*Decision, 2026-09-27: the C8 pair is stopped.* On the exact ridge, warmup steps took 2-82 min,
so both C8 runs were at warmup step 20 of 300 after 9-10 h
and would have reached the 24 A100-h cap long before the dense run's first mass-matrix update
(step 100), with no posterior. They were cancelled at warmup step 20 (15 A100-h spent;
checkpoints kept). Figs 6/6b take their weakly identified example from C14+unsat, which now
runs uncapped (~55 A100-h). The exact C8 answer is covered without sampling by
`identifiability_report.py --selftest`, which recovers a synthetic `12*d1 + d2` ridge. The SI
mass-matrix comparison is dropped, unless a dense C14+unsat run is added (~55 A100-h).

*Reversed the same day.* With `max_steps` 1000 both C8 runs resumed from their step-20
checkpoints (section 4). C8 dense finished (400 draws, 19.3 A100-h), and the diagonal run
continues under its cap. Dense twins of the C14+unsat and C18 runs, submitted 14:20, give the SI
mass-matrix comparison on all three systems.

**R6 (grouping, Fig 5).** `c3` ties six TesA hydrolysis constants with six distinct nominal
values. It is split into short-chain (`c3s`: C4-C8) and long-chain (`c3l`: C10-C14 plus the
unsaturated species) halves in a script-generated variant of the C14+unsat reaction set
(`Reactions/EC_FAS_ME1/C14+unsat+c3split`); the reaction files are never hand-edited. The
runs:
- (a) grouped `a1`+`c3` on the standard data. This is R1, reused.
- (b) split `a1`+`c3s`+`c3l` on the same data. New; ~40 A100-h, as R2 (measured 6.5).
- (c) data generated at `c3s` = 1, `c3l` = 3, fit with both models. New; ~8 + ~40 (measured
  6.6 + 8.3).

At `c3l` = 3 the C10-C14 species move by −15% to +144% while C4-C8 move about 6%, a pattern
a single `c3` cannot produce.

What each part answers. (b), on the standard data, where the six constants really do share
their fitted ratios: the split fit should recover `c3s` ≈ `c3l` ≈ 1, the grouping's 1:1, and gain
nothing on LOO over the grouped fit, so the grouping is supported. (c) is the negative control
from outline 3.2: with a true 1:3 ratio the grouping is wrong, so the split fit should recover
`c3s` ≈ 1 and `c3l` ≈ 3 and LOO should prefer it over the grouped fit. Together they show that
LOO can tell a justified grouping from an unjustified one. The three-parameter cost is borrowed from the main fit, but
`c3s`/`c3l` may be more correlated than that. Measure (b) before running (c). (They were not:
correlation -0.04 in (b).) Gate every
Δelpd on Pareto-k ≤ 0.7. `a1` is included so the posterior has enough spread for LOO to
work.

**R7 (Fig 9).** The expected information is computed for every grid cell on the main-fit
model, at zero sampling cost, and that draws the figure. Three cells are then sampled with
`a1`+`c3`:
- profile only (the cheapest data type)
- the best cell per data point
- the full dataset (R1, reused)

This checks that the information ranking matches sampled posterior contraction.

Built 2026-09-25 as `Tier1 C14+unsat - a1c3 - profile` and `Tier1 C14+unsat - a1c3 - rates`,
each a subset of the standard data. The rates dataset is the best-per-point cell. The five
initial rates carry exactly the information of total fatty acid at 150 s (4.27 nats either
way). For a like-for-like prediction the grid was re-run on the `a1`+`c3` model
(`expected_information_grid_a1c3.json`). Predicted log-scale posterior contraction for `a1` /
`c3`:

| Cell | Points | Information (nats) | `a1` | `c3` |
|---|---|---|---|---|
| full data (R1) | 23 | 6.34 | 0.9991 | 0.9967 |
| profile only | 8 | 5.77 | 0.9975 | 0.9962 |
| rates only | 5 | 4.27 | 0.9928 | 0.9705 |

So the sampled order to check is full > profile > rates, with `c3` separating the cells most.
Per point, the ranking reverses (rates 0.85 nats, profile 0.72, full 0.28).

**R4/R5 on C8.** Calibration and robustness test whether stated uncertainty is honest, which
doesn't depend on network size. R4 starts with a 10-replicate pilot and continues to 40 only
if the pilot's coverage and rank histogram look sane. SBC ranks and coverage come from the
same replicates. The R5 noise-level datasets reuse the 10% dataset's random draws, rescaled,
so the noise axis carries no draw-to-draw scatter. Compare the prior-offset runs on log-scale
contraction (`contraction` in `recovery_report.py`), which doesn't change when the prior
median moves.

The prior shifts run as two series (decided 2026-09-26, before either started):
- **Main: PyMC's default start**, each shifted prior's mean (68x and 219x the truth at +3 and
  +4 sd), jittered ±1 in log space. It asks the question Fig 4 is for: does the workflow still
  succeed when all you know is a wrong prior, as on an uncharacterized system with no
  published estimate to start from? At +4 sd the data give almost no gradient at the start
  (−0.6 for `a1`, −0.5 for `c3`), so warmup must find the data's peak nearly unguided. A run
  whose chains all stay near the prior's mean together could pass r-hat and report a confident
  wrong answer, the one outcome the outline names as undermining the paper. This series is
  what can show it.
- **Paired: started at the ME1 values** (`posterior_sampling.initial_values`; tag `init1`).
  With the start taken out of the question, it measures only how far the prior pulls the
  posterior. That attributes a main-series failure: if the default start fails but its `init1`
  twin recovers, the search is the problem; if both are off, the posterior itself is pulled.
  The start cannot change a converged posterior, and no second region is being skipped: the
  log posterior at the prior's mean is about 760 below its peak (−791 vs −26 at +3 sd, −793 vs
  −33 at +4 sd).
- If a default-start run fails, the follow-up tests remedies: 600 warmup steps, or more widely
  dispersed starts.

Warmup of 300 steps is established for runs that start near the answer (Appendix A; R0 reached
the posterior bulk within 7 steps). It was not yet established for the SBC replicates with
extreme truths (e.g. `sbc004`/`sbc005`, `a1` ≈ 8.7), whose chains start up to ~1.8 log units
away. SBC needs identical settings for every replicate, so if the pilot shows these failing,
the remedy (more warmup or another start) applies to all 40. The pilot passed: its chains,
the extreme truths included, reached their true values within 40-55 warmup steps (section 4),
so all 40 ran with 300.

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
8. Fig 9's information grid, computed at the truth on R2's model (`expected_information_grid.py`,
   `expected_information_grid.json` and `.png`), under the five Tier-1 conditions and the
   Tier-1 noise model. Per data point, total fatty acid at 150 s is the most informative cell
   (1.09 nats per point). Individual species buy most of the attainable contraction (mean
   0.997-0.999, log scale). The ACP intermediates (ketoacyl-, hydroxyacyl-, enoyl- and acyl-ACP) add
   little beyond that at their concentrations under the 0.01 µM floor (0.03-0.06 nats per
   point). The Tier-1 design scores 0.39 nats per point over 23 points (contraction 0.9986 /
   0.9902 / 0.9957 for `a1` / `c3` / `a2`). So R7's "best cell per data point" is total fatty acid at
   150 s, and "cheapest data type" is the endpoint profile.
9. R4's pilot built (`sbc.py generate`): 10 replicates on C8, each truth drawn from the fit's
   own prior, with its data in `Data/Tier1_rates/Chain_C8_sbc<i>/` and its config in
   `Results/Tier1/Tier1 C8_sbc<i> - a1c3/`, all recorded in `sbc_manifest.json`. The truths
   span a1 0.35-8.7 and c3 0.27-9.9, and every replicate's data generated. `sbc.py selftest`
   confirms the rank code: ranks come out uniform on an exact posterior (chi-square p 0.67 and
   0.78), and an over-confident one is caught (p < 10⁻³, 57% coverage at 90%). All 10 pilot
   configs pass the pre-flight. Their clean-data error reaches 8×10⁻⁵ on the extreme truths,
   still far below the noise. Expect replicates 4 and 5 (a1 ≈ 8) to cost more than a typical C8
   run. The sampler starts near the prior's centre, where their log posterior is about −4×10⁴
   with gradients of about 10⁵, so warmup has a long way to travel.
10. jax 0.7.2 does not cure the laptop's JIT abort. A throwaway clone of the env with jax
    0.7.2 (`Bayesian_jax072_test`) aborted the same way in the in-process toy loop. Sampler
    numerics also shift between jax versions (toy seed 0: r-hat 1.74 on 0.7.0, 1.61 on
    0.7.2), so results will not reproduce bit for bit across the pending pin change.

Done 2026-09-25, once CURC was back:
- On the cluster, renamed `job_files/masking_check` → `chain_scaling_tests`, `bench` →
  `multiparam_tests` and `scaling_rank` → `chain_system_sensitivity_analysis`, and updated
  the paths inside them.
- Synced: 441 files pushed and checksum-verified. `--update` left alone the 5 files that were
  newer on the cluster, and those were then pulled into git.
- Cleaned up the cluster (verified duplicates deleted, cluster-only studies archived and
  mirrored in git, loose logs tarred).
- R0 submitted: Blanca job 28506399 runs it on an A100 (`bgpu-biokem2`), and the Alpine twin
  was cancelled.

Done 2026-09-25, while R0 runs:
- `environment.yml` and `requirements.txt` are pinned to Alpine's stack, except pymc 6.1.0
  where Alpine has 6.0.1. Alpine runs pymc 6.0.1 with pytensor 3.1.2, but 6.0.1 declares
  pytensor < 3.1 (`pip check` flags it there), so that pair cannot be installed from pins.
  pymc 6.1.0 declares pytensor ≥ 3.1.2, < 3.2. nutpie comes from pip, because conda-forge has
  no Python 3.12 build of it for osx-64. The conda part solves on osx-64 and linux-64 (dry
  run). Tabled: bringing the laptop env and Alpine's pymc in line with the files.
- R7's two configs (section 2, R7 notes), pre-flighted: profile noise z mean −0.75 sd 0.70,
  rates +0.41 sd 0.74, logp and gradient finite.
- Fig 6/6b module (`identifiability_report.py`). Its `--selftest` recovers a synthetic
  `12*d1 + d2` ridge (tight direction at 5% of prior variance, flat one at 1.0). On the C14
  `a1`+`c3` pilot posterior it gives correlation −0.47.
- `submit_stage2.sh` for Stage 2.
- R4's other 30 replicates (`sbc.py generate --start 10 --n 30`, `sbc010`-`sbc039`), so all
  40 are built. These truths span a1 0.062-3.46 and c3 0.14-5.53. All 30 pass the pre-flight
  (clean-data error ≤ 1.7×10⁻⁵, finite logp and gradient). They are submitted only once the
  pilot passes (Stage 3).
- The cluster-only analysis tools under `job_files/` (`multiparam_tests`,
  `chain_system_sensitivity_analysis`, `warmup_rates.py`, job-ID records) are now in git.

**Stage 1: R0 (5.7 A100-h).** Checks plumbing: resume, stopping on
r-hat + ESS, finalize, figures, `recovery_report.py`. It also decides the replicate system.
Nothing else is queued until R0 has been inspected. Done 2026-09-25 and inspected 2026-09-26
(section 4). Every check passed except resume, which one segment could not exercise; the
first multi-segment Stage-2 run tests it. C8 stays as the replicate system.

**Stage 2: everything cheap and independent, one batch (planned ~225 A100-h; 252 used of ~418
estimated at 22:00 on 2026-09-27, the excess almost all R3's ridge runs).** R1, R3, R5, R7, R8
and the R4 pilot, 29 runs, submitted 2026-09-26 by `tier1/submit_stage2.sh` from `job_files/`
(R7 and R5's ME1-start series were added the same day). It refuses until R0 has finalized, and
it skips runs that are finished or already queued, so rerunning it resumes the ones that
stopped at the wall clock. `tier1/tier1_status.py` shows each run's state, its A100-h used and
estimated, and when it should finish. Four runs were added on 2026-09-27: C18 `d1`+`d2`, the
C14+unsat and C18 dense twins, and the `max_steps` 1000 twin of the prior +4 sd fit, so 33 in all.
- Gate to R2: R1 converges and recovers both parameters. Passed 2026-09-26 (section 4).
- Gate to the full R4: the pilot looks sane. Passed; the other 30 were submitted 2026-09-27.
- Status at 22:00: 28 of 33 finished. R3's C8 diagonal, C14+unsat diagonal and dense, and C18
  diagonal and dense runs are still going (section 4).

**Stage 3: the main fit and its dependents (planned ~290 A100-h; took 254).** All 34 runs have
finished: R2 (~19:50 on 2026-09-27), R6's three fits and R4's other 30 replicates (section 4).
- R2 and R6 (b) submitted 2026-09-27 03:00; both started at once on H200s.
- R6 (c)'s two fits (grouped and split, on the 1:3 data) submitted 2026-09-27, once (b) was
  running at a normal rate.
- Results, 2026-09-27 10:50:
  - R6 (b), the split model on standard data (true `c3s` = `c3l` = 1): `c3s` 0.835
    [0.652, 1.031], z -1.58; `c3l` 1.011 [0.847, 1.194], z +0.09; `a1` 0.998, z -0.06.
    Contraction 0.990 / 0.995 / 0.999. Both halves are identified, and the data show no
    difference between them, as they should. 400 draws, r-hat ≤ 1.003; warmup check OK
    (first-block drift ≤ 0.08 sd). 6.5 A100-h.
  - R6 (c), the grouped model on the 1:3 data (`c3s` 1, `c3l` 3): `c3` 1.277 [1.09, 1.48],
    a compromise weighted toward the short chains; `a1` 0.996 is unaffected. The posterior
    predictive check shows the misfit: C10 is +4.6 noise sd off (3.47 observed against 1.79),
    C8 -3.2 sd (2.3 of it the shared noise draw), the TesA 0.5 µM rate +3.0 sd; 3 of 23 points
    fall outside the 95% predictive interval and p_loo is 7.3 for 2 parameters. (An earlier note
    here had C8 and C10 both over 4 sd; corrected 2026-09-27.) 6.6 A100-h.
  - R2 at 105 sampling draws: too early for the drift test; acceptance over the last 50
    warmup steps 0.777.
  - R2 finished 2026-09-27 (~19:50): converged at 600 draws (checks at 500 and 600 passed),
    finalized at 700, no divergences, 16.7 A100-h by `cost_estimate.py` (plan estimate ~40). Its
    confirmation check at 700 failed (`a1` r-hat 1.0102 against 1.01, 18:32). Its last segment
    started at ~14:30 on the older code, so the failure was logged and ignored, as in the three
    runs reopened earlier (section 0). Reopened at 22:30, it passed the checks at 800, 900
    and 1000 draws (r-hat 1.0094, 1.0069, 1.0054) and finalized at 1000 on 2026-09-28 04:44,
    21.5 A100-h in all. Final: truth inside every 95% interval, `a1` 1.036 (log z +0.79), `c3`
    1.099 (+0.68), `a2` 0.889 (-1.49); contraction 0.9986 / 0.9885 / 0.9953. Correlations: `c3`-`a2`
    -0.81, `a1`-`a2` -0.61, `a1`-`c3` +0.51, so `a2` trades off against both. Unblocks Figs 2, 6/6b, 7 and 8. Fig 9 comparison in
    section 4.
  - R6 (c), the split model on the 1:3 data (13:45): `c3s` 0.835 (truth 1, z -1.61), `c3l`
    2.95 (truth 3, z -0.19), `a1` 0.999; contraction 0.990 / 0.986 / 0.999; 8.3 A100-h. It
    recovers the 1:3 split. `c3s` matches R6 (b)'s 0.835 because both data sets share one
    noise draw.
  - Fig 5 drafted (`tier1_result_figures.py fig5`, `grouping_test.png`). PSIS-LOO, grouped minus
    split: standard data +0.2 ± 1.4 elpd (no preference; the grouping holds), 1:3 data -23.8 ±
    11.8 (2.0 SE, the split predicts better). The 1:3 comparison fails the Pareto-k gate at one
    point (the split fit's C10, k 0.90, carrying 10.9 of the 23.8); setting C10's difference to 0
    still leaves 12.9 ± 6.2 (2.1 SE). ArviZ 1.x `az.compare` rounds its table to 2 significant
    figures by default (it showed -20 ± 12): use `round_to='none'` or the pointwise `elpd_i`.
- The three-parameter fits run with no compute cap (decided 2026-09-26). At R1's measured
  rates and the pilot factors, R2 was estimated at ~40 A100-h: ~10 h of warmup (~119 s per
  step) and ~27 h for ~1100 draws (~88 s per draw), about four 12 h segments. It measured 16.7:
  66.6 s per warmup step, 51.3 s per draw and 700 draws (appendix B). R1 passed its gate on
  2026-09-26 (section 4).
- R2: inspect warmup with `convergence_and_warmup.py` at the first segment boundary. If
  acceptance is well below 0.8, or the first block still drifts, restart with 600 warmup
  steps and record it. Acceptance was 0.777 (10:50 above); R2 was not restarted and finished
  on its 300 warmup steps.
- R6 (b), then (c) once (b)'s cost is known. Done: (b) 6.5 A100-h, then (c) 6.6 and 8.3.
- The remaining R4 replicates (built and pre-flighted), ~160 A100-h at the pilot's median.
  Submitted 2026-09-27 as `submit_stage2.sh`'s group R4b with `--nice 25000`: above the age
  factor's weight (20160), so any other run's waiting segment outranks them however long they
  have waited (they sit at priority 1 while waiting). `tier1_status.py` checks each call that no
  replicate starts while another run waits, and the monitor resubmits clean segment ends
  itself every 10 min, so a finished segment's GPU goes back to its run. All 30 had finished by
  22:00 on 2026-09-27, at 215 A100-h.
  - sbc018 (truth `a1` 0.34, `c3` 0.38) took 71 min for warmup steps 6-10: 59 leapfrog steps
    at 72 s each, against 2 s before and after, with short trees (1-15 steps) and one
    divergence. So the cost was the ODE solves, not tree length. Its chains started far from
    the mode (log density -940 to -2535) and the first three steps diverged at step sizes 2.3
    and 0.23. C8's solver needs 71-188 steps for `a1` and `c3` anywhere in e^-10 to e^10 except
    a band at log `a1` ≈ 7 (`a1` ≈ 1100, ~6 prior sd out): 5389 steps to the full 20000 with
    failed solves. Proposals inside trees reaching that band are the likely cost. It recovered
    by itself (steps 11-20 in 5 min) just before it was cancelled on 2026-09-27 to resume with
    `max_steps` 1000; the cancel lost ~1.5 min, it resumes from step 20, and its nice was set
    to 16000 on Alpine and 4000 on Blanca so it goes ahead of the other replicates but below
    every main run. The band has negligible posterior mass, so the sampled posterior is
    unchanged.
  - Accepted states, every Tier-1 run (`accepted_solver_steps.py`, GPU job 28519196, 2026-09-27;
    `Results/Tier1/figures/accepted_solver_steps.json`): all 67 runs' warmup and sampling states,
    ~170k unique positions, solved at each run's own tolerances with a 20000-step limit. No
    sampling draw in any run needed more than 131 solver steps; warmup states stay at or under
    189 (R8's rtol 1e-5 run, C18 184) except in one run: the C8 prior +4 sd run from the default
    start, whose 6 of 1040 warmup states needed 1000-3087 steps (none failed), the worst at
    warmup step 44, a1 ≈ 350, c3 ≈ 314, a chain drifting toward the misplaced prior (median 110)
    before the data pulled it back. So `max_steps` 1000 would never have cut off a posterior draw;
    only those six warmup excursions, and the rejected proposals inside trees (not stored) that
    it is meant to cut short.
  - R2's space (C14+unsat, `a1` + `c3` + `a2`, rtol 1e-3; 9^3 grid over e^-10 to e^10, 2026-09-27):
    median 90 steps, and at most 163 within ±3 in log (±2.6 prior sd). A band at log `a1` ≈ +6
    (`a1` ≈ 400, ~5 prior sd out) needs 598-7679 steps (5 of 729 points above 1000, none failed),
    like C8's band at log `a1` ≈ 7. R2's slow warmup chunks (a 22-min chunk at steps 175-180, an
    hour without a checkpoint at 185) fit rejected proposals reaching it; its accepted states
    needed at most 97. A 1000 cap suits any rerun of R2 or the full-model fits.
  - Direct check, submitted 2026-09-27: `Tier1 C8 - a1c3 - prior+4sd - cap1000` (R8), the +4 sd
    fit rerun with `max_steps` 1000, same seed and start. The two chains match until a proposal
    needs more than 1000 steps, then take different but equally valid paths, so the test is
    agreement within Monte Carlo error (as in the settings check), not identical draws. It goes
    in the SI settings check as a third variant. Proposed Methods wording: max_steps 20000 (1000
    for the ridge runs); no posterior draw in any fit needed more than 131 steps, and a rerun of
    the one fit whose warmup exceeded 1000 matched within Monte Carlo error.
    Result (2026-09-27 ~21:00): `a1` 1.0656 ± 0.054 against the original's 1.0636 ± 0.055, `c3`
    1.0005 ± 0.062 against 0.9991 ± 0.063: differences of 0.04 and 0.02 posterior sd, within Monte
    Carlo error. The cap does not change the answer.
  - SBC at 18 replicates (2026-09-27 10:50): `a1` z sd 0.66 (one-sided p = 0.021 against 1),
    `c3` 0.82 (p = 0.16); 50% coverage 0.67 and 0.78; rank quantiles uniform by KS (p = 0.36,
    0.76). The eight newest replicates alone give `a1` z sd ~0.78. The data are not the cause:
    (noisy - clean) / sigma pooled over all 40 replicates' 720 points has mean +0.03 and sd
    1.009, and the replicates' noise draws are independent (correlation sd 0.242 against 0.236
    expected). The finished set is not yet a random sample (the slow replicates are still
    running), so the verdict waits for all 40.
    At 22 replicates (13:45): `a1` z sd 0.62 (p = 0.005), `c3` 0.92 (p = 0.33); binned ranks
    `a1` p = 0.010, `c3` p = 0.37; KS on quantiles p = 0.24 and 0.69. Sampled / Laplace sd
    median 1.02 and 1.00 (sbc018's `a1` 0.49: four chains agree, no divergences, r-hat 1.00;
    low `a1` is where the Laplace approximation at the truth is poorest). Mahalanobis² sum
    31.95 on 44 df, P(lower) 0.088. Noise within each replicate is independent too
    (correlations sd 0.160 against 0.158 expected, time-series lag-1 within ±0.28).
  - At 38 replicates (2026-09-27 18:58; sbc022 and sbc039 still running) the over-coverage is
    gone: z sd 0.93 (`a1`, p = 0.29) and 1.05 (`c3`, p = 0.68); coverage 0.61 / 0.89 / 0.95 (`a1`)
    and 0.63 / 0.84 / 0.92 (`c3`) at 50 / 90 / 95%; KS on the rank quantiles p = 0.33 and 0.87. The
    binned chi-square for `a1` is still marginal (p = 0.032). So the earlier z sd of 0.62 came from
    which replicates finished first: the slower ones had the larger errors. sbc029 was reopened
    under the three-pass rule and finalized at 700 draws (c3 z -1.61).
  - At 39 replicates (21:40), sbc022 looked confidently wrong: truth `a1` 0.48 / `c3` 0.75, posterior
    0.197 ± 0.070 (log z -3.66) and 0.652 ± 0.034 (-2.82). Four chains agree, no divergences, but
    it mixed slowly (converged at 1600 draws). Coverage falls to 0.90 / 0.90 at 95%.
    Checked by 22:30 with a fine grid posterior: the posterior has two peaks, the main one at
    `a1` 0.17 / `c3` 0.66 and a minor one at `a1` 0.41 / `c3` 0.73, next to the truth. The grid
    puts 5.3% of the mass in the minor peak and the chains put 7.2% of their draws there, so
    the sampler visited it in about the right proportion. The truth is at quantile 0.992 of the
    grid posterior (0.984 of the draws): a tail draw, not a sampler failure. sbc022 stays in.
  - The fixed-truth data sets (C8, C12, C14+unsat, C18, the noise levels) share one noise
    draw, rescaled by the noise level: common random numbers across systems. Their fits are
    one realisation, so e.g. every C8 fit's `a1` z of ~+0.9 is the same offset, and coverage
    across them is not an independent test; SBC is.

**Stage 4: post-processing (forward solves).** Figs 6/6b from R2 and R3. Figs 7 and 8 from
R2's posterior (`posterior_morris.py`, `posterior_ratio_response.py`). Fig 9's grid is done
at the truth (item 8), and R7 checks it against sampled contraction.

**Stage 5: draft every figure, then choose what goes to Tier 2** (section 6).

---

## 4. Results

### R0: C8, `a1`+`c3` (Blanca job 28506399, 2026-09-25)

One segment on an A100 (`bgpu-biokem2`), 5 h 46 min of wall time. The figures are in
`Results/Tier1/Tier1 C8 - a1c3/`, all of them together in `run_summary.png`.

**Recovery.** Both parameters recovered, truth inside the 95% interval for both:

| | Truth | Posterior mean ± sd | z (log scale) | Contraction (log scale) |
|---|---|---|---|---|
| `a1` | 1.00 | 1.049 ± 0.055 | +0.89 | 0.9980 |
| `c3` | 1.00 | 0.981 ± 0.062 | −0.34 | 0.9971 |

The `a1`-`c3` posterior correlation is +0.45 (`identifiability_report.py`). In the
prior-standardised eigendecomposition, the two directions keep 0.13% and 0.36% of the prior
variance.

**Convergence.** The stopping rule fired at 700 sampling draws (checked every 100), and the run
finalized at 800. r-hat is 1.008 / 1.002 and bulk ESS 1376 / 1567. r-hat set the run's length,
not ESS: it held at 1.013-1.018 from 100 to 500 draws, while bulk ESS passed 400 at about 300.
Checked afterwards at 10-draw resolution, the criteria are first met at 530 sampling draws.
- No divergences. BFMI is 1.06-1.18 and tree depth never exceeds 4.
- About 4.3 leapfrog steps per draw. Acceptance is 0.91-0.93, above the 0.8 target.
- The four chains agree: the spread of their means is 5-6% of the posterior sd, about what
  Monte Carlo noise alone gives.

**Fit.** LOO gives elpd_loo −15.74 ± 8.26 and p_loo 1.42, with all 18 Pareto k ≤ 0.7 (the
largest is 0.32). The posterior predictive matches the time series and all five initial rates.
The profile's C8 observation sits about 2 noise sd below the posterior mean. The truth and the
posterior agree, so that is a low noise draw, not misfit.

**Cost: 5.7 A100-h for the whole job**, 4.95 of it sampler compute (one A100 40GB segment).
- Warmup: 1.73 h, 21 s per step.
- Sampling: 3.05 h, 800 draws at 13.7 s each. It converged at 700 and finalized at 800 under
  the stopping rule (two passing checks, then one more block).
- Startup took about 11 min, and finalize 46 min (log-likelihood, prior samples and posterior
  predictive).
- Every condition, the four observed at 150 s included, is solved to 720 s. A shorter horizon
  would save little: in `cost_profile.py`, going from 50 to 150 s adds about 7% per gradient,
  since most solver steps fall in the early transient.
- These are the rates behind every fixed-truth C8 estimate in section 2 until more C8 runs
  finish.

**Gate checks.**
- Stopping on r-hat + ESS, finalize and `recovery_report.py`: all worked.
- Figures: the pipeline did not draw them, because `inference_runner.py` never calls the
  plotting code. They were drawn afterwards, and `tier1.sbatch` now draws them for every
  finished run.
- Resume: not exercised, since R0 finished in one segment.
- Replicate system: C8 stays.

---

### Stage 2 (2026-09-26 to 27): 28 of 33 finished (22:00)

All 29 finished runs (R0 and 28 of Stage 2's 33) have the truth inside the 95% interval for
every parameter.

*Scores (changed 2026-09-27).* z and posterior contraction follow Schad, Betancourt & Vasishth
(2021, eqs. 4-5): z = (posterior mean − truth) / posterior sd, and contraction = 1 − posterior
variance / prior variance. For the LogNormal groups both are computed on log(x), where the
prior is Normal with sd ln(10)/1.96 wherever its median sits; on the natural scale a LogNormal's
variance is dominated by its upper tail, so contraction there tracks where the posterior sits
as much as what the data taught (sbc002's `c3`, at 9.9, reads 0.28 natural against 0.955 log).
Earlier versions of this plan quoted 1 − posterior sd / prior sd; the rankings are unchanged. R1 passed the gate to R2. Only R3's five ridge runs are still going (status at
22:00 below).

| Run | `a1` z | `c3` z | Contraction `a1` / `c3` | Draws | A100-h |
|---|---|---|---|---|---|
| R1: C14+unsat | −0.05 | −0.97 | 0.9990 / 0.9966 | 400 | 7.2 |
| R7: C14+unsat, profile only | +1.07 | −1.61 | 0.9971 / 0.9952 | 400 | 6.5 |
| R7: C14+unsat, rates only | −0.82 | +1.01 | 0.9930 / 0.9314 | 600 | 9.8 |
| R5: C8, noise 5% | +0.88 | −0.21 | 0.9994 / 0.9990 | 600 | 4.1 |
| R0: C8, noise 10% | +0.89 | −0.34 | 0.9980 / 0.9971 | 800 | 5.7 |
| R5: C8, noise 20% (reopened 2026-09-27) | +0.91 | −0.43 | 0.9925 / 0.9909 | 800 | 5.4 |
| R5: C8, noise 40% | +0.85 | −0.52 | 0.9686 / 0.9621 | 500 | 4.4 |
| R5: C8, prior +1/+2/+3/+4 sd, default start | +0.93 / +1.06 / +1.10 / +1.16 | −0.26 / −0.17 / −0.09 / −0.05 | 0.9979-0.9980 / 0.9971-0.9972 | 500-800 | 3.7-6.5 |
| R5: C8, prior +1/+2/+3/+4 sd, ME1 start | +0.97 / +1.03 / +1.09 / +1.18 | −0.25 / −0.20 / −0.12 / −0.06 | 0.9980 / 0.9971-0.9973 | 400-800 | 3.5-4.9 |
| R8: C8, target acceptance 0.95 (reopened 2026-09-27) | +0.88 | −0.37 | 0.9980 / 0.9972 | 800 | 6.5 |
| R8: C8, `rtol` 1e-5 | +0.91 | −0.32 | 0.9980 / 0.9972 | 500 | 7.6 |
| R8: C8, prior +4 sd, `max_steps` 1000 | +1.22 | −0.02 | 0.9981 / 0.9973 | 500 | 4.9 |
| R4: SBC pilot, 10 replicates | −1.31 to +0.84 | −1.06 to +0.58 | 0.9914-0.9999 / 0.9552-0.9980 | 400-1100 | 4.0-10.7 |
| R3: C8 `d1`+`d2`, dense (columns are `d1` / `d2`) | +0.05 | +0.02 | 0.510 / 0.509 | 400 | 19.3 |

z and contraction are on log scale here (see the note on scores above the table), except R3's
`d1` and `d2`, whose priors are Normal and which are scored on the natural scale.

- **R1 (gate to R2).** Converged at 300 draws (r-hat 1.0049, ESS 758), finalized at 400. No
  divergences, BFMI 0.97-1.12, elpd_loo −22.15 ± 7.13. Correlation +0.03.
- **Fig 9 check.** Sampled log-scale contraction matches the expected-information prediction to
  within 0.001 on the full data (0.9990 / 0.9966 against 0.9991 / 0.9967) and the profile
  alone (0.9971 / 0.9952 against 0.9975 / 0.9962). With rates alone, `a1` matches (0.9930
  against 0.9928) but `c3` is wider than predicted (0.931 against 0.971). Its posterior is
  skewed (mean 1.45, sd 0.85, log-scale z +1.01, correlation +0.32), where the local, linear
  prediction does not reach. Fig 9 should say that its predictions hold where the posterior is
  close to Gaussian. The skew comes from saturation. Almost all the rates' `c3` information is
  in the TesA 0.5 µM condition, whose rate goes 1.16, 2.10, 3.33, 4.15, 4.50 µM C16/min at
  `c3` = 0.25, 0.5, 1, 2, 4, with sensitivity d log(rate) / d log(`c3`) 0.91, 0.79, 0.50,
  0.18, 0.07. Below the truth the rate falls steeply; above it, TesA stops limiting and the
  rate flattens (2 to 4 moves it 8%, under the 10% noise), so large `c3` fits almost as well.
  With the full data, the chain-length profile, which TesA also shapes, closes the tail.
  `joint_posterior.png` shows it.
- **Noise.** The posterior widens with the noise and the truth stays covered: `a1`'s sd is 0.029,
  0.055, 0.112 and 0.252 at 5, 10, 20 and 40%. The z-scores barely move (+0.85 to +0.91),
  since every noise level reuses the 10% dataset's standardized draws.
- **Prior shift.** From the default start (the shifted prior's mean) the chains found the truth
  at every shift, with posteriors matching the ME1-start series (z within 0.04). The start does
  not matter here. The prior pulls both parameters up slightly as it moves (`a1` z +0.89 at no
  shift, +1.16 at +4 sd; `c3` −0.34 to −0.05), with contraction unchanged. That is about +0.07
  z per prior sd, a real, small prior pull, as the correlation-aware Gaussian formula predicts
  (0.068 / 0.073).
- **Settings.** Target acceptance 0.95 and `rtol` 1e-5 both reproduce R0 (z within 0.03).
  The 0.95 run was reopened under the three-pass rule and finalized at 800 draws: `a1` 1.0493,
  `c3` 0.9793, against R0's 1.0492 / 0.9806. `rtol` 1e-5 moves the log-likelihood by ~1e-4 nats.
  They cost 1.1x (lockstep leapfrog steps) and 1.85x per draw. The `max_steps` 1000 twin of the
  +4 sd fit matches it within 0.04 posterior sd (section 3).
- **Warmup.** `convergence_and_warmup.py --summary` passes R1, sbc000, sbc004 and sbc005:
  acceptance 0.777-0.780 over the last 50 warmup steps, first-block drift ≤ 0.15 sd, and
  dropping the first block moves r-hat by ≤ 0.006. The pilot's chains reached their true values
  within 40-55 warmup steps, the extreme truths (`a1` ≈ 8, `c3` ≈ 10) included.
- **SBC pilot: every truth covered, intervals look too wide.** All 10 replicates cover the
  truth at 90% and 95% (50%: 8/10 and 9/10). But the log-scale z-scores have sd 0.52 against 1,
  and the sum of Mahalanobis² is 9.33 on 20 df (lower-tail p = 0.02). The rank chi-square
  (p = 0.009 / 0.018) is fragile at one replicate per bin. `sbc_width_check.py` rules out the
  two obvious causes:
  - The realized noise matches the stated σ: z sd 1.02 over 180 points, with seeds 0-9 and
    uncorrelated noise vectors.
  - Each sampled posterior sd matches the Laplace (Fisher) sd at its truth: ratios 0.93-1.15,
    median 1.02.

  So the fit gives the width the likelihood implies, and the small errors are most likely
  chance. The other 30 replicates settle it. (With all 40 finished the z sd is 1.07 (`a1`) and
  1.11 (`c3`); Stage 3 below.)
  sbc000 uses data seed 0, R0's seed, so it reuses
  R0's standardized noise at a different truth (its z, +0.84 / −0.39, echoes R0's +0.89 / −0.34). That is
  harmless for SBC, since the noise is drawn independently of the truth.
- **R3 with the dense mass matrix, 2026-09-28** (`Results/Tier1/figures/d1d2_dense_mass_matrix.png`,
  `tier1_result_figures.py r3_dense`). All three dense runs finished sampling; none of the
  diagonal twins has. Posterior in prior-standardised units against the expected-information
  prediction: C8, loose direction 98% of prior variance left (predicted 100%), tight 0.18%
  (0.17%), per-parameter contraction 0.51 / 0.51; C14+unsat, loose 67% (67%), tight 0.14%
  (0.11%), contraction 0.67 / 0.66, `d1` z −0.92, `d2` +1.05; C18 predicted loose 6.6%, tight
  0.06% (correlation −0.98), sampled posterior still finalizing. So the pair separates as the
  network grows, as predicted, and the dense fits match the prediction. Compute (whole-run,
  `cost_estimate.py`): dense 19.3 / 16.1 / ~20.0 A100-h on C8 / C14+unsat / C18; diagonal C8 hit
  its 24.6 A100-h cap in warmup (~79 needed), C14+unsat 51.5 used of ~67, C18 34.2 of ~62. The
  run-summary plots had failed on these runs (the joint posterior put the signed `d` parameters
  on log axes); `inference_plotting.plot_joint_posterior` now uses linear axes for them.
- **R3 on the ridge** (`Results/Tier1/figures/r3_step_size_and_cost.png`). On C8, warmup steps
  took 2 to 82 min each. Trees stayed modest (depth at most 8; 240-350 leapfrog steps per
  5-step chunk on the slowest chain), and step sizes followed almost the same history as on
  C14+unsat (same seed and starts; 0.01-0.3, dual averaging swinging on the ridge's thin
  direction). What changed was the cost of each gradient: about 2 s in steps 6-10, about 65 s
  in steps 11-15 and 15 s in steps 16-20, against a steady 3.3-5.4 s on C14+unsat. The cause,
  from solver-step counts at the runs' own settings: an overshooting leapfrog step lands far
  off the ridge, and C8 has a stiff band there that C14+unsat lacks. Both systems take ~95-100
  solver steps along the ridge and out to ±5 prior sd across it; at +10 prior sd across
  (12·`d1` + `d2` ≈ +17, TesA binding ~10⁷-fold slower) C8 takes 2860, C14+unsat 100. With
  the chains in lockstep, one such solve stalls all four; the proposal is rejected, but its
  cost is paid. Beyond about +5 both systems sequester all their ACP as acyl-ACP of the
  longest chain (free ACP 0, fatty-acid output ~0), so sequestration marks the region but does
  not by itself explain why only C8's solver crawls there. After 9-10 h both C8 runs were at
  warmup step 20 of 300.
  - A cheap guard for future fits: scan solver steps on a coarse grid well away from the
    posterior before submitting (a few CPU minutes), and consider a lower `max_steps` (it is
    20000; healthy solves take ~100), so a runaway solve fails fast and is rejected instead of
    stalling the chains.
  - Test, 2026-09-27: the C8 dense run resumed from its step-20 checkpoint with `max_steps`
    1000 (solver settings are outside the resume check; the posterior region needs 95-180
    steps, so only far-off proposals are affected). If its cost per leapfrog step stays near
    2 s, the diagonal run can resume the same way for the SI mass-matrix comparison; that waits
    on the Monday discussion.
  - Result, 2026-09-27 07:30 (warmup steps 20-35 on an A100 40GB): 4-5 s per leapfrog step on
    steps 25-35 (~11 s on steps 20-25, including the segment's start), against 13-103 s at
    20000 and 2 s where solves are healthy. Divergences are unchanged (3 in 60 transitions,
    2 in 20 before), acceptance is on target (0.77-0.79) and the chains are still spread along
    the ridge. Steps 35-55 (to 10:50) cost 5-14 s per leapfrog as the chains spread further
    along the ridge (to ±1.6 prior sd), ~10 min per warmup step, so it reaches its 24.5 A100-h
    cap near warmup step 120: past the first dense metric update at 100, not the one at 150.
    `max_steps` stayed 1000 from step 20 on (one job, config unchanged). Two separate changes
    followed, and only the second is the metric's:
    - Steps 60-100, still with the identity metric (the dense and diagonal runs are the same
      algorithm until step 100): cost per leapfrog fell back to 2.4-2.6 s, ~1.8 min per warmup
      step. Not the cap alone (it was in place for steps 20-55 too, at 4-14 s), so most likely
      the chains' positions stopped sending proposals into the band.
    - Step 100, the first dense metric update: step size 0.02-0.15 -> 0.26-3.3, and 4-8x fewer
      leapfrog steps per warmup step (200-260 per 5 steps -> 30-60), so ~0.4 min per warmup
      step at the same 2.2 s per leapfrog. Warmup ended by 13:45 at 17.2 A100-h; sampling at
      15 s per draw. Divergences only in bursts right after the metric updates at 150 (4 of 20)
      and 250 (5), when the step size restarts. The chains still spread ±1.4 prior sd along the
      ridge.
    So the cap (and whatever changed at step 60) made C8 affordable, and the dense metric made
    each warmup step ~4x cheaper on top by lengthening the step, not by avoiding the band. At
    steps 60-100's rate, a diagonal run with the cap might also finish warmup (~6 A100-h for 200
    steps) if a diagonal metric cannot align with the ridge and keeps the step small; that is
    the untested half of the SI comparison, and the held diagonal run with `max_steps` 1000
    would test it. The step-length effect is what matters for the diagonal R3 runs on
    C14+unsat (sampling at 240 s per draw, step sizes 0.02-0.06) and C18 (warmup, 225 s per
    step): neither has a stiff band, and their ridges are strongly correlated (predicted
    -0.996 and -0.980), which a diagonal metric cannot follow.
  - Submitted 2026-09-27 14:20: the diagonal C8 run resumed from its step-20 checkpoint with
    `max_steps` 1000 (same 24 A100-h cap as the dense run, 6.1 used), and dense twins of
    C14+unsat and C18 `d1`+`d2` (`- dense`: same seed, start, data and tolerance as the
    diagonal runs, uncapped, `max_steps` 1000 as a guard). The diagonal C14+unsat and C18 runs
    continue, so each system has both halves of the mass-matrix comparison.
  - C8 dense converged at 400 draws (r-hat 1.0039, ESS 1210), ~19.3 A100-h. The diagonal twin
    resumed from nearly the same step-20 state with the same seed and cap. Checked at step 30:
    identical tree sizes at every step, positions within 0.008, but not bit-identical. One chain
    first differs at step 14 (the two runs' first segments ran on different GPU models), two more
    at steps 25-29, so floating-point differences will separate the paths before step 100. Still
    a fair comparison: same start, seed, algorithm and cap, differing at the noise level until the
    first metric update, then in the metric. (An earlier note here said they would retrace
    exactly; corrected 2026-09-27.)
  - Status at 22:00, 2026-09-27 (`tier1_status.py`):
    - C8 dense: finished, 400 draws, 19.3 A100-h. Per-parameter contraction 0.51 / 0.51 against
      0.50 / 0.50 predicted for an exact ridge; z +0.05 / +0.02.
    - C8 diagonal: warmup 100 of 300, its first diagonal metric update; 13.2 of its 24.6 A100-h
      cap. It tracks the dense twin closely, but not bit for bit (floating point, different GPUs).
    - C14+unsat diagonal: sampling, 225 draws; r-hat 1.064 and ESS 98 at 200 draws; ~3 min per
      draw; 37 of ~63 A100-h.
    - C14+unsat dense: warmup 190 of 300 at ~38 s per step (314 s before its first dense update);
      warmup ends ~23:10.
    - C18 diagonal: warmup 240 of 300 at ~225 s per step; warmup ends ~01:45 Monday.
    - C18 dense: warmup 100 of 300, at its first dense update; ~263 s per step before it.
  - Preemption: Blanca's PreemptMode is REQUEUE, so a preempted copy goes back on the Blanca
    queue only (same job id and submit time, claim kept, resumes from its checkpoint); its
    Alpine twins were cancelled when it first started. Only a clean segment end is resubmitted
    to all four queues (by the monitor). sbc036-039 were preempted at 14:51 on 2026-09-27.
- **Where the chains went** (`Results/Tier1/figures/r3_warmup_ridge.png`: top, the two
  parameters against each other in prior-sd units, with the prior's 95% circle and the
  predicted posterior; below, each parameter's trace per chain). On both systems the four
  chains move from the prior's centre straight onto the ridge and spread along it, each in a
  different place: 12·`d1` + `d2` sits within ±0.1 prior sd of the truth from about warmup
  step 7, while the other direction ranges over ±2 prior sd. None stuck, none bunched, and
  they swap places along the ridge (they mix).
  - How to read it: an identifiable pair is a compact cloud around the truth, much smaller
    than the prior circle in every direction, with narrow overlapping traces (R1's `a1`/`c3`).
    A non-identifiable pair is a ridge, thin across (the combination the data fix) and as long
    as the prior allows along it, with wide wandering traces that mirror each other, since
    moving along the ridge trades one parameter against the other (correlation −0.997). C8's
    predicted ridge runs from one side of the prior circle to the other; C14+unsat's ends
    inside it, a third of the prior variance removed along it: weakly identified.
  - On C14+unsat the chains sit off-centre along the ridge (12·`d1` ≈ −0.5, `d2` ≈ +0.6 prior
    sd), with the truth still inside: along the direction the data barely see, noise and the
    prior set where the mass sits. One chain ran past the predicted end in warmup; the sampled
    posterior will show whether the ridge is longer than the linear prediction.
- **Predicted before sampling** (`predict_identifiability.py`, the Fig 9 calculation for the
  sampler's own coordinates; `identifiability_predicted.json` in each run folder):

  | | Contraction `d1` / `d2` | Correlation | Along 12·`d1` + `d2` | Along the other direction |
  |---|---|---|---|---|
  | C8 | 0.499 / 0.499 | −0.997 | 0.998 | 0.000 (all prior variance left) |
  | C14+unsat | 0.665 / 0.659 | −0.997 | 0.999 | 0.325 (0.675 left) |

  C8 reproduces the known answer. On C14+unsat the per-parameter cutoff of 0.5 would not flag
  either parameter, and only the eigen-analysis shows a direction keeping two thirds of the prior
  variance: the case Fig 6b exists for, if the sampled posterior agrees.
- **Which system separates `d1` and `d2`, and which is clean to sample** (2026-09-27). The pair
  enters TesA's binding constants as 1/exp(k·`d1` + `d2`), with k = 12 for C4-C12 and
  2·chain − 12 above (16 at C14, 20 at C16, 24 at C18, 28 at C20), so C8-C12 are exact ridges
  and each longer system adds a step that breaks the tie. Predicted with
  `predict_identifiability.py --system`, and solver steps scanned out to ±20 prior sd across
  the ridge at the runs' own settings:

  | System | Contraction along the loose direction | Per parameter `d1` / `d2` | Stiff band off the ridge |
  |---|---|---|---|
  | C8, C12 | 0.00 (exact ridge) | 0.50 / 0.50 | yes: C8 5733 steps at +7.5, 2860 at +10, 539 at +12.5, the full 20000 and a failed solve at +15; C12 4199 at +10 |
  | C14, C14+unsat | 0.32-0.33 | 0.66 / 0.65 | no (C14+unsat ≤ 172) |
  | C16 | 0.76 | 0.88 / 0.87 | not scanned |
  | C18 | 0.93 | 0.97 / 0.96 | no (≤ 138) |
  | C20 | 0.93 | 0.97 / 0.96 | not scanned |
  | C20+unsat | 0.97 | 0.99 / 0.98 | not scanned |

  For Figs 6/6b's range: C14+unsat as the weakly identified case (running), and C18 as the
  clearly identified one (clean solver, about C14+unsat's cost per gradient and less
  ridge-like). C20 adds nothing over C18, and C20+unsat little, at 1.8-3x the cost. An exact
  ridge on C12 would hit the same stiff band as C8. C18 was submitted uncapped on 2026-09-27
  with its chain-ladder solver settings, rtol 1e-5 (C14+unsat's config has 1e-3); new noisy
  data in `Data/Tier1_rates/Chain_C18/`, generated at rtol 1e-8.
- **The C8 runs' segments.** The dense run's first segment stopped at 7.4 h, at step 15,
  because its next 5-step chunk would not fit the 11.5 h budget; it was resubmitted. The two are identical until the
  dense run's first mass-matrix update at warmup step 100 (then 150 and 250, BlackJAX's schedule
  for 300 steps). Both should reach the 24 A100-h cap before then. On C14+unsat (3-9 min per
  step, at step 130) warmup should end on Sunday afternoon at about the cap; finishing needs
  ~55 A100-h. Decided 2026-09-27: the C8 pair is stopped and C14+unsat runs uncapped (section
  2, R3 notes). Superseded the same day: with `max_steps` 1000 neither C8 run reached the cap
  by step 100, and the dense one finished at 19.3 A100-h. C14+unsat diagonal is now estimated at
  ~63 A100-h (status above).

---

### Stage 3 (2026-09-27): all 34 finished

R2, R6's three fits and R4's other 30 replicates, 254 A100-h in all (planned ~290). Details
and the day's notes are in section 3; this is the summary.

| Run | z | Contraction | Draws | A100-h |
|---|---|---|---|---|
| R2: C14+unsat, `a1` / `c3` / `a2` | +0.79 / +0.68 / −1.49 | 0.9986 / 0.9885 / 0.9953 | 1000 | 21.5 |
| R6 (b): split, standard data, `a1` / `c3s` / `c3l` | −0.06 / −1.58 / +0.09 | 0.9991 / 0.9899 / 0.9946 | 400 | 6.5 |
| R6 (c): grouped, 1:3 data, `a1` / `c3` | −0.13 / none (no single truth) | 0.9991 / 0.9956 | 500 | 6.6 |
| R6 (c): split, 1:3 data, `a1` / `c3s` / `c3l` | −0.05 / −1.61 / −0.19 | 0.9991 / 0.9902 / 0.9861 | 400 | 8.3 |

- **R2, the main fit.** Finished ~19:50. Truth inside every 95% interval: `a1` 1.036, `c3`
  1.098, `a2` 0.890. Converged at 600 draws, finalized at 700, no divergences. Its confirmation
  check at 700 failed (`a1` r-hat 1.0102) and was ignored by the older code its last segment ran
  on, so it is not yet finished under the three-pass rule (section 3). It was reopened at
  22:30 with up to 1000 more draws, like the other three, and finalized at 1000 on 2026-09-28
  04:44 (21.5 A100-h in all): `a1` z +0.79, `c3` +0.68, `a2` −1.49, contraction 0.9986 / 0.9885 /
  0.9953, within 0.02 z of the 700-draw answer. Figs 2 and 6 are redrawn. Correlations `c3`-`a2`
  −0.81, `a1`-`a2` −0.61, `a1`-`c3` +0.51. In prior-standardised units the loosest direction
  (mostly `c3` against `a2`) keeps 1.5% of the prior variance and the other two 0.08% and 0.15%
  (`identifiability.json`), so no parameter is weakly identified. Fig 2's final panel is `posterior_vs_truth_C14+unsat_a1c3a2.png`. R2
  unblocks Figs 2, 6/6b, 7 and 8.
- **Fig 9 on the main fit.** Sampled contraction against the expected-information prediction
  for the Tier-1 design (Stage 0 item 8): 0.9986 / 0.9885 / 0.9953 against 0.9986 / 0.9902 /
  0.9957. In posterior sd that is 1.00x, 1.08x and 1.05x the prediction: `a1` matches, `c3`
  and `a2` are 8% and 5% wider. So the zero-cost prediction holds to about 10% in sd on the model
  Fig 9 is drawn on. Its three-parameter rates-only cell is still unsampled.
- **R2 dense twin (added 2026-09-28).** The main fit rerun with a dense mass matrix and nothing
  else changed (`Tier1 C14+unsat - a1c3a2 - dense`: same seed, data, tolerances, stopping rule
  and uncapped compute; `build_tier1_configs.py` plan_runs, group R2 in `submit_stage2.sh`), to
  see whether the answer depends on the metric where the pairs are most correlated (`c3`-`a2`
  r = −0.81). Compare posterior means and sds against R2 in Monte Carlo error units, as R8 does.
- **Figs 7 and 8 (outline 3.4-3.5), 2026-09-28.** Forward solves over 50 draws of R2's final
  posterior at the scripts' defaults, on the cluster (`tier1/posterior_figs.sbatch`, submitted by
  `tier1/wait_then_fig78.sbatch` once R2's new posterior was written; 5 minutes on one GPU, no
  failed solves). Fig 7 (`enzyme_sensitivity.png`): the Morris ranking of the nine enzymes at the
  point estimate (= the truth at Tier 1) holds across the posterior. The top targets keep their
  rank in every draw: FabH, FabF, FabZ for total production; TesA, FabF for chain length; FabB, FabI
  for unsaturated fraction. Only near-tied pairs trade places (FabI/FabA in 26% and 76% of draws
  on total production and chain length, FabH/FabA in 4% on unsaturated fraction); rank correlation
  with the point estimate at least 0.98 in every draw. Fig 8 (`ratio_strategy.png`): raising
  (FabF, FabB) against TesA at a fixed geometric mean lengthens the chains in all 50 draws, 10.2 to
  13.8 carbons over 0.1-10x, slope 1.73 [1.68, 1.78] carbons per decade (point estimate 1.76). The
  optimisation half (the enzyme levels giving the longest and shortest chains) is uninformative as
  built: unconstrained over 0.1-10x each, every draw picks the same corners (FabF 10x, FabB and TesA
  0.1x for the longest; FabF and FabB 0.1x, TesA 10x for the shortest). It needs the constraint
  Mains et al. 2022 used (asked at the 2026-09-28 meeting). The posterior is tight (each parameter
  known to 4-13%), so agreement is the expected Tier-1 outcome; Tier 2 is the real test.
- **R6, Fig 5.** The split model reads the ratio: `c3l`/`c3s` 1.21 [0.92, 1.62] on the standard
  data, and 3.53 [2.44, 4.96] on the 1:3 data, with P(ratio > 1) = 1.00. The grouped fit on the
  1:3 data settles on a compromise, `c3` 1.277 [1.09, 1.48], that excludes both 1 and 3; `a1` is
  unaffected, and the misfit shows only in data space (3 of 23 points outside the 95% predictive
  interval). PSIS-LOO, grouped minus split: +0.2 ± 1.4 on the standard data, −23.8 ± 11.8 on the
  1:3 data (2.0 SE; the Pareto-k caveat is in section 3). R6's new compute was 21.4 A100-h
  against ~87 planned. Fig 5 is drafted (`grouping_test.png`).
- **R4, Fig 3: SBC.** All 40 replicates finished (284 A100-h: pilot 69, the other 30 215).
  - sbc022, the most extreme replicate: truth `a1` 0.48 / `c3` 0.75, posterior `a1` 0.197 ±
    0.070 (z −3.66) and `c3` 0.652 (z −2.82). Its posterior has two peaks: `a1` 0.17 / `c3` 0.66
    (log-likelihood −9.1) and `a1` 0.41 / `c3` 0.73 (−13.3), by the truth. The data match the
    truth (residuals at the truth within ±2.2 sd). A fine grid posterior puts 5.3% of the mass in
    the minor peak; the chains put 7.2% of their draws there. The truth is at quantile 0.992 of
    the grid posterior (0.984 of the draws), inside the 99% interval. So the sampler handled a
    two-peak posterior correctly and the truth fell in its tail. z misleads on a posterior like
    this, so the confidently-wrong flag now reads the posterior's own quantiles
    (`recovery_report.py`: truth outside the central 99% interval with contraction > 0.5; it was
    |z| > 2). sbc022 was left out of Fig 3 while this was checked and is back in.
  - On all 40: z sd 1.07 (`a1`, p 0.50 against 1) and 1.11 (`c3`, p 0.33). Coverage at 50 / 90 /
    95%, from the central posterior intervals as Fig 3 plots it: 24 / 35 / 36 of 40 (`a1`) and
    26 / 34 / 36 (`c3`); from the log-scale z: 24 / 35 / 37 and 25 / 33 / 36. KS on the rank
    quantiles p 0.23 and 0.78. ECDF difference against a simultaneous 95% band (Säilynoja et al.
    2022): p 0.33 and 0.67; the same on the log-likelihood (Modrák et al. 2023): p 0.37. Binned
    chi-square: `a1` p 0.035 (a central hump), `c3` p 0.19.
  - The confidently-wrong flag fires once in 80 parameter checks: sbc035's `c3` (truth 5.53,
    median 3.34, quantile 0.998). At the 99% level 0.8 are expected by chance.
  - The extreme ranks cluster at low `a1`: for the 10 replicates with `a1` truth below 0.5 the
    mean |2q − 1| is 0.58, against 0.36 for the other 30 (0.50 expected; Spearman p 0.03). That
    is where the posterior is widest and least Gaussian (sbc018, sbc022). Two follow-ups are on
    hold (2026-09-27): rerunning those 10 with 8 chains, and extending to 100 replicates.
  - The earlier over-coverage (z sd 0.62 at 22 replicates) came from which replicates finished
    first. sbc029 was reopened under the three-pass rule and finalized at 700 draws (`c3` z −1.61).

## 5. Code status

**Exists** (all in `job_files/tier1/` unless noted):
- `build_tier1_configs.py`: `--plan` writes every config in section 2 except R4's (`sbc.py`
  builds those); `--datasets` fits a subset of the standard data (R7)
- `make_tier1_rate_data.py`: the data above, including off-grouping and noise-level variants
- `make_c3_split_variant.py`: R6's reaction variant, with an exact-equivalence check
- `check_model_vs_data.py`: pre-flight (model vs. clean data, noise z, logp/gradient)
- `multimodality_toy.py`: 3.1's known-answer toy
- `recovery_report.py`: per run and parameter, z and posterior contraction (log scale for LogNormal parameters, natural scale kept for reference) and
  50/90/95% coverage, scored against each run's recorded truth
- `tier1.sbatch`, `convergence_and_warmup.py`
- initial-rate, C16-equivalent and mole-fraction observables in
  `Calculation Files/Full_FAS/FA_conc.py`
- `Calculation Files/Full_FAS/FA_acylACP_conc.py`: everything in `FA_conc.py`, plus ketoacyl-,
  hydroxyacyl-, enoyl- and acyl-ACP per chain length and saturation. Each sums the free form
  and the enzyme complexes a quench would release. The exposed set is the `MEASURED_FORMS`
  table at the top.
- `forward_model.py`: the batched forward solver shared by Figs 7-9. It is compiled once, so
  it takes any scaling values and initial conditions, and it loads posterior draws.
- `expected_information_grid.py` (Fig 9), `posterior_morris.py` (Fig 7),
  `posterior_ratio_response.py` (Fig 8; both run on the cluster by `posterior_figs.sbatch`, and
  `wait_then_fig78.sbatch` submits it once a run's new posterior is written),
  `identifiability_report.py` (Figs 6/6b: contraction, correlations, prior-standardised covariance
  eigen-directions), and `plot_tier1_drafts.py` for Figs 6 and 9 (Figs 7 and 8 are
  `tier1_result_figures.py fig7` / `fig8`)
- `sbc.py` (Fig 3): `generate` draws truths from the prior and builds each replicate's data and
  config. `ranks` scores finished runs (rank histograms, a uniformity test, coverage and a draft
  figure). `selftest` checks the rank code on a known answer.

- `submit_stage2.sh`: Stage 2's 33 runs and Stage 3's (groups R2, R6 and R4b), with the R0 gate
- `tier1_status.py`: one line per run, joining Slurm on both clusters with each run's
  checkpoint and log (state, GPU, progress and rate, last convergence check, A100-h used and
  estimated, finish time, preemptions). It logs every queued run's estimated start.
- `cost_estimate.py`: the A100-h estimates (appendix B), and how far Slurm's estimated starts
  have been from the actual starts, by how far ahead each estimate was made
- `predict_identifiability.py`: Figs 6/6b's answer predicted at the truth from the Fisher
  information (contraction, correlation, eigen-directions), to set beside the sampled one
- `sbc_width_check.py`: each SBC posterior's sd against the Laplace (Fisher) sd at its truth,
  with the z-scores and Mahalanobis distances, to tell a mis-sized posterior from chance
- `postprocess.py`: pull, score and redraw every finished run in one command;
  `tier1_result_figures.py` for the Fig 2 and 4 drafts and the R7 and R8 checks
- Run figures, drawn by `tier1.sbatch` once a run finalizes (all in `Utilities/`):
  `plot_convergence_trajectory.py` (convergence, chain mixing, sampler energy, LOO),
  `plot_trace_diagnostics.py` (prior vs. posterior and traces, and the joint posterior), `plot_predictive_check.py`
  (predictive checks, with a Tier-1 layout), and `plot_run_summary.py`, which combines them
  into `run_summary.png`. Each is titled "<run> — <section>", and its file is named after
  the section.

Nothing in the plan's to-build list remains. R2 and R4 have finished; Figs 7 and 8 are drawn from
R2's posterior (`Results/Tier1/figures/enzyme_sensitivity.png`, `ratio_strategy.png`), and Fig 3 is drafted on all 40 replicates (`sbc_ranks.png`,
with three alternative forms: `sbc_ecdf.png`, `sbc_coverage.png` and `sbc_recovery.png`). Figs
6/6b have R2's panel (`identifiability_C14+unsat_a1c3a2.png`) and wait on R3's C14+unsat and C18
runs. Every figure outside a run folder's run_summary set is in `Results/Tier1/figures/` (the
Morris screen's in `Results/Sensitivity Screen/figures/`); job_files holds code and data only.

---

## 6. After the drafts: which figures get recreated on the full model

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
| Fig 9 | only the cells the real data allow; if R7 shows the information ranking predicts sampled contraction, Tier 2 uses the information analysis only | |
| SI tolerance / `target_accept` | one check on the Tier-2 fit, if budget allows | |

The Tier-2 core is therefore **one three-parameter fit** of the full model to the real data,
plus grouping fits only if Fig 5 earns them. Its parameters come from the Tier-1 Fig 6
results and the sensitivity screen. The FabH-knockout kinetics conditions in ME1 Dataset S1
are old data and are not used (outline 2.8).

---

## Appendix A: evidence behind the sampler settings

- **4 chains.** In the chain-count test (C12, `a1`+`c3`, 4/8/16/32/64 chains, equal
  sampling budget), 4 chains converged in 4.00 A100-h against 7.10 at 8. Splitting the
  64-chain run into disjoint groups gives the SI's diagnostic-power result.
- **300 warmup steps at two parameters.** Acceptance over the last 50 warmup draws is
  0.777-0.780 against the 0.8 target. The first sampling block's mean is within 0.12 sd of
  the rest, r-hat moves ≤ 0.004 when that block is dropped, and there are zero divergences.
  Step sizes scatter 8-10x across chains; with per-chain adaptation that means each chain
  reaches the same acceptance by a different route. At three parameters, R2's acceptance over
  the last 50 warmup steps was 0.777, and it converged on the same 300 steps (section 3).
- **A minimum of 3 chains.** One stranded chain out of four must not block a run. Two or more
  triggers a rerun.

## Appendix B: cost model

`tier1/cost_estimate.py` rebuilds every estimate from the measured runs; the monitor reruns it
every 30 minutes (section 2). Every condition is solved to 720 s, the latest save time.

**A run's cost**, in A100-equivalent hours (GPU wall time x its speed factor, H100 NVL 1.37,
H200 1.23 and A100 1.00, as the sampler counts its cap). The H200 factor was measured on
2026-09-27 (C8 `a1`+`c3` sampling: 1.62 s per leapfrog step against 1.99 on an A100-40GB, over
740 and 1460 chunks); before then H200 time counted at 1.00, so segments run on H200 before
that date are undercounted by about 20%:

startup (0.2 h per 11.5 h segment) + 300 warmup steps x s/step + draws x s/draw + finalize and
figures

- **Rates**, most specific first:
  - the run's own progress log;
  - the median of measured runs with the same system and free parameters, pooling SBC
    replicates and fixed-truth runs separately and leaving out solver variants and data subsets;
  - the same system's `a1`+`c3` rates, times the three-parameter pilot factors for a
    three-parameter run.
- **s/draw before a run samples:** its warmup s/step x the sampling-to-warmup ratio of runs that
  have done both (0.65 on R0).
- **Draws to finish:** the median over finalized runs (800 on R0), x 2.24 for three parameters.
- **Finalize:** measured per draw on the finished runs of each system (R0: 3.4 s per draw, 46 min
  for 800), scaled to other systems by warmup cost. The recovery report and figures after it
  are timed to the job's end in sacct.

**Measured so far** (Sun 2026-09-27 21:53, `cost_estimate.json`; R3's running rates from
`tier1_status.py` at 22:00):

| | s per warmup step | s per draw | Whole run, A100-h |
|---|---|---|---|
| C8 `a1`+`c3`, fixed truth (R0, R5, R8; 15 finished) | 19.5-36.3 | 13.1-25.5 | 3.5-7.6 |
| C8 `a1`+`c3`, SBC (40 finished) | 21.3-78.6 | 11.4-30.7 | 3.8-16.3 |
| C14+unsat `a1`+`c3` (R1, R7, R6 (c) grouped; 4 finished) | 38-46 | 25-28 | 6.5-9.8 |
| C14+unsat `a1`+`c3`+`a2` (R2) | 66.6 | 51.3 | 16.7 |
| C14+unsat `a1`+`c3s`+`c3l` (R6 split; 2 finished) | 50.0-51.6 | 26.2-29.5 | 6.5-8.3 |
| C14+unsat `d1`+`d2` (R3, running) | diagonal ~290; dense 314 before its first update, ~38 after | diagonal ~174 (~3 min) | ~63 diagonal, ~38 dense (est.) |
| C18 `d1`+`d2` (R3, in warmup) | diagonal ~225; dense ~263 before its first update | | ~60 diagonal, ~52 dense (est.) |
| C8 `d1`+`d2` (R3) | 120-4900 on the ridge at `max_steps` 20000; ~24 after the dense run's first update | 15.5 (dense) | 19.3 dense; diagonal capped at 24.6 |

The 60 finished two-parameter runs took 400-1700 draws, median 600; the three three-parameter
runs 400-700. Sampling costs 0.57x a warmup step per draw. Recovery report and figures after
finalize: 0.02 h, from the job end in sacct.

**Three parameters** (measured 2026-09-27). The pilot medians put `a1`+`c3`+`a2` at ~3.0x per
warmup step and ~3.5x per draw relative to `a1`+`c3`, with 2.24x the draws. On C14+unsat at R1's
rates that was ~119 s per step, ~88 s per draw and ~1100 draws: ~40 A100-h. R2 measured 66.6 s
per step (1.68x R1), 51.1 s per draw (2.04x) and, after its reopening, 1000 draws (2.5x): 21.5 A100-h. R6's split
trio ran at 1.26-1.30x per step, 1.04-1.17x per draw and 1.0x the draws of R1: 21.4 A100-h for R6's three fits against ~87 estimated. So the estimates from
the pilot factors were 1.9x too high for R2 and 4x for R6.

**Systems not yet measured** (C20+unsat) are scaled from the nearest measured one by cost per
draw ∝ reactions^1.21, fitted on the C6/C10/C14 pilots (C14+unsat to C20+unsat: 1.85x).

Measured cost per leapfrog step (A100-seconds, median over checkpoint chunks, 2026-09-27). The
reaction-count rule leaves out tolerance: C18 and C20+unsat run at `rtol` 1e-5, C8 and
C14+unsat at 1e-3.

| System | `rtol` | Tier-1 design | Chain ladder (`a1`, time series + profile) |
|---|---|---|---|
| C8 | 1e-3 | 1.96 (`a1`+`c3`), 2.20 (`d1`+`d2` dense) | |
| C14+unsat | 1e-3 | 3.97 (`a1`+`c3`), 4.10 (`a1`+`c3`+`a2`), 4.70 (`d1`+`d2`) | 4.40 |
| C18 | 1e-5 | 9.85 (`d1`+`d2`) | 7.92 |
| C18+unsat | 1e-5 | | 11.90 |
| C20+unsat | 1e-5 | | 13.23 |

So C20+unsat costs ~3.0x C14+unsat per leapfrog (ladder), or ~14 s from C18's Tier-1 cost and
the ladder's C18 to C20+unsat ratio (1.67x). An `a1`+`c3`+`a2` fit on C20+unsat with R2's
leapfrog budget (~4600 in warmup, ~12 per draw, ~650 draws: ~12k) would need ~12-14 s x 12k
= ~40-50 A100-h, or ~30-35 h of H100 time: three segments plus queue gaps, 2-3 days. Unknown:
whether its posterior needs more leapfrogs per draw than R2's, and whether a looser `rtol`
passes the pre-flight accuracy check (the main lever on cost). Its solver steps across the
prior would need the same scan as the others first.

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

**How Fig 9's predictions are calculated** (`expected_information_grid.py`). This is the
Bayesian D-optimal design calculation for a normal linear model (Chaloner & Verdinelli 1995,
§2.1-2.2), applied to the model linearized at the truth, the normal approximation their §4.2
uses for nonlinear models. No sampling is needed; a grid takes 6-8 min on a CPU.
1. **Sensitivities.** Solve the model at the true parameters and again with each parameter
   scaled by e^±0.001. Central differences give every data point's sensitivity to
   log(parameter), J = ∂y/∂log θ; a step of 3x that agrees to 10⁻⁴ relative.
2. **Noise.** Each point gets the data's own noise model, σ = 10% of its value plus the floor.
3. **Fisher information:** F = Jᵀ W J, with W = diag(1/σ²).
4. **Prior and posterior.** On log θ the prior is Normal with sd ln(10)/1.96 = 1.175 (the
   LogNormal with 95% in [0.1, 10]), precision I/1.175². The linear-Gaussian posterior
   covariance is Σ = (F + I/1.175²)⁻¹, Chaloner & Verdinelli's σ²(nM + R)⁻¹.
5. **Scores.** Predicted contraction per parameter, 1 − Σⱼⱼ / 1.175², comparable with
   `recovery_report.py`'s log-scale contraction. Expected information gain, ½ log(det prior
   covariance / det Σ) nats, their eq. 4 up to a constant; divided by the number of points it
   gives Fig 9's value per measurement.
6. **Cells.** Repeated for every measurement set × timing, and for the Tier-1 design's own
   parts (time series, profile, rates, all three), which are R7's predictions.

The approximation is local and linear: exact for a posterior that is Gaussian in log θ, and it
misses skew or curvature away from the truth. That is where R7's rates-only `c3` departs from
it (section 4).

---

## References: what Tier-1 decisions rest on

Only sources that a Tier-1 setting or gate actually rests on; the paper's wider reading list is
in `paper_outline.md`. Entries marked [verify] need their details checked before citing.

**External sources, and the decision each informed**
- Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021). Rank-normalization, folding,
  and localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*
  16(2):667-718. → The stopping rule: rank-normalised split r-hat ≤ 1.01 and bulk ESS ≥ 50 per
  split chain (400 at 4 chains), held on three consecutive checks 100 draws apart (two, then a
  confirmation block; from 2026-09-27, section 0); 4 chains (Appendix A).
- Hoffman MD, Gelman A (2014). The No-U-Turn Sampler: adaptively setting path lengths in
  Hamiltonian Monte Carlo. *JMLR* 15:1593-1623. → NUTS with dual-averaging step-size
  adaptation; `target_accept` 0.8, and 0.95 as the R8 check.
- Schad DJ, Betancourt M, Vasishth S (2021). Toward a principled Bayesian workflow in cognitive
  science. *Psychological Methods* 26(1):103-126 (arXiv:1904.12765). Eqs. 4-5, the posterior
  z-score and posterior contraction (1 − posterior variance / prior variance): the scores in
  `recovery_report.py`, on log(x) for the LogNormal groups (adopted 2026-09-27).
- Talts S, Betancourt M, Simpson D, Vehtari A, Gelman A (2018). Validating Bayesian inference
  algorithms with simulation-based calibration. arXiv:1804.06788. → R4's design: truths drawn
  from the fit's own prior, ranks of the truth among L = 99 thinned draws, uniformity test.
- Brun R, Reichert P, Künsch HR (2001). Practical identifiability analysis of large
  environmental simulation models. *Water Resources Research* 37(4):1015-1030. → The
  collinearity index and its gamma > 10-15 cutoff: `d1`/`d2` never paired in a recovery fit,
  `a1`+`c3` chosen as the best-separated pair (Appendix C).
- Morris MD (1991). Factorial sampling plans for preliminary computational experiments.
  *Technometrics* 33(2):161-174; and Campolongo F, Cariboni J, Saltelli A (2007). An effective
  screening design for sensitivity analysis of large models. *Environmental Modelling &
  Software* 22(10):1509-1518. → The sensitivity screen (mu*, radial design) behind the
  parameter choice (`select_testcase.py`'s influence gate), the C8-over-C6 argument (section
  0), and Fig 7's method.
- Vehtari A, Gelman A, Gabry J (2017). Practical Bayesian model evaluation using leave-one-out
  cross-validation and WAIC. *Statistics and Computing* 27:1413-1432; and Vehtari A, Simpson D,
  Gelman A, Yao Y, Gabry J (2024). Pareto smoothed importance sampling. *JMLR* 25(72):1-58. →
  PSIS-LOO for Fig 5 (R6), and the gate that every Pareto k ≤ 0.7 before a Δelpd is trusted.
- Betancourt M (2016). Diagnosing suboptimal cotangent disintegrations in Hamiltonian Monte
  Carlo. arXiv:1604.00695. → The BFMI caution floor of 0.3 in the energy diagnostic.
- Chaloner K, Verdinelli I (1995). Bayesian experimental design: a review. *Statistical
  Science* 10(3):273-304 (doi:10.1214/ss/1177009939); and Lindley DV (1956). On a measure of
  the information provided by an experiment. *Annals of Mathematical Statistics*
  27(4):986-1005. → Fig 9's expected-information grid, which picked the data design and R7's
  cells: the normal linear model's posterior covariance σ²(nM + R)⁻¹ and expected Shannon
  information (Chaloner & Verdinelli §2.1-2.2, Bayes D-optimality), applied through the
  normal approximation with the Fisher information that their §4.2 uses for nonlinear models
  (appendix D spells out the steps).

**Internal evidence** (recorded in this plan or the named file)
- Chain count and warmup length: the chain-count study on C12 (Appendix A;
  `tier1/convergence_and_warmup.py`, `chaintest_convergence.json`), confirmed on R0
  (section 4).
- Replicate system and cost: the Morris screen (`chain_system_sensitivity_analysis/`), the
  reaction files' chain-length-specific constants, and the per-gradient cost profiles
  (`multiparam_tests/cost_profile.py`) (section 0); R0's measured cost (section 4, Appendix B).
- Data design: `scan_rate_multipliers.py` and `check_rate_design.py` (Appendix D).
- Parameter sets: `collinearity.py` and `select_testcase.py` (Appendix C), and the
  two-parameter pilots on C6, C10 and C14 (`multiparam_tests/mp_posterior.json`).
- Sampler plumbing: stranded-chain detection and the multimodality toy (Stage 0 item 6;
  `Notes/efficiency_logic_map.md`).
- The R5 start point: the log posterior at the shifted prior's mean against near the truth
  (section 2, R4/R5 notes).

