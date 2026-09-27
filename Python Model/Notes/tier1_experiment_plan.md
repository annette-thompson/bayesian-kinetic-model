# Tier 1 run plan: the fits that draft every figure in the paper

Current as of 2026-09-24. This is the companion to `Notes/paper_outline.md`. The outline says
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
  parameters with shrinkage > 0.5, C12 takes its place, at ~1.6x the cost. R0 recovered both
  at shrinkage 0.98 (section 4), so C8 stays.

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

Costs are A100-equivalent hours for the whole run (compute, finalize and figures), estimated by
`tier1/cost_estimate.py` from the rates measured on the runs so far (appendix B). The sampler
solves every condition to the latest save time (720 s), including the rate conditions observed
at 150 s, and every estimate assumes that. A run that has started is estimated from its own
progress log. One that has not uses the median of measured runs with the same system and free
parameters. The three-parameter fits, not yet measured, are scaled from R1 by the pilot factors.
The monitor refreshes the estimates every 30 minutes (`Results/Tier1/figures/cost_estimate.json`,
with the history in `job_files/tier1/cost_estimate_history.jsonl`), and this table follows them.
Last updated Sat 2026-09-26 20:45.

Two-parameter runs cap compute at 24 A100-h (`max_total_hours`), a runaway guard about twice
their estimates. The three-parameter fits (R2, R6 b and c) run with no cap (decided 2026-09-26).

Run folders are `Results/Tier1/<run name>/`, written by `build_tier1_configs.py --plan`.

| ID | Run | System | Free params | Fits | A100-h | Feeds |
|---|---|---|---|---|---|---|
| R0 | smoke test, production stopping rule | C8 | `a1`+`c3` | 1 | 5.7 (measured) | plumbing; chooses the replicate system; centre point of Fig 4 |
| R1 | main-system duo | C14+unsat | `a1`+`c3` | 1 | 7.2 (measured) | Fig 2 fallback; Fig 5 grouped model; Fig 9 full-data cell |
| R2 | **main fit** | C14+unsat | `a1`+`c3`+`a2` | 1 | ~40 | Figs 2, 6, 6b, 7, 8 |
| R3 | `d1`+`d2` known-answer control | C8 diagonal, C8 dense, C14+unsat diagonal | `d1`+`d2` | 3 | ~74 (all three at the 24 cap) | 3.3 case; Fig 6/6b; SI mass matrix |
| R4 | SBC replicates, truth drawn from the prior | C8 | `a1`+`c3` | 10 pilot, then 30 more | ~67 + ~160 | Fig 3 |
| R5 | robustness: noise 5/20/40% (10% = R0), prior median shifted +1/+2/+3/+4 prior sd, each shift run from the default start and from the ME1 values | C8 | `a1`+`c3` | 11 | ~48 | Fig 4 |
| R6 | `c3` split test | C14+unsat | see below | 3 | ~87 | Fig 5 |
| R7 | data-type check, 3 cells (R1 is the third) | C14+unsat | `a1`+`c3` | 2 new | ~16 | Fig 9 |
| R8 | `target_accept` 0.95; `rtol` 1e-5 | C8 | `a1`+`c3` | 2 | ~12 | SI |
| | **Total** | | | **~64 new** | **~520** | |

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
begins. Ridge posteriors can hit the tree-depth ceiling. R3 keeps the standard 24 A100-h cap,
about 4x the C8 estimate and 2x the C14+unsat one; hitting it under diagonal but not dense is
itself the SI result. *Optional:* C20+unsat (~20 A100-h: R1's rate scaled by reaction count)
shows clear separation. Add it only if the figure needs a third point.

**R6 (grouping, Fig 5).** `c3` ties six TesA hydrolysis constants with six distinct nominal
values. It is split into short-chain (`c3s`: C4-C8) and long-chain (`c3l`: C10-C14 plus the
unsaturated species) halves in a script-generated variant of the C14+unsat reaction set
(`Reactions/EC_FAS_ME1/C14+unsat+c3split`); the reaction files are never hand-edited. The
runs:
- (a) grouped `a1`+`c3` on the standard data. This is R1, reused.
- (b) split `a1`+`c3s`+`c3l` on the same data. New; ~40 A100-h, as R2.
- (c) data generated at `c3s` = 1, `c3l` = 3, fit with both models. New; ~8 + ~40.

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

Built 2026-09-25 as `Tier1 C14+unsat - a1c3 - profile` and `Tier1 C14+unsat - a1c3 - rates`,
each a subset of the standard data. The rates dataset is the best-per-point cell. The five
initial rates carry exactly the information of total fatty acid at 150 s (4.27 nats either
way). For a like-for-like prediction the grid was re-run on the `a1`+`c3` model
(`expected_information_grid_a1c3.json`). Predicted log-scale shrinkage for `a1` / `c3`:

| Cell | Points | Information (nats) | `a1` | `c3` |
|---|---|---|---|---|
| full data (R1) | 23 | 6.34 | 0.969 | 0.942 |
| profile only | 8 | 5.77 | 0.950 | 0.938 |
| rates only | 5 | 4.27 | 0.915 | 0.828 |

So the sampled order to check is full > profile > rates, with `c3` separating the cells most.
Per point, the ranking reverses (rates 0.85 nats, profile 0.72, full 0.28).

**R4/R5 on C8.** Calibration and robustness test whether stated uncertainty is honest, which
doesn't depend on network size. R4 starts with a 10-replicate pilot and continues to 40 only
if the pilot's coverage and rank histogram look sane. SBC ranks and coverage come from the
same replicates. The R5 noise-level datasets reuse the 10% dataset's random draws, rescaled,
so the noise axis carries no draw-to-draw scatter. Compare the prior-offset runs on log-scale
shrinkage (`shrinkage_log` in `recovery_report.py`), which doesn't change when the prior
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
the posterior bulk within 7 steps). It is not yet established for the SBC replicates with
extreme truths (e.g. `sbc004`/`sbc005`, `a1` ≈ 8.7), whose chains start up to ~1.8 log units
away. SBC needs identical settings for every replicate, so if the pilot shows these failing,
the remedy (more warmup or another start) applies to all 40.

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
   (1.09 nats per point). Individual species buy most of the attainable shrinkage (0.95-0.97
   mean, log scale). The ACP intermediates (ketoacyl-, hydroxyacyl-, enoyl- and acyl-ACP) add
   little beyond that at their concentrations under the 0.01 µM floor (0.03-0.06 nats per
   point). The Tier-1 design scores 0.39 nats per point over 23 points (shrinkage 0.96 / 0.90
   / 0.93 for `a1` / `c3` / `a2`). So R7's "best cell per data point" is total fatty acid at
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

**Stage 2: everything cheap and independent, one batch (~225 A100-h).** R1, R3, R5, R7, R8
and the R4 pilot, 29 runs, submitted 2026-09-26 by `tier1/submit_stage2.sh` from `job_files/`
(R7 and R5's ME1-start series were added the same day). It refuses until R0 has finalized, and
it skips runs that are finished or already queued, so rerunning it resumes the ones that
stopped at the wall clock. `tier1/tier1_status.py` shows each run's state, its A100-h used and
estimated, and when it should finish.
- Gate to R2: R1 converges and recovers both parameters.
- Gate to the full R4: the pilot looks sane.

**Stage 3: the main fit and its dependents (~290 A100-h).**
- The three-parameter fits run with no compute cap (decided 2026-09-26). At R1's measured
  rates and the pilot factors, R2 needs ~40 A100-h: ~10 h of warmup (~119 s per step) and
  ~27 h for ~1100 draws (~88 s per draw), about four 12 h segments. R1 passed its gate on
  2026-09-26 (section 4).
- R2: inspect warmup with `convergence_and_warmup.py` at the first segment boundary. If
  acceptance is well below 0.8, or the first block still drifts, restart with 600 warmup
  steps and record it.
- R6 (b), then (c) once (b)'s cost is known.
- The remaining R4 replicates (built and pre-flighted), ~160 A100-h at the pilot's median.

**Stage 4: post-processing (forward solves).** Figs 6/6b from R2 and R3. Figs 7 and 8 from
R2's posterior (`posterior_morris.py`, `posterior_ratio_response.py`). Fig 9's grid is done
at the truth (item 8), and R7 checks it against sampled shrinkage.

**Stage 5: draft every figure, then choose what goes to Tier 2** (section 6).

---

## 4. Results

### R0: C8, `a1`+`c3` (Blanca job 28506399, 2026-09-25)

One segment on an A100 (`bgpu-biokem2`), 5 h 46 min of wall time. The figures are in
`Results/Tier1/Tier1 C8 - a1c3/`, all of them together in `run_summary.png`.

**Recovery.** Both parameters recovered, truth inside the 95% interval for both:

| | Truth | Posterior mean ± sd | z | Shrinkage (natural / log) |
|---|---|---|---|---|
| `a1` | 1.00 | 1.049 ± 0.055 | +0.90 | 0.984 / 0.956 |
| `c3` | 1.00 | 0.981 ± 0.062 | −0.32 | 0.982 / 0.946 |

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

### Stage 2, first results (2026-09-26, 20:30)

Fourteen runs have finished (R0 and 13 of Stage 2's 29), and every one has the truth inside the
95% interval for every parameter. R1 passed the gate to R2.

| Run | `a1` z | `c3` z | Log-scale shrinkage `a1` / `c3` | Draws | A100-h |
|---|---|---|---|---|---|
| R1: C14+unsat | −0.04 | −0.97 | 0.968 / 0.942 | 400 | 7.2 |
| R7: C14+unsat, profile only | +1.06 | −1.69 | 0.946 / 0.931 | 400 | 6.4 |
| R5: C8, noise 5% | +0.88 | −0.19 | 0.976 / 0.968 | 600 | 4.0 |
| R0: C8, noise 10% | +0.90 | −0.32 | 0.956 / 0.947 | 800 | 5.7 |
| R5: C8, noise 20% | +0.89 | −0.38 | 0.911 / 0.903 | 500 | 3.5 |
| R5: C8, noise 40% | +0.87 | −0.43 | 0.823 / 0.805 | 500 | 4.4 |
| R5: C8, prior +1 sd, default start | +0.94 | −0.23 | 0.954 / 0.946 | 500 | 3.6 |
| R5: C8, prior +2 sd, default start | +1.06 | −0.14 | 0.956 / 0.947 | 800 | 4.7 |
| R8: C8, target acceptance 0.95 | +0.84 | −0.38 | 0.955 / 0.947 | 500 | 5.0 |
| R4: SBC 000, 001, 002, 004, 005 | −0.16 to +0.84 | −1.04 to +0.62 | 0.963-0.988 / 0.788-0.920 | 400-800 | 4.0-7.0 |

- **R1 (gate to R2).** Converged at 300 draws (r-hat 1.0049, ESS 758), finalized at 400. No
  divergences, BFMI 0.97-1.12, elpd_loo −22.15 ± 7.13. Correlation +0.03.
- **Fig 9 check.** Sampled log-scale shrinkage matches the expected-information prediction to
  within 0.01: full data 0.968 / 0.942 against 0.969 / 0.942, profile only 0.946 / 0.931
  against 0.950 / 0.938. The rates-only cell is still sampling.
- **Noise.** The posterior widens with the noise and the truth stays covered: `a1`'s sd is 0.029,
  0.055, 0.114 and 0.252 at 5, 10, 20 and 40%.
- **Prior shift, default start.** At +1 and +2 prior sd the chains, started at the shifted
  prior's mean, found the truth, and the posterior is almost R0's (log-scale shrinkage
  unchanged). +3 and +4 sd, and the ME1-start series, are still running.
- **Settings.** Target acceptance 0.95 reproduces R0 (z within 0.06); `rtol` 1e-5 is still running.
- **Warmup.** `convergence_and_warmup.py --summary` passes R1, sbc000, sbc004 and sbc005:
  acceptance 0.777-0.780 over the last 50 warmup steps, first-block drift ≤ 0.15 sd, and
  dropping the first block moves r-hat by ≤ 0.006. The pilot's chains reached their true values
  within 40-55 warmup steps, the extreme truths (`a1` ≈ 8, `c3` ≈ 10) included.
- **SBC.** Five of the 10 pilot replicates are in; the rank test needs all 10. `c3` at 9.9
  (sbc002) is the least identified (log-scale shrinkage 0.79; natural-scale 0.15, since the
  natural scale stretches the upper tail).
- **R3 is slow on the ridge.** On C8, warmup steps build trees of up to 183 leapfrog steps at
  ~3.7 s each through step 10, and steps 10-15 took 6.9 h (about 80 min each, at or near the
  1023-step ceiling), with the GPUs busy throughout. The dense run's first segment stopped at
  7.4 h, at step 15, because the next 5-step chunk would not fit its 11.5 h budget; it was
  resubmitted. Both runs are identical until the dense run's first mass-matrix update at
  warmup step 100 (then 150 and 250, BlackJAX's schedule for 300 steps), about 110 h away,
  and both should reach the 24 A100-h cap near step 25, with no posterior. C14+unsat (~320 s
  per step) should reach the cap during sampling. Open: run both to the cap (the default),
  stop the diagonal run and lift the dense cap, or rethink the C8 ridge test.

## 5. Code status

**Exists** (all in `job_files/tier1/` unless noted):
- `build_tier1_configs.py`: `--plan` writes every config in section 2 except R4's (`sbc.py`
  builds those); `--datasets` fits a subset of the standard data (R7)
- `make_tier1_rate_data.py`: the data above, including off-grouping and noise-level variants
- `make_c3_split_variant.py`: R6's reaction variant, with an exact-equivalence check
- `check_model_vs_data.py`: pre-flight (model vs. clean data, noise z, logp/gradient)
- `multimodality_toy.py`: 3.1's known-answer toy
- `recovery_report.py`: per run and parameter, z, shrinkage (natural and log scale) and
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
  `posterior_ratio_response.py` (Fig 8), `identifiability_report.py` (Figs 6/6b: shrinkage,
  correlations, prior-standardised covariance eigen-directions), and `plot_tier1_drafts.py`
  for their drafts
- `sbc.py` (Fig 3): `generate` draws truths from the prior and builds each replicate's data and
  config. `ranks` scores finished runs (rank histograms, a uniformity test, coverage and a draft
  figure). `selftest` checks the rank code on a known answer.

- `submit_stage2.sh`: Stage 2's 29 submissions, with the R0 gate
- `tier1_status.py`: one line per run, joining Slurm on both clusters with each run's
  checkpoint and log (state, GPU, progress and rate, last convergence check, A100-h used and
  estimated, finish time, preemptions). It logs every queued run's estimated start.
- `cost_estimate.py`: the A100-h estimates (appendix B), and how far Slurm's estimated starts
  have been from the actual starts, by how far ahead each estimate was made
- `postprocess.py`: pull, score and redraw every finished run in one command;
  `tier1_result_figures.py` for the Fig 2 and 4 drafts and the R7 and R8 checks
- Run figures, drawn by `tier1.sbatch` once a run finalizes (all in `Utilities/`):
  `plot_convergence_trajectory.py` (convergence, chain mixing, sampler energy, LOO),
  `plot_trace_diagnostics.py` (prior vs. posterior and traces), `plot_predictive_check.py`
  (predictive checks, with a Tier-1 layout), and `plot_run_summary.py`, which combines them
  into `run_summary.png`. Each is titled "<run> — <section>", and its file is named after
  the section.

Nothing in the plan's to-build list remains. Figs 6/6b wait on R2 and R3, Figs 7 and 8 on R2's
posterior, and Fig 3 on the R4 fits.

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
| Fig 9 | only the cells the real data allow; if R7 shows the information ranking predicts sampled shrinkage, Tier 2 uses the information analysis only | |
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
  reaches the same acceptance by a different route. Not yet validated at three parameters,
  hence the R2 warmup check.
- **A minimum of 3 chains.** One stranded chain out of four must not block a run. Two or more
  triggers a rerun.

## Appendix B: cost model

`tier1/cost_estimate.py` rebuilds every estimate from the measured runs; the monitor reruns it
every 30 minutes (section 2). Every condition is solved to 720 s, the latest save time.

**A run's cost**, in A100-equivalent hours (GPU wall time x its speed factor, H100 NVL 1.37 and
A100 1.00, as the sampler counts its cap):

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
- **Finalize:** 3.4 s per draw on R0 (46 min for 800), scaled to other systems by warmup cost.
  The figures after it are a 0.1 h placeholder until a Stage-2 run measures them.

**Measured so far** (Sat 2026-09-26 20:45):

| | s per warmup step | s per draw | Whole run, A100-h |
|---|---|---|---|
| C8 `a1`+`c3`, fixed truth (R0, R5 noise and +1/+2 sd, R8 ta0.95; 7 finished) | 17-24 | 10.9-16.9 | 3.5-5.7 |
| C8 `a1`+`c3`, SBC pilot (5 finished) | 22-42 | 11.4-13.9 | 4.0-7.0 |
| C14+unsat `a1`+`c3` (R1, R7 profile) | 39-40 | 25-26 | 6.4-7.2 |
| C14+unsat `d1`+`d2` (R3, in warmup) | ~320 | | 24 (cap) |
| C8 `d1`+`d2` (R3, in warmup) | ~1600, on the ridge | | 24 (cap) |

The finished two-parameter runs took 400-800 draws, median 500. Sampling costs 0.61x a warmup
step per draw (25 runs).

**Three parameters** (not yet measured): `a1`+`c3`+`a2` costs ~3.0x per warmup step and ~3.5x per
draw relative to `a1`+`c3`, and needs 2.24x the draws (pilot medians). On C14+unsat at R1's rates
that is ~119 s per step, ~88 s per draw and ~1100 draws: ~40 A100-h.

**Systems not yet measured** (C20+unsat) are scaled from the nearest measured one by cost per
draw ∝ reactions^1.21, fitted on the C6/C10/C14 pilots (C14+unsat to C20+unsat: 1.85x).

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

---

## References: what Tier-1 decisions rest on

Only sources that a Tier-1 setting or gate actually rests on; the paper's wider reading list is
in `paper_outline.md`. Entries marked [verify] need their details checked before citing.

**External sources, and the decision each informed**
- Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner P-C (2021). Rank-normalization, folding,
  and localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*
  16(2):667-718. → The stopping rule: rank-normalised split r-hat ≤ 1.01 and bulk ESS ≥ 50 per
  split chain (400 at 4 chains), held for 2 consecutive checks; 4 chains (Appendix A).
- Hoffman MD, Gelman A (2014). The No-U-Turn Sampler: adaptively setting path lengths in
  Hamiltonian Monte Carlo. *JMLR* 15:1593-1623. → NUTS with dual-averaging step-size
  adaptation; `target_accept` 0.8, and 0.95 as the R8 check.
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
  Science* 10(3):273-304; and Lindley DV (1956). On a measure of the information provided by
  an experiment. *Annals of Mathematical Statistics* 27(4):986-1005. → Fig 9's
  expected-information grid (Gaussian/Laplace approximation to the information gain), which
  picked R7's cells.

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

