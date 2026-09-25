# Experiment map for the whole paper

Written 2026-09-16, against `Outlines/Bayesian Framework/Outline_v2.docx` (the live outline;
the `.md` is stale). Covers every experiment sections 3.1-3.6, the conclusion and the SI
depend on, ordered cheapest-first within dependency constraints. Tier-1 run mechanics live
in `tier1_experiment_plan.md` section 4; this document is the layer above it.

Two tiers, from outline 2.8. **Tier 1** is synthetic data with known ground truth on the
small ladder (C8, C12, C14, C14+unsat) -- the only way to check the machinery gives correct
answers is to know the answer in advance. **Tier 2** applies the identical, unmodified recipe
to the full ME1 model with the real datasets the previous point-estimate model was fit to.

## 1. The shape of the problem

Six results sections, but **not six independent experiments**. One Tier-1 multi-parameter
model is the hub: 3.1 fits it, 3.3 reads identifiability straight off that same posterior at
no extra cost, and 3.4/3.5/3.6 are post-processing or re-fits of it. Only 3.2 needs a
genuinely different model.

```
        Tier-1 core fits (3.1 Fig 2)  <-- the hub; waves 0-2 of tier1_experiment_plan.md
                 |
   +-------------+---------------+---------------+------------------+
   |             |               |               |                  |
  3.3           3.4             3.5             3.6            3.1 Fig 3/4
identifiab.  posterior-int.  ratiometric    data-type grid    SBC + robustness
 (free from   sensitivity     robustness    + rate-limiting    (20-50 refits)
  the hub)   (forward solves) (fwd solves)      conditions
                                                                   |
   3.2 grouping  -- separate model, needs constants freed          |
                    independently (largest infra gap)              |
                                                                   |
                          Tier 2: full ME1 + real data <-----------+
```

That structure decides the order: **do the hub first, harvest everything free from it, then
spend on the things that need their own fits.**

## 2. Cost model

Every number anchors on the single directly comparable measurement -- Tier-1 data, C12,
`a1`+`c3`, 4 chains, `tune=300`: **4.00 A100-h** (warmup 2.05, sampling 1.95, converged draw
400) -- scaled by `reactions^1.21` (fitted) and by measured parameter-count multipliers. Per
system a duo costs 2.5 / 4.0 / 4.8 / 6.4 h (C8 / C12 / C14 / C14+unsat); a trio costs
13 / 21 / 26 / 34 h.

| # | experiment | A100-h | cumulative | confidence |
|---|---|---|---|---|
| 1 | 3.1 wave 0: smoke test, `a1`+`c3` on C8 | 2 | 2 | measured |
| 2 | 3.1 wave 1: 3 duos x 4 systems | 53 | 56 | measured |
| 3 | 3.1 wave 2: trio `a1`+`c3`+`a2` x 4 systems | 95 | 150 | measured multipliers |
| 4 | 3.3 identifiability: shrinkage + correlation + eigen | **0** | 150 | reuses #1-3 |
| 5 | 3.3 `d1`+`d2` negative control (C8, C14) | 7 | 158 | measured |
| 6 | 3.6 rate-limiting conditions (2 sets, C12) | 8 | 166 | measured |
| 7 | 3.6 data-type grid (9 cells, C12) | 36 | 202 | measured + infra |
| 8 | 3.1 Fig 3 SBC/coverage pilot, N=10 on C8 | 25 | 226 | measured |
| 9 | 3.1 Fig 3 SBC/coverage full, N=40 on C8 | 99 | 325 | measured |
| 10 | 3.1 Fig 4 robustness: 5 prior offsets + 4 noise levels | 22 | 348 | measured |
| 11 | 3.2 grouping: 3 variants x 2 systems | ~600 | ~950 | **extrapolated** |
| - | 3.4 posterior-integrated sensitivity | forward solves only | | needs infra |
| - | 3.5 ratiometric across draws | forward solves only | | needs infra |
| 12 | Tier 2: full ME1, real data, 5 conditions (trio) | 46 | ~1000 | measured multipliers |

Two entries dominate and both deserve scrutiny before they are committed:

- **#11 (3.2 grouping) is ~600 A100-h on an extrapolated cost** and no 4-parameter run has
  ever been done. Freeing a group's constituent constants means one free parameter *per
  constant*; a group with four constants is a 4-parameter fit, and the quad multiplier is a
  guess. This is the single biggest budget risk in the paper.
- **#9 (SBC/coverage) is ~100 A100-h of embarrassingly parallel small runs.** Cheap per run,
  large in aggregate, and the outline already tables it pending a small-N confirmation.

3.4 and 3.5 cost forward ODE solves over posterior draws, not sampling -- a different and
much cheaper budget, but both need code that does not exist.

## 3. Order of execution

### Stage A -- the hub (~150 A100-h, 1-2 weeks)
Waves 0-2 of `tier1_experiment_plan.md` section 4.3. Produces **Figure 2** and every
posterior the rest of Tier 1 reads.

### Stage B -- free harvest (~15 A100-h, days)
Runs almost entirely on Stage A's output.

- **3.3 / Figures 6 and 6b** -- prior-to-posterior shrinkage per parameter (flag < 50% as
  weakly identified), the pairwise posterior correlation matrix, and an eigendecomposition of
  the posterior covariance to surface identifiable *combinations*. All of this is
  post-processing on posteriors Stage A already produced.
- **3.3 negative control** -- `d1`+`d2`, provably inseparable below C14 (only `12*d1 + d2` is
  identifiable) and separating as longer chains enter. If the shrinkage diagnostic does not
  show these two individually unconstrained, the diagnostic is not trustworthy for Tier 2.
  Run on C8 (where it should fail cleanly) and C14 (where separation begins).
- **SI items, all free from Stage A**: convergence diagnostics per run, divergence counts at
  the chosen `target_accept`, stranded-chain gap detection (already validated), chain-count
  diagnostic power (already measured).

This stage is where the paper gets the most result per A100-hour in its entire plan.

### Stage C -- cheap targeted experiments (~45 A100-h, days)
- **3.6 rate-limiting conditions** -- already designed (`step_limit_conditions.py`); two
  condition sets matched on count and noise, same free parameters, compare draws to
  convergence and posterior width. Predict the outcome from `info_vs_conditions.py` first so
  the test also scores whether that cheap analysis predicts sampling efficiency.
- **3.6 data-type grid / Figure 9** -- 3 timings x 3 measurement types. Blocked on two new
  modules (section 4). Rank by shrinkage gained *per data point*, not raw shrinkage.

### Stage D -- infrastructure-gated, cheap compute (days once built)
- **3.4 / Figure 7** -- posterior-integrated sensitivity vs. Morris-at-point-estimate, nine
  enzymes x three objectives. Tier 1 can carry the full three-objective comparison **because
  C14+unsat engages FabA/FabB** -- that is the main reason an unsaturated system is in the
  Tier-1 set at all.
- **3.5 / Figure 8** -- re-run the FabF/FabB:TesA optimization across posterior draws. Tier 1
  gives at most an attenuated version (truncated chain-length range); Tier 2 is the primary
  evidence regardless.

### Stage E -- replicate-heavy (~125 A100-h, 1-2 weeks, highly parallel)
- **3.1 Figure 3** -- N=10 pilot first (the outline's own gate), then N=40 if the curve is
  sane. SBC rank statistics (Talts et al. 2018) come nearly free from the same replicates and
  give a full rank histogram plus a uniformity test, so **always compute both**.
- **3.1 Figure 4** -- prior misspecification as a dose-response (shrinkage/z-score/coverage
  vs. mismatch in prior SDs), and noise swept over 5/10/20/40%. Not pass/fail.

### Stage F -- 3.2 grouping (~600 A100-h, gated)
Three variants per tested group: (a) true grouped model, (b) constants freed independently,
(c) an off-manifold data-generating variant where the true values do *not* share the fixed
ratio -- the negative control that shows the independent model is correctly favoured when
grouping is genuinely wrong. Compare by LOO, reporting `delta elpd +/- SE` plainly, and
**gate on Pareto-k > 0.7 before trusting any of it** -- 3.1 already hit degenerate LOO ("all
tail values are the same") on a tightly constrained single parameter.

Start with **one group on one system** and measure before committing the rest. The cost here
is extrapolated, and this stage alone is larger than everything before it.

### Stage G -- Tier 2
Identical recipe, full ME1 model, real data. See section 5.

## 4. Infrastructure that does not exist

Ordered by what blocks the most.

1. **Freeing individual rate constants (blocks all of 3.2).** A free parameter is currently a
   *scaling group* named on each reaction in the YAML (`scaling_group` in
   `reaction_model_builder.py`), and `get_free_parameter_names()`
   (`inference_runner.py:265`) reads only `param_name`. Configs carry an `rxn_name` field but
   it is `None` everywhere it is written. Freeing a group's constituents independently --
   the entire point of 3.2 -- has no verified path. **Check this before planning 3.2**: it is
   either a small wiring job or a model-builder change, and which one decides whether 3.2 is
   weeks or months away.
2. **Off-manifold data generation (blocks 3.2's negative control).** Generating synthetic data
   where tied constants deliberately do not share the fixed ratio.
3. **Posterior-integrated sensitivity (blocks 3.4).** No module matches; `morris_screen.py`
   does Morris at a point, not integrated over draws.
4. **Acyl-ACP intermediate observables (blocks 3.6's richest grid cells).** The outline notes
   this is a copy of the existing final-product observable module with a different species
   list, at **zero additional ODE-solve cost** since the full state trajectory is already
   computed.
5. **Initial-rate data generation (blocks 3.6's timing axis).** A short-time-window variant of
   the existing generator.
6. **Optimization across posterior draws (blocks 3.5).**
7. **Multi-parameter recovery reporting.** Per run: is truth inside the 50/90/95% interval,
   z-score, shrinkage vs. prior. Exists for `a1` on the ladder; needs a multi-parameter
   version. **This gates Stage A being interpretable at all.**
8. **Multimodality toy case (3.1).** A purpose-built symmetric construction (two
   interchangeable parameters / label switching) as a known-answer test.

Already present, contrary to what might be assumed: **dense mass matrix**
(`is_mass_matrix_diagonal` in `resumable_sampler.py:154`), so the SI's diagonal-vs-dense
known-answer test needs no new sampler code.

## 5. Tier 2: now estimated

**The real datasets carry 5 conditions** (user, 2026-09-16) -- two fewer than Tier 1's
seven. Cost is roughly linear in conditions, since each unique initial condition is one ODE
solve per gradient evaluation, so the real data is *cheaper per gradient* than the synthetic
data it is validated against. That removes the largest unknown in this map.

Full ME1 is the untruncated model, ~C20+unsat scale (586 reactions) = 2.98x C12 per draw;
times 5/7 for conditions = 2.13x a Tier-1 C12 run.

| free parameters | per fit | x3 datasets |
|---|---|---|
| 2 (duo) | **9 A100-h** | 26 |
| 3 (trio) | **46 A100-h** | 137 |
| 4 (quad) | 213 A100-h (extrapolated) | 640 |

So a two- or three-parameter Tier 2 is comfortably affordable -- a trio on the full real
model costs about half of Stage A. The cliff is at four parameters, the same place every
other estimate in this document breaks down, and for the same reason: nothing above three
free parameters has ever been run.

**What this changes:** Tier 2 is no longer the scary unknown at the end of the plan. The
prioritized subset from the 3.3 screen should be held to **three parameters** unless the
quad multiplier gets measured first. If it is, Tier 2 costs less than the Tier-1 work that
justifies it, which is the right shape for the paper.

Still to confirm: where the real datasets live (they are not under `Data/`), and whether all
five conditions carry the same observables as the Tier-1 design.

## 6. What could invalidate this plan

- **`tune=300` at three or more parameters.** Validated at two (acceptance 0.777-0.780 vs. a
  0.8 target, first-block drift < 0.12 sd, r-hat shifting <= 0.004 when the first block is
  dropped, zero divergences). The trio gate in Stage A exists for exactly this.
- **`a1`+`c2` never converged** despite gamma 1.21-1.58 saying well separated. Unresolved, and
  it is the SI's own evidence that a diagonal mass matrix cannot represent that correlation. A
  short pilot decides whether it is geometry or data; it should not block the waves.
- **The quad and grouping multipliers are extrapolations.** Nothing above three free
  parameters has ever been run. Stage F must start with one group on one system.
- **LOO degeneracy.** Already observed at one tightly constrained parameter. 3.2's entire
  decision rule rests on LOO, so the Pareto-k gate is not optional.
