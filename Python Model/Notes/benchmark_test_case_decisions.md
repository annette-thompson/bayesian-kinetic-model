# Benchmark test case: decision record

Purpose: a base configuration on which Bayesian inference runs as fast as possible
**without foreclosing any downstream test**, so that later changes (more parameters,
different data types, different systems) are measured against a cheap, well-behaved
reference rather than against a slow one whose own noise hides the effect.

Every decision below is made by a computed metric, and the metric is stated so the
same procedure reproduces the choice on a different system. Where a threshold is
needed it is expressed relatively (a fraction of the strongest parameter, a multiple
of the system's own baseline cost) rather than as an absolute constant, so it
transfers. Tools: `job_files/bench/select_testcase.py`, `job_files/bench/cost_profile.py`.

---

## D1. System: C8

**Metric:** per-draw cost decomposed into sampler work and solve work, measured on
completed a1 runs.

| | C4_NoFB | C6 | C8 |
|---|---|---|---|
| Species | 45 | 79 | 119 |
| Leapfrog steps/draw | 3.3 | 2.8 | 2.7 |
| Draws to converge | 1800 | 1400 | 1400 |
| Total ODE solves | 419,920 | 176,480 | 174,520 |
| sec/draw | 8.7 | 11.0 | 21.8 |
| Posterior sd(log) for a1 | 0.2957 | 0.0162 | 0.0106 |

**Reasoning.** C4_NoFB is the smallest system but the **most expensive in total**
(2.4x C6's ODE solves), because its single observable cannot separate a1 from a global
rate rescaling, costing it both more draws and more leapfrog steps. Cheapness measured
per draw would have selected it; cheapness measured per *converged run* rules it out.
C8 costs 1.98x C6 per draw with identical sampler behaviour, buying a third observable
and a second elongation step.

**Known limitation, recorded so it is not rediscovered later.** C8 does *not* exercise
FabF/FabB chain-length specificity: their condensation constants are identical at C6
and C8 (FabF 254.3 both; FabB 162.6 both), first differ at C10 by 1.4%, and only
collapse meaningfully at C16 and above (FabF 254 -> 80 -> 6.2). Any test of chain-length
specificity therefore requires an expensive system and cannot be done on the base case.

---

## D2. Free parameter: capability gates before cost

A parameter that is cheap because the data cannot constrain it would converge quickly
and teach nothing. So candidates are screened on capability first, and anything failing
a gate is excluded regardless of speed.

| Gate | Requirement | Why a failure forecloses a test |
|---|---|---|
| **G1** | ties >= 2 constants with **distinct** nominal values | A group tying one constant, or several identical ones, is mathematically identical to freeing a raw rate constant, so it cannot test whether the *grouping* is supported |
| **G2** | present in all 14 chain-length systems | Results must transfer along the ladder |
| **G3** | not inside a compound expression | `(b1/b2)` makes a posterior partly a statement about b2, so per-parameter shrinkage is unattributable |
| **G4** | mu* >= 10% of the strongest parameter on some objective | Below this the posterior equals the prior |

**Result on C8: 3 of 17 pass.** Notable exclusions:

- **a2** ranks first for chain length across the ladder but ties **26 constants that are
  all the same value** (spread 1.00x). It is a grouping in name only and fails G1.
- **b3** likewise: 6 constants, one value.
- **b1, d1, d2** fail G3 (compound expressions).
- **x1-x4** fail G2 (absent from C4_NoFB).

**Discriminating among survivors: spread of tied constants.** Count of distinct values
is a weak metric; how far apart they are determines whether the grouping is a
substantive claim.

| Group | Constants | Distinct | Spread | Enzymes | mu* (% of top) | sigma/mu* |
|---|---|---|---|---|---|---|
| c2 | 14 | 5 | **14.7x** | 4 | 55.5% | 1.77 |
| a1 | 4 | 2 | 3.69x | 2 | 82.0% | **1.50** |
| c3 | 3 | 3 | 2.03x | 1 | **100%** | 1.55 |

**Decision: a1 as the base case; c2 designated for the grouping test.**

a1 is chosen because it has the **lowest interaction ratio** of the three (1.50), which
makes single-parameter results more likely to survive into joint inference; it is
strongly identified (82% of top); and it already has validated posteriors on three
systems, which both provides continuity and is what calibrated the prior rule in D4.
Its grouping test is real if modest (two values 3.69x apart across FabD and FabH).

c2 is the strongest available grouping test (14 constants, 4 enzymes, 14.7x spread) but
is a worse *reference*: lower influence and the highest interaction ratio. It is
therefore a grid cell for Section 3.2 rather than the base case.

c3 has the highest influence but the weakest grouping test (one enzyme, 2.03x spread).

---

## D3. Data configuration

**Structural fact, verified in `inference_runner._build_simulator`:** one `SaveAt`
carrying every requested time is applied to every condition, and `solve_all_conditions`
maps over the condition matrix. **Cost equals the number of unique initial conditions.**
Extra time points ride on solves that already happen.

Consequences, each measured rather than assumed:

- **Time points are free.** Measured on C6: 1100.8 ms/gradient with the timeseries,
  1102.2 ms without. Dropping it saves nothing and halves the data. Therefore
  **maximise time points**.
- **Conditions are the cost.** They are the only term that multiplies solve count.
- **The baseline condition is the most expensive on C8** (84 steps against 62 for the
  cheapest, ranking 9th of 10). Selecting conditions cheapest-first measured ~14%
  faster at matched count. For a test system there is no requirement that conditions be
  experimentally representative, so this is free.

Remaining choices (condition count, time horizon) are set from the cost profile grid;
see D5.

**Horizon caveat recorded:** the real experiment runs 720 s. Any shortened horizon is a
test-system convenience and must never be quoted as an experimental condition.

---

## D4. Prior rule

Three earlier criteria were tried and rejected, recorded so they are not retried:

1. **Step-count ceiling.** Bounds where the ODE is affordable; silent on where the data
   constrains. A solvability criterion, not an information one.
2. **Sensitivity threshold** (does a 2x move exceed the 10% noise on one observable).
   Too conservative by roughly sqrt(n): it declared a1 uninformative below 1.0, while
   a1's measured posterior on C4_NoFB is [0.422, 1.296] with 66% of its mass there. A
   prior built from it would truncate two thirds of the real answer.
3. **Merging solvability and informativeness into one stopping rule.** Hides which
   binds: a2 reported [1, 2], which reads as low information but was purely the step
   ceiling firing.

**Adopted: profile log-likelihood.** Bounds are where the aggregate log-likelihood
falls a fixed amount below its maximum, evaluated on the real PyMC model so the profile
is exactly what inference optimises. Only the observed-data term is used; including the
prior would measure the prior's own curvature.

**Threshold calibrated against measured posteriors** rather than assumed:

| System | Measured 95% | Fitted delta | Residual | delta=2 gives |
|---|---|---|---|---|
| C6 | [0.9667, 1.0307] | 2.15 | 0.0004 | [0.9682, 1.0304] |
| C8 | [0.9792, 1.0207] | 2.10 | 0.0002 | [0.9798, 1.0197] |
| C4_NoFB | [0.4220, 1.2957] | 2.45 | 0.0215 | [0.4877, 1.3254] |

The conventional delta = 2 is accurate to within 0.2% where the posterior is
well-identified, and **truncates by 16% on the wide side** for the poorly-identified,
asymmetric case. Since a prior that is too narrow biases the result while one that is
too wide only costs sampling time, the calibrated **delta = 2.23** is used.

A prior set to exactly the 95% profile interval would be as tight as the answer, so
prior bounds are set at a larger delta and the prior-to-posterior width ratio is
reported per run, making the margin visible rather than assumed.

---

## D5. Sampler settings

Held identical across every benchmark cell so the only things varying are the parameter
and the data: `chains=8`, `target_accept=0.8`, `checkpoint_every_steps=5`,
`rhat_check_every=100`, `convergence_consecutive_checks=2`, `post_convergence_checks=1`.

**Observed:** r-hat is the binding convergence criterion in every completed run; ESS
cleared its 400 floor well before r-hat cleared 1.01 (C4_NoFB had ESS 775 at 500 draws
while r-hat was still 1.0139). The rank-ECDF check never bound. The ESS floor and the
rank-ECDF check are therefore currently inert, and could be relaxed if convergence cost
needs reducing further.

*(Condition count and horizon pending the cost profile grid.)*
