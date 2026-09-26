# Uncertainty-Guided Kinetic Modeling: A Bayesian Framework for Metabolic Pathway Engineering

Working outline, current as of 2026-09-24. The companion run plan for the synthetic-data
(Tier 1) results is `Notes/tier1_experiment_plan.md`. It lists, figure by figure, the exact
fits that produce every Tier-1 panel below, with costs.

---

## Abstract **[to be written once the Section 3 figures are drafted]**

---

## Executive summary: the paper in one page

**The goal.** Build and validate the modeling workflow that *uncharacterized* systems will
require. Our *E. coli* fatty-acid-synthesis (FAS) work (Ruppe et al. 2020, "PNAS"; Mains et
al. 2022, "ME1") succeeded because it had what most systems don't: extensive kinetic
characterization, multiple purpose-collected datasets, and in vivo validation of its
engineering predictions. Applying the same modeling approach to a new organism or pathway
means working without those advantages, and without a way to know how far the resulting
model can be trusted.

**What we're doing.** Refitting the existing model with Bayesian inference, which returns a
*distribution* over each parameter rather than a single value. It's the same model and the
same data; only the fitting machinery changes. FAS is the proving ground because its answers
are already known and independently validated, so we can check whether the new machinery
gives correct answers.

**What this adds for the next system.** Four capabilities that matter most when there is no
prior characterization to fall back on:

1. **How much should this model be trusted?** Quantified, testable uncertainty on every
   parameter and prediction, where there is no in vivo result to check against.
2. **Are the simplifying assumptions supported?** A quantitative, per-group test of whether
   the data support a given grouping of rate constants. In the next model the right groupings
   won't be known in advance.
3. **What does the data actually constrain?** Which parameters are pinned down, which are
   unconstrained, and which are identifiable only in combination.
4. **What should be measured next?** Which experiment types most reduce uncertainty. This is
   transferable guidance for planning characterization of a new system.

**How we know it works.** Two tiers, deliberately using different data:

- **Tier 1 (synthetic data, known answer).** We generate data from the model at a known
  parameter value, then check that the method recovers it. Knowing the true answer in advance
  is the only way to verify the machinery.
- **Tier 2 (real data, unknown answer).** The identical, unmodified pipeline applied to the
  real ME1 measurements. Only real data can show whether Bayesian conclusions *differ* from
  the point-estimate ones.

**How the work is sequenced.** Every results figure is first drafted at Tier 1 on small,
truncated versions of the network, where fits take hours rather than days. Only once all
drafts exist do we choose which figures to recreate on the full model with real data
(Section 2.8). Nothing is run on the full model that no figure needs.

**What the results will look like.** We expect the posterior to confirm ME1's groupings,
targets and enzyme-ratio strategy. That would establish that the method returns correct
answers on a system where correctness can be independently checked. Where the posterior also
surfaces something a point estimate cannot report (a parameter the data leave unconstrained,
or a sensitivity ranking that shifts once uncertainty is carried through), that quantifies
what the added machinery buys, and how much it is likely to matter in a system with less data
behind it.

---

## Status at a glance

| Section | What it answers | Status |
|---|---|---|
| 2. Methods | How the model is solved and fit | Solver settings, negative-concentration handling, sampler settings, Tier-1 data design and Tier-1 systems all settled |
| 3.1 Recovery + calibration | Does it work? | **Single-parameter recovery complete on all 14 truncated systems** (truth recovered everywhere; shrinkage 0.94 to >0.9999). Two-parameter pilots done. Multi-parameter recovery (Fig 2), calibration (Fig 3) and robustness (Fig 4) planned, not run |
| 3.2 Parameter grouping | Were the groupings justified? | Designed (split-group test on `c3`); not run |
| 3.3 Identifiability | What does the data constrain? | Parameter-level sensitivity screen complete across all 14 systems and 3 objectives; posterior analysis waits on Fig 2's fit |
| 3.4 Sensitivity vs. Morris | Same enzyme targets as before? | Designed; needs a posterior-integrated sensitivity module |
| 3.5 Ratiometric strategy | Is the enzyme-ratio heuristic robust? | Designed; needs an optimisation-across-draws module |
| 3.6 Experimental design | What should we measure next? | Designed; initial-rate data exists, the acyl-ACP observable module does not |
| 4. Conclusion | Synthesis | Waits on 3.2-3.5 |
| SI | Methodological-validity checks | Chain-count and stranded-chain items done; the rest are planned alongside Tier 1 |

**Open items:**
1. Which parameters to fit at Tier 2. Choose after the Tier-1 identifiability results
   (Fig 6), holding to three free parameters unless a four-parameter cost has been measured.
2. Confirm which datasets the ME1 point-estimate fit used (Section 2.1).

---

## 1. Introduction

### 1.1 Motivation

Fitting a mechanistic kinetic model to noisy data with a single-point solution, then using
that one solution to guide downstream engineering, silently discards the uncertainty that
should inform those decisions. This paper validates a Bayesian alternative on a system where
the answer is already known, so the validation itself can be checked rather than assumed.

The standard workflow fits an ODE model's parameters by least squares, then runs sensitivity
analysis and point optimisation on that single fitted solution. That leaves blind spots that
uncertainty analysis fills:
- There is no confidence metric on how sure we are in a best-fit point.
- There is no identifiability check. A successful fit doesn't mean the data constrain every
  parameter; multiple solutions can hide behind one converged point.
- Sensitivity screens and point-optimisation results are both anchored to that one estimate.
  There is no way to know whether the "important" steps or "optimal" concentrations would look
  different under an equally plausible parameter set.
- Simplifying assumptions go untested. Here that means grouping related rate constants to
  reduce the number of fitted parameters; whether a grouping meaningfully affects the fit was
  not something the workflow could check.

These blind spots matter most where the field is headed. As less-characterized pathways and
non-model organisms are targeted, data get sparser and parameter uncertainty grows, which
makes a validated method for parameterizing under high uncertainty necessary rather than
optional. Our own FAS work is a representative instance of the standard workflow and its
blind spots. It is also one of the few systems characterized well enough to check the
Bayesian alternative against an existing, independently validated answer.

The scope extends beyond metabolic engineering. Chemocatalytic cascades, other in vitro
enzyme cascades and in vivo pathways have historically used the same single-point-estimate
approach for multi-step reaction-network rate constants.

### 1.2 Research questions

**The three central questions:**
- **Does it work?**
  - Can it recover the previous model's solution when given synthetic data?
  - How do noise and prior misspecification affect the final solution?
- **What can it do?**
  - Which parameters are identifiable?
    - How does this change with the types of data given?
  - Which parameters are correlated?
  - Does it find multiple sets of solutions?
  - Can it test the validity of our previously designated scaling groups?
  - Can it design maximally informative experiments to guide further modeling?
- **What did it find?**
  - Is the previous solution in the set(s) of solutions it gives?
  - How does an uncertainty-informed sensitivity analysis change calculated enzyme influence?
  - What concentrations maximise or minimise average chain length, and do they differ from
    our previous results?

**Not answering / out of scope:**
- Not claiming perfect biophysical accuracy of the fitted constants.
- Not (necessarily) presenting new *E. coli* FAS biology as the contribution.
- Not yet applied to an uncharacterized or non-model system.

### 1.3 System overview
- Type II FAS elongation cycle: the FabD-TesA reaction network with nine enzymes (FabA, FabB,
  FabD, FabF, FabG, FabH, FabI, FabZ, TesA).
- **Figure 1** (network diagram, plus a schematic of the old vs. new workflow):
  - (old) minimisation → point estimate → {Morris sensitivity, concentration optimisation,
    mutation/knockout modeling}
  - (new) priors → posterior → posterior-integrated sensitivity and optimisation (exact panel
    content TBD).

## 2. Methods

### 2.1 Previous approach
- Full model (Mains et al. 2022, "ME1"): 194 kinetic constants, 328 reactions, 318 species,
  fit through 14 scaling parameters, each of which multiplies a group of similar constants.
- The Python implementation used here is structurally identical to the ME1 MATLAB model. A
  full structural diff matches all 588 directed reaction steps and all rate constants. The
  only differences are inert: an unused ACC module, and a FabF acetyl-CoA branch whose
  constants are all zero in ME1.
- The fitted scaling values from ME1 are already built into the reaction files' rate
  constants. So in this parameterisation the previous solution is the point where every
  scaling parameter sits at its no-op value.
- Fitting: simultaneous least squares **[confirm which datasets: Ruppe et al. 2020 and/or
  the ME1 Dataset S1 measurements]**.
- Morris-method elementary-effects sensitivity screen against three objectives (PNAS).
- Model-guided ratiometric enzyme-concentration tuning (ME1).

### 2.2 Mechanistic kinetic model
- Stiff adaptive ODE solver: Kvaerno5 with a PID-controlled step size (JAX/diffrax).
- **Solver settings:** `atol=1e-7, pcoeff=0.4, icoeff=0.3, dcoeff=0.0, dt0=1e-6,
  max_steps=20000`, with `rtol` selected **per system**:

| `rtol` | Systems | Cost vs. 1e-5 |
|---|---|---|
| 1e-3 | C4_NoFB, C6, C8, C10, C12, C12+unsat, C14, C14+unsat | 1.26 to 1.59x cheaper |
| 1e-4 | C16, C16+unsat | 1.04x cheaper |
| 1e-5 | C18, C18+unsat, C20, C20+unsat | unchanged |

**How these were chosen.** This matters because the ODE solve runs inside every likelihood
evaluation, so its cost and accuracy set the whole study's feasibility.

1. Swept 4 PID settings × `rtol` ∈ {1e-4, 1e-5, 1e-6} × `atol` ∈ {1e-6, 1e-7, 1e-8} (36
   combinations) on the largest model.
2. Scored each on three axes: the number of usable test conditions found, the solver steps
   required, and the accuracy against the same PID run at near-exact tolerance
   (1e-10/1e-12).
3. Carried the survivors through every chain-length model, to confirm they generalise rather
   than suiting one case.

Two regions fail regardless of PID setting and are excluded:
- `rtol=1e-4`, where 60-100% of solver steps are rejected.
- `rtol≈1e-5` paired with `atol=1e-6`, which gives 0.08-0.40% error, ~30-100× worse than any
  other combination.

Convergence at the strict reference tolerance also proved **PID-dependent, not merely
tolerance-dependent**: settings with a proportional coefficient of 0.1 or 0.3 fail to converge
on the shortest-chain model at any `atol`. The chosen setting cleared all chain lengths
identically on CPU and GPU. Across all 14 chain-length models it needs at most 220 solver
steps (mean ~127), with a maximum error of 0.0067% (mean 0.0009%) against the near-exact
reference.

**Why `rtol` varies by system.** With the PID coefficients fixed, we measured total
integration steps at `rtol` of 1e-3, 1e-4 and 1e-5 for every system. We kept the cheapest
tolerance that still produced a full set of diverse test conditions. A looser tolerance is not
reliably cheaper:
- At C18 the loosest setting costs 1608 steps, against 1438 at 1e-5.
- At C20+unsat the middle setting costs 3478, against 1399.

In both cases the step-size controller starts rejecting steps faster than it saves them. A
rule that simply tried the loosest tolerance first would have assigned those systems their
most expensive option. Numerical error stays negligible throughout: the largest disagreement
between any kept condition and the near-exact reference is 0.05%, roughly 200 times smaller
than the 10% relative noise applied to the data.

**Negative concentrations.** The right-hand side is evaluated on the solver's state as-is,
with no clamp to zero. Solves that fail inside the sampler return non-finite values
(`EQX_ON_ERROR=nan`), which the sampler treats as rejected proposals rather than crashes. A
clamp was tested and rejected for three reasons:
- It did not prevent the linear-solver failures it was meant to prevent, which arise in the
  gradient pass.
- It slowed sampling as systems grew, by up to ~8-9x at C10.
- It stopped sampling altogether above ~C8-C10 (0% acceptance).

Under the production prior, the unclamped model's log-likelihood differs from the clamped
one's by at most 4×10⁻⁷ nats at posterior draws. An alternative linear-solver setting
(`AutoLinearSolver(well_posed=False)`) was also rejected: it returns confidently wrong
solutions instead of failing.

### 2.3 Parameter-level global sensitivity screen

The earlier work screened **enzyme concentrations** (Ruppe et al. 2020, Fig. 3E), which
answers "what should an engineer titrate". Choosing which parameters are worth inferring is a
different question, so we applied the same method directly to the 18 kinetic scaling
parameters.

- **Method:** Morris elementary effects, radial design with Latin hypercube base points,
  matching the earlier implementation (SAFE toolbox, radial method). `r = 200` base points, so
  `N = r*(k+1)` model evaluations per system, roughly 3,600 to 3,800.
- **Coverage:** all 14 chain-length systems. The objectives are the same three the earlier
  work used: total production, average chain length, and (for systems with an unsaturated
  branch) unsaturated fraction.
- **Sample space:** uniform in log10 over a 100-fold window centred on each parameter's
  nominal value. We sample in log space because scaling parameters are multipliers; a linear
  sweep over the same window would put 90% of its mass above the nominal value and barely
  probe reductions.
- **d-group handling:** the `d`-prefixed parameters enter additively inside an exponential,
  so a raw value is not comparable to a multiplier. They are sampled on the rate multiplier
  they induce and then converted, which puts all 18 parameters on one axis.
- **Convergence** was checked as in the earlier work: mean elementary effects were averaged
  over random subsets of trajectories to confirm that the ranking stabilises.
- **Reported quantities:** `mu*` (mean absolute elementary effect) ranks influence;
  `sigma/mu*` indicates how much a parameter's effect depends on where the other parameters
  sit, i.e. nonlinearity and interaction.

### 2.4 Synthetic data generation (Tier 1)

**Truncated systems.** Tier 1 uses truncations of the full network that cap the maximum
fatty-acid chain length, while keeping all nine enzymes at every rung. There are 14 rungs:
- C4_NoFB
- C6 through C20, saturated only
- C12 through C20, with the unsaturated branch

The rungs are generated from the full reaction set by script and have zero dead species at
every rung. C4_NoFB is the one exception to the nine-enzyme rule. It omits FabF and FabB,
because their elongation step is inactive at a 4-carbon cap and they otherwise sequester ACP
in dead-end complexes.

**The Tier-1 dataset mirrors the three kinds of measurement in the real ME1 data** (Section
2.8): one time course, a chain-length profile and a set of initial rates. All data are
generated at the known no-op parameter values, i.e. at the previous model's own solution.

| Dataset | Condition | What is recorded |
|---|---|---|
| Time series | baseline | total fatty acid in C16 equivalents at 10 times, 72-720 s |
| Profile | baseline | each fatty-acid species (saturated and unsaturated) at 720 s |
| Initial rates | 5 conditions | C16 equivalents accumulated by 150 s ÷ 150 s, in µM C16/min, the same definition as the measured rates |

The five initial-rate conditions are: baseline; FabH lowered 10x (0.1 µM); FabB knocked out;
TesA lowered 20x (0.5 µM); FabZ knocked out.

"C16 equivalents" is the carbon-weighted total, Σ (n/16)·[C*n* FA], summed over saturated
and unsaturated species. This matches how total fatty acid is quantified against a C16
standard.

The baseline initial concentrations are 500 µM malonyl-CoA, 500 µM acetyl-CoA, 1000 µM NADPH,
1000 µM NADH, 10 µM ACP, 10 µM TesA and 1 µM of every other enzyme. The baseline is identical
across all systems; only the chain-length cap varies.

**How the rate conditions were chosen.** Each enzyme was scanned over a one-significant-figure
grid of concentrations from 0 to 30 µM. A setting was kept only if it met all four criteria on
every candidate system:
- It moves the initial rate at least 0.2 decades from baseline. (FabB's knockout reaches
  0.19 decades on C14+unsat, the closest miss, and is kept as the only handle on the
  unsaturated branch.)
- It keeps output above 5% of baseline.
- It needs no more than 1.5x the baseline's solver steps.
- It gives the same answer when re-solved at near-exact tolerance.

Together the four perturbations touch 12 of the 14 live scaling parameters:
- FabH adds `a1`, `c1` and `e`.
- FabB adds `c2`.
- TesA adds `a3`, `c3`, `d1` and `d2`.
- FabZ adds `c4`.

FabD (`b2`) and FabA (`f`) have no usable setting anywhere in the range. FabF adds nothing
beyond FabB and FabH, and FabB is the only perturbable handle on the unsaturated branch.

**Noise.** Every dataset carries injected Gaussian noise with a per-point standard deviation
of 10% of the clean value plus an absolute floor (0.01 µM for concentrations, 0.004 µM C16/min
for rates). The standard deviation used to draw the noise is written alongside each value, and
the likelihood uses exactly that value. The noise is not clipped at zero, because clipping
would bias near-zero points against the unbounded Normal likelihood.

**Model/data consistency check.** The data are generated at tight tolerance (`rtol` 1e-8,
`atol` 1e-10) with the production PID coefficients, so they carry no solver error of their
own. Before any inference run, the fitting model is built from the run's own configuration,
solved at the data-generating parameter values at the system's working tolerance, and
compared against the noise-free copy of the data. The agreement is 10⁻⁶ to 10⁻⁵ relative
error, which is the working tolerance's own error and about four orders of magnitude below
the noise. The same check confirms that the log posterior and its gradient are finite where
the sampler starts. This guards against data and fitting model being built from subtly
different parameterisations. In that case inference can't succeed at any parameter value,
and the symptom (a sampler that fails to converge) looks like a statistical problem rather
than a bookkeeping one.

*Parameterisation note.* The scaling parameters are a re-parameterisation layered over the
published rate constants. Most are multiplicative and are no-ops at 1. The `d`-type
parameters enter additively inside an exponential and are no-ops at **0**. Setting them to 1
distorts one TesA-linked rate by a factor that grows with chain length (~4×10⁵ at C12 to
~4×10¹² at C20). Every run's configuration therefore states all scaling values explicitly
rather than relying on code defaults.

**Single-parameter ladder data (Section 3.1's completed table).** The recovery ladder used an
earlier, simpler design: a baseline time series of the individual fatty-acid species plus
baseline and perturbed endpoints, selected by an automated search. Those datasets were
generated **without** injected noise, while the likelihood assumed a 10% standard deviation.
The ladder therefore demonstrates recovery, convergence and cost scaling, but not calibration.
Calibration is tested on the noisy Tier-1 data (Fig 3).

### 2.5 Bayesian inference framework
- **Priors:** LogNormal on each multiplicative scaling parameter, with the median pinned to
  the no-op value 1 and 95% of the mass in [0.1, 10] (sigma = 1.18). The `d`-type parameters
  get Normal priors centred on 0, matched so that the rate multiplier they induce spans the
  same [0.1, 10] window. Wider windows ([0.01, 100], [0.001, 1000]) were tested on the ladder.
  They put most of their mass where the observables no longer respond to the parameter, which
  costs warmup without adding information. They also make a spurious secondary posterior
  maximum (at `a1`≈9.4, thousands of nats below the true mode) reachable at initialisation.
- **Likelihood:** Normal, with the per-point standard deviations written into each dataset
  (Section 2.4).
- **Sampler:** NUTS (BlackJAX), driven one adaptation step at a time so runs checkpoint and
  resume exactly across cluster job limits. The model and log-density come from PyMC; the
  forward model is the JAX ODE solve.
- **Convergence diagnostics:** rank-normalised r-hat and bulk ESS (Vehtari et al. 2021), with
  automated early stopping once both criteria are met on two consecutive checks. A
  rank-normalised ECDF mixing check, computed against a simulation-calibrated simultaneous
  confidence band.
- **Stranded chains:** a chain whose mean log-posterior sits far below its siblings' is
  excluded from diagnostics and posterior summaries (never from the saved draws), provided
  enough chains remain to diagnose convergence (SI).
- **Model comparison:** LOO-CV with Pareto-smoothed importance sampling, used in Section 3.2
  to compare grouped and split parameterisations. LOO is uninformative when the posterior has
  almost no spread; it fails outright on the single-parameter ladder from C8 upward. So every
  comparison is gated on Pareto-k first (Section 3.2).

### 2.6 Final inference settings

| Setting | Tier-1 multi-parameter runs | Single-parameter ladder (completed) |
|---|---|---|
| chains | 4 | 8 |
| warmup | 300 | 1000 |
| convergence rule | r-hat ≤ 1.01 AND bulk ESS ≥ 100 × chains | r-hat ≤ 1.01 AND bulk ESS ≥ 400 |
| checked every | 100 draws, two consecutive passes | 100 draws, two consecutive passes, plus one extra 100-draw block |
| minimum chains after stranded-chain exclusion | 3 | — |
| `target_accept` | 0.8 | 0.8 |

- **Why 4 chains:** a chain-count test (C12, `a1`+`c3`, 4 to 64 chains) reached convergence
  in 4.0 A100-hours at 4 chains, against 7.1 at 8. Four is also Vehtari et al.'s recommended
  minimum. The same runs quantify how much the diagnostics' resolution depends on chain count
  (SI).
- **Why 300 warmup steps:** at two free parameters, acceptance over the last 50 warmup draws
  is 0.777-0.780 against the 0.8 target, and the first sampling block's mean sits within 0.12
  sd of the rest. Dropping the first block changes r-hat by ≤ 0.004, and there are zero
  divergences. For three-parameter fits this is re-checked at the end of warmup, and the
  warmup length is raised if needed.
- Approximate compute cost: **[fill in from the Tier-1 runs, in A100-hours]**.

### 2.7 Code & data availability
- Link to GitHub repository.

### 2.8 Validation strategy

Two tiers, deliberately using different data.

**Tier 1** uses simulated data with known ground truth. Knowing the true answer in advance is
the only way to check that the machinery gives correct answers (simulation-based calibration;
Talts et al. 2018).

**Tier 2** uses the real ME1 measurements. Only real data can reveal whether the Bayesian
analysis's conclusions *diverge* from the previous model's, which is the point of Sections
3.4-3.5. A synthetic test can only recover whatever truth was put into it.

**Tier-1 systems.** Most Tier-1 figures come from one main system, **C14+unsat** (352
reactions). It is the smallest truncation that meets all three requirements below, and all
condition-design analyses were run on it:
- It engages the FabA/FabB unsaturated branch, so all three sensitivity objectives in 3.4 are
  meaningful.
- Its chain-length profile is spread across several species (6 species above 1% of the
  total, against 1 at C8 and 3 at C12), so the profile carries real information.
- It is where the two TesA free-energy parameters `d1` and `d2` begin to separate (3.3).

Figures that need many repeated fits (calibration, robustness) use a cheap two-parameter
model on **C8**, provided C8 recovers both parameters cleanly; otherwise they use C12.
Calibration tests whether the machinery's stated uncertainty is honest, which does not depend
on network size.

**Drafting first, then choosing what goes to Tier 2.** All Section 3 figures are drafted at
Tier 1 before any Tier-2 run. The default expectation:
- Figs 3 and 4 and the known-answer controls stay Tier-1-only, since they need a known truth.
- The recovery question becomes "is the published estimate inside the real-data posterior".
- Figs 6, 7 and 8 are recreated from one Tier-2 fit.
- Figs 5 and 9 go to Tier 2 only if their Tier-1 drafts show the analysis discriminates, and
  only as far as the real data allow.

**Tier 2 data (ME1 Dataset S1):**
- 17 initial-rate conditions (µM C16/min). These are the reference plus FabH, FabF, FabB and
  acetyl-CoA knockouts, and a FabB × FabH titration without FabF. The FabH-knockout
  conditions among them are old data and are not used.
- 14 GC/MS chain-length profiles (C4-C18, µM) across FabF/TesA/FabH/FabB combinations.
- One reference time course (6 points to 720 s).

**FabH-knockout conditions: excluded.** They are old data and are not used at Tier 2
(decided 2026-09-25), so the model's zero fatty acid in six of them no longer blocks Tier 2.
The model's reference condition agrees well with the measurements (rate within 1.11x, time
course within 0.78-1.19x). The Tier-1 perturbations were chosen independently of the real
conditions, so the Tier-1 drafts preview the methods, not the exact information content of the
real data.

## 3. Results / Discussion

### 3.1 Recovering the point estimate, with uncertainty, and is that uncertainty trustworthy?

> **In one sentence:** before trusting anything this method says about the real system, show
> that it recovers a known answer from synthetic data *and* that its stated uncertainty is
> honest rather than merely centred correctly.

- **Recovering the previous model's solution** (*Does it work?*): the reaction files' rate
  constants already *are* the ME1 fit, and the scaling parameters sit at their no-op values
  there (Section 2.1). Data generated at those values is data generated at the previous
  model's own solution, so fitting back to 1 (or 0 for `d`-type parameters) *is* the recovery
  test.
- **Is the previous solution in the posterior's credible region?** (*What did it find?*): an
  explicit check, distinct from mode/mean recovery. Does the truth fall inside the 50/90/95%
  central intervals, not just near the centre?
- **Calibration:** a distinct question from recovery, since centring on the right mean
  doesn't mean the uncertainty is honest. Across many synthetic datasets with truths drawn
  from the prior, credible intervals should contain the truth at their nominal rate.
- **Robustness:** (a) *prior misspecification*: how recovery degrades as the truth sits
  further from the prior's centre, measured in prior SDs, as a dose-response curve rather than
  pass/fail; (b) *noise*: the same treatment over 5/10/20/40% relative noise.
- **Multiple solutions / multimodality:** genuinely separate posterior modes, as distinct
  from one broad or ridge-shaped region. NUTS mixes poorly across separated modes, so this
  needs an explicit check:
  - A purpose-built symmetric toy model (two interchangeable parameters, a label-switching
    construction) as a known-answer test.
  - Reporting whether independently initialised chains settle in disagreeing regions. The
    ladder already contains two real cases: a secondary maximum of the computed posterior at
    `a1`≈9.4 (C10) and `a1`≈0.43 (C8), 533-6,975 nats below the true mode, trapping chains at
    initialisation. With only one free parameter, a second physically valid solution isn't
    expected, so these are numerical features of the posterior surface, not alternative
    kinetic regimes. They are caught by the stranded-chain check (SI).

**Completed: single free parameter (`a1`) on all 14 truncated systems.** `a1` ranks first for
total production and second for average chain length in the sensitivity screen (Section 2.3).
It is present in all 14 systems, never appears inside a compound expression, and its solve
cost stays flat across the whole prior. `a2` ranks higher for chain length, but its solvable
range differs so much between systems that no single prior covers the ladder.

| | C4_NoFB | C6 | C8 |
|---|---|---|---|
| Observables | 1 (C4_FA) | 2 | 3 |
| Posterior median | 0.911 | 0.9995 | 0.9995 |
| Central 95% | [0.422, 1.296] | [0.967, 1.031] | [0.979, 1.021] |
| True value 1.0 recovered | yes | yes | yes |
| Prior-to-posterior shrinkage | 0.937 | 0.9998 | 0.9999 |
| Sampling draws to converge | 700 | 300 | 300 |
| Precision gain vs. previous row | | **18.3x** | 1.53x |
| What sqrt(n) alone would predict | | 1.41x | 1.22x |

Across all 14 systems, every posterior mean lies within half a posterior SD of the truth, and
shrinkage rises from 0.94 at C4_NoFB to >0.9999 from C6 onward. The data removed nearly all of
the prior's variance, which is stronger evidence that the answer wasn't presupposed than any
argument about the prior's width. (These datasets carry no injected noise; see Section 2.4.)

**The pattern across observable count is a result in its own right.** Adding a second
observable buys a factor of 18 in precision; adding a third buys only 1.5. Pure data volume
would predict 1.41 and 1.22. A single observable supplies only a magnitude, which is hard to
separate from a global rate rescaling. The second supplies the first *ratio* between chain
lengths, a qualitatively different constraint. Later observables add ratios that are largely
redundant with the first. Convergence follows suit: 700 sampling draws for one observable,
300 for two or three. As experimental-design guidance: **measure at least two chain lengths;
the third and beyond give diminishing returns.**

**Completed: two-parameter pilots** (C6, C10, C14). `a1`+`c3` converged on all three systems;
`a1`+`c2` converged on none. With `a1`+`c2`, NUTS's adapted step size never stabilises: it
oscillates between ~0.01-0.03 when crossing the correlated region and ~0.2-0.3 elsewhere.
`a1`+`c3` settles within the first 10-15% of warmup. The `a1`-`c3` posterior correlation is
modest (-0.47 to -0.61) and a diagonal mass matrix handles it. A dense mass matrix cut
`a1`+`c2`'s cost by only ~35% and still hadn't finished warmup in 12 h. This is preliminary
evidence for the SI mass-matrix item; `c2` is kept out of the main Tier-1 fits.

**Figures (Tier 1; runs in `tier1_experiment_plan.md`):**
- **Figure 2:** posteriors for the main three-parameter fit, `a1`+`c3`+`a2` on C14+unsat,
  with truth overlaid. It reports the z-score (posterior mean minus truth, in posterior SDs),
  shrinkage and 50/90/95% coverage per parameter, with r-hat/ESS as an inset. The choice of
  parameters: `a1` leads total production, `a2` leads chain length and unsaturated fraction,
  and `c3` is the TesA handle; the three are well separated in the data. A two-parameter fit
  (`a1`+`c3`) on the same system is the fallback panel.
- **Figure 3:** calibration from simulation-based calibration. A 10-replicate pilot is run
  first, then 40 replicates in total, each with a truth drawn from the prior and a fresh
  noisy dataset (`a1`+`c3`, C8). It shows the SBC rank histogram with a uniformity test and
  nominal-vs-observed coverage from the same replicates.
- **Figure 4:** robustness on the same model. Shrinkage, z-score and coverage are plotted
  against (i) the truth's distance from the prior centre (four offsets) and (ii) noise level
  (5/10/20/40%).
- **Expected results, both directions:** clean recovery with honest calibration at realistic
  noise is the validating result. Degraded but still honest uncertainty under high noise or
  prior misspecification is *also* a valid outcome: the method correctly reports "I don't
  know" rather than converging confidently on the wrong answer. Only a confident wrong answer
  would undermine the paper's central claim, and the two are distinguished explicitly.

**Tier 2:** "is the published estimate inside the real-data posterior", using the Tier-2 fit
chosen after the Tier-1 drafts.

### 3.2 Testing the parameter-grouping assumptions

> **In one sentence:** the old model assumed ~194 rate constants move together in 14 fixed
> groups; this tests whether the data support that, or whether the simplification hides real
> structure.

- Does the data support a given grouping (constants assumed to move by one shared
  multiplier), or does splitting the group reveal structure the grouping doesn't capture?
  Grouping related constants is standard practice at this model size; the contribution here
  is a quantitative test of a given grouping against the data.
- **Figure 5:** posteriors of the grouped and split parameterisations, with Δelpd_loo ± SE,
  and the negative control.
- **Decision rule:** Δelpd_loo between the two models against its standard error (rule of
  thumb: |Δelpd| > 2×SE as meaningfully better). Sivula et al. (2020/2022) show this threshold
  is often overconfident at moderate observation counts like ours, so Δelpd ± SE is reported
  plainly rather than as a bare pass/fail.
- **Gate on Pareto-k first:** k > 0.7 for a held-out point means PSIS has broken down for it.
  Check this before trusting any Δelpd.
- **Expected results, both directions:** if the grouped model is favoured or statistically
  indistinguishable, the grouping is supported. If the split model is favoured, with the
  halves moving apart, the data carry structure finer than the grouping resolves. That would
  be a candidate refinement to the parameterisation, and a demonstration that the diagnostic
  detects one when present.

**Tier 1 (C14+unsat):**
- **Test group: `c3`,** the TesA hydrolysis multiplier. It ties six rate constants with six
  distinct nominal values, one per chain length, so the shared-multiplier assumption is
  substantive. `c3` also sits in the main fit. Groups whose constants share a single nominal
  value (`a2`, `a3`, `b3`) are trivial 1:1 groupings and uninformative as tests.
- **The split:** `c3` is divided into short-chain (C4-C8) and long-chain (C10-C14 plus
  unsaturated) halves (`c3s`, `c3l`) in a script-generated variant of the reaction set. The
  question becomes "is one multiplier enough for this group, or does it need two". Freeing
  all six constants separately would mean a 7-parameter fit, beyond what the data can
  identify.
- **Three variants:**
  - (a) grouped model `a1`+`c3` on the standard data
  - (b) split model `a1`+`c3s`+`c3l` on the same data
  - (c) data generated *off* the grouping (`c3s`=1, `c3l`=3), fit with both models. This is
    the negative control confirming that the split model is correctly favoured when the
    grouping is genuinely wrong. At `c3l`=3 the long-chain species move by −15% to +144%
    while the short-chain ones move about 6%, a pattern one shared multiplier cannot
    produce.

  `a1` is included so the posterior has enough spread for LOO to work.

**Tier 2:** only if the Tier-1 test discriminates (the negative control is correctly flagged
and Pareto-k passes). Then the same test is run on the real data for a prioritised group.

### 3.3 Identifiability: what does the data actually constrain?

> **In one sentence:** which parameters the data genuinely pin down, which they leave
> essentially unknown, and which are knowable only in combination. None of these can be read
> off a point-estimate fit.

- Which parameters' posteriors collapse tightly (well constrained), and which remain close to
  the prior (essentially unconstrained)?
- **Which parameters are correlated with each other:** a parameter can look poorly
  identified individually *because* it trades off against another, which per-parameter
  shrinkage alone won't show.
- **Figure 6:** prior-to-posterior shrinkage per parameter, flagging shrinkage < 0.5 as
  weakly identified.
- **Figure 6b:** the pairwise posterior correlation matrix, plus an eigendecomposition of the
  posterior covariance. The eigendecomposition reveals identifiable *combinations* (e.g. only
  a sum or ratio pinned down) even when individual parameters are not, reported as
  eigenvalues rather than only a visual plot.
- **Parameter selection for the full model:** fitting all 14 scaling parameters at once isn't
  the goal; a data-driven subset is. The earlier enzyme-level ranking (Ruppe et al. 2020,
  Fig. 3E) answers a different question, so the same method was run on the scaling
  parameters across all 14 systems and all three objectives. The table gives median `mu*`
  across systems, as a percentage of the leading parameter for each objective:

| Parameter | Avg. chain length | Total production | Unsat. fraction | median `sigma/mu*` |
|---|---|---|---|---|
| **a2** | **100%** | 43.8% | **100%** | 0.79 to 1.29 |
| **a1** | 76.7% | **100%** | 66.6% | 1.03 to 1.47 |
| b3 | 77.0% | 68.9% | 74.5% | 1.11 to 1.27 |
| c1 | 61.3% | 59.5% | 51.7% | 1.43 to 1.55 |
| c2 | 56.3% | 37.9% | 47.4% | 1.09 to 1.38 |
| c3 | 26.2% | 77.7% | 13.2% | 1.32 to 2.15 |
| d1 / d2 | 40.5 / 42.5% | 57.6 / 54.8% | 15.9 / 15.8% | 0.94 to 2.10 |
| a3 | 22.6% | 20.8% | 7.3% | 1.04 to 2.11 |
| **f** | 3.4% | 12.4% | **85.2%** | 0.95 to 1.61 |
| b1 | 6.6% | 6.1% | 33.0% | 1.35 to 2.13 |
| e, x1, x4, c4 | 2 to 8% | 3 to 15% | 3 to 4% | 1.5 to 3.5 |
| x3 | 1.1% | 0.19% | 0.22% | 2.6 to 3.0 |
| **x2** | **0.026%** | **0.034%** | **0.044%** | 2.1 to 2.8 |

  Three features support using this screen for selection:
  - `a2` leads chain length on 13 of 14 systems. The exception is C4_NoFB, the one system
    without FabF and FabB, which supply 4 of `a2`'s acyl-ACP binding constants. That is a
    controlled test of the mechanism inside our own ladder.
  - `f`, which exists only in unsaturated systems, ranks second for unsaturated fraction at
    85% of the leader while sitting at 3-12% for the other two objectives. The screen resolves
    real structure rather than noise.
  - The bottom tier is stable across every system and objective, with `x2` roughly **3,800
    times** below the leader for chain length.

  One caveat applies to every downstream use: `sigma/mu*` exceeds 1 for almost every
  parameter, so interactions are pervasive, and a one-at-a-time analysis won't extrapolate
  cleanly to joint inference.

- **Worked identifiability case: `d1` and `d2`.** They enter TesA binding as
  `1/exp(n*d1 + d2)`, where `n` is fixed at 12 for chains up to C12 and becomes `2*chain - 12`
  above it, reflecting the steric penalty of a longer acyl chain. Up to C12 every d-scaled
  reaction therefore shares one coefficient, so only `12*d1 + d2` is identifiable and the two
  parameters are **provably** inseparable. Their sensitivity curves are bit-for-bit identical
  on C6. Separation appears only as chain lengths above C12 enter, and it grows with their
  spread: 0% at C6, 0.38% at C14, 2556% at C20+unsat. Identifiability here emerges with system
  scale rather than with data volume, and a point-estimate fit can't report it.
- **Expected results, both directions:** a handful of tightly constrained parameters against a
  majority near the prior is itself the expected, informative result. The analysis identifies
  *why* the unconstrained ones are unconstrained (correlated with something else, or genuinely
  diffuse), which feeds Section 3.6. Uniformly tight shrinkage would suggest the datasets are
  more informative than previously appreciated.

**Tier 1:**
- Figs 6 and 6b are computed from the main three-parameter fit (Fig 2), with no extra fits.
- **Known-answer control:** `d1`+`d2` fit on C8, where they are provably inseparable, and on
  C14+unsat, where separation begins. The shrinkage diagnostic must flag both parameters
  individually on C8. The eigen-analysis must find one tight combination (`12*d1 + d2`) and
  one flat direction. If either fails, the diagnostic isn't trustworthy for real-data claims.
  The same C8 data, fit with a diagonal and a dense mass matrix, supplies the SI's
  mass-matrix test.

**Tier 2:** Figs 6 and 6b recomputed from the Tier-2 fit.

### 3.4 Target-enzyme identification: posterior-integrated sensitivity vs. Morris

> **In one sentence:** does accounting for parameter uncertainty change which enzymes the
> model says to engineer? This is the most directly consequential comparison in the paper.

- Compared directly against Ruppe et al. 2020 Fig. 3E: does a sensitivity analysis
  integrated over the posterior identify the same influential enzymes, on the same three
  objectives (total production, unsaturated fraction, average chain length), as the original
  Morris screen at the single fitted point? Or does parameter uncertainty shift the ranking?
- **Distinct from the screen in 2.3:** that one varies *kinetic parameters* to decide what is
  worth inferring; this one varies *enzyme concentrations* to decide what to engineer. It's
  the same method with a different input space and a different question, and the writing
  should keep them apart.
- **Writing note:** frame this as a controlled comparison against the *original study's own
  method*, not a claim that Morris is the best global sensitivity method available. That
  pre-empts the reviewer question "why not Sobol".
- **Figure 7:** side-by-side ranking, Morris at the point estimate vs. posterior-integrated
  sensitivity, for the same nine enzymes and three objectives.
- **Where rankings disagree:** at Tier 1 the truth settles which ranking is correct. At Tier 2
  a held-out-data check decides which ranking the data support.
- **Expected results, both directions:** agreement shows the original targets are robust
  across the full range of parameter values the data support, a stronger statement than a
  point screen can make. Disagreement bounds how much uncertainty it takes before target
  selection from a single point diverges from what the data support. That is the quantity
  deciding whether uncertainty analysis is optional or necessary in a less-characterized
  system.

**Tier 1:** Morris over enzyme concentrations, repeated across draws from the Fig 2
posterior. This needs forward solves only, no new fits. C14+unsat engages FabA/FabB, so all
three objectives are meaningful.

**Tier 2:** the same analysis over the Tier-2 posterior (primary evidence).

### 3.5 Robustness of the model-guided ratiometric strategy

> **In one sentence:** collaborators already use the FabF/FabB-vs-TesA ratio heuristic in the
> lab; this asks whether that recommendation holds across every parameter set the data find
> plausible, or only at the single point it was derived from.

- Mains et al. 2022 predicted that increasing FabF/FabB while decreasing TesA shifts
  production toward longer chains (and vice versa), and validated this in vivo. Does that
  direction hold consistently across the posterior, or could an equally plausible parameter
  set predict a different or reversed relationship?
- **Figure 8:** predicted chain-length response to the FabF/FabB:TesA ratio across posterior
  draws. A consistent direction and magnitude establishes the strategy as robust; spread
  across draws quantifies how much of the confidence rests on loosely constrained
  parameters.
- **Concretely:** re-run the optimisation that predicts high- and low-chain-length
  concentration profiles for each posterior draw, and compare the resulting ratio
  distribution against the originally reported point ratios.
- **Expected results, both directions:** a consistent directional prediction puts a
  quantitative confidence bound on an experimentally confirmed engineering result.
  Directional disagreement among draws would mark this as a prediction type that needs
  uncertainty analysis before it's relied on in a system without in vivo confirmation.

**Tier 1:** forward solves over the Fig 2 posterior. A truncated model spans a narrower
chain-length range than the in vivo demonstration (ME1 Fig. 5A/5B), so expect at most an
attenuated version of the shift. This is a cheap check that the direction is visible.

**Tier 2:** full model, real data. This is the section's primary evidence.

### 3.6 Designing maximally informative experiments: what data matters most

> **In one sentence:** if a collaborator can afford one more experiment on a new system, this
> says which one. It is the paper's most directly reusable deliverable.

- Which measurement types most efficiently tighten the posterior, crossed two ways:
  - **Timing:** final (endpoint) profiles only vs. initial rates only vs. both.
  - **What is measured:** total fatty acid only vs. individual fatty-acid species vs.
    individual species plus acyl-ACP pathway intermediates.
- **Figure 9:** posterior contraction across the timing × measurement-type grid, ranked by
  shrinkage gained *per data point* rather than raw shrinkage, so the ranking reflects value
  for collection cost.
- **Expected results, both directions:** if intermediates or initial rates give
  disproportionate gains relative to their (presumably higher) cost, that is the paper's most
  useful guidance for collaborators. If cheap total-FA endpoint data is nearly as informative,
  that is equally valuable: it says existing low-cost data collection is likely sufficient for
  a new system.
- **Comparison metric:** not LOO, which is invalid across differently scoped datasets.
  Posterior contraction toward the known synthetic truth is used instead.

**Tier 1 (C14+unsat):**
- The expected information (Fisher-based posterior contraction) is computed for every cell of
  the grid on the Fig 2 model, with no sampling. This draws the full figure.
- Three cells are then sampled (`a1`+`c3`) to confirm that the information ranking matches
  real posterior shrinkage: the cheapest data type, the best cell per data point, and the
  full dataset.
- The initial-rate observable already exists. The intermediates cells need an acyl-ACP
  observable module (zero extra solve cost, since the full state trajectory is already
  computed).

**Tier 2:** only the cells the real data allow (it has initial rates, profiles and one time
course, but no intermediates). If Tier 1 shows that the information analysis predicts
sampled shrinkage, Tier 2 uses the information analysis alone.

## 4. Conclusion **[not yet started]**
- On a system where the answers are largely known, Bayesian inference recovers the previous
  point-estimate results. It also delivers what a point-estimate fit structurally cannot:
  - honest, calibration-checked uncertainty
  - a quantitative, LOO-backed test of the grouping assumptions
  - a genuine identifiability check, including correlations and whether distinct solution
    sets exist
  - an uncertainty-aware alternative to Morris-based target identification
  - a robustness check on the ratiometric design strategy already in engineering use
  - guidance on which data types to prioritise when characterizing the next system
- Directly answering the three central questions:
  - **Does it work?** It recovers the previous model's solution from synthetic data, with
    calibrated uncertainty that degrades appropriately under noise and prior
    misspecification.
  - **What can it do?** Identifiability, correlation, multimodality and data-informativeness
    results that collaborators can act on immediately.
  - **What did it find?** **[state plainly whether the Bayesian analysis confirmed or revised
    the previous model's groupings, targets and ratiometric strategy]**.
- Minimisation plus local sensitivity plus point optimisation is standard practice across
  metabolic engineering, so this validated workflow is offered as a reusable alternative via
  the linked repository.
- It establishes the trust needed to apply the same, unmodified method to a system without
  known answers in a follow-up paper.

## SI
Methodological-validity checks, framed as "why this choice doesn't undermine the posterior",
never as speed claims.

- **Simulation-based calibration on the reduced-scale synthetic system** (Talts et al.
  2018): the umbrella for Tier 1's validity checks (Fig 3).
- **Full MCMC convergence diagnostics** for every reported run: r-hat/ESS trajectories and
  divergence counts.
- **Rank-normalised ECDF mixing check, and why the default plotted envelope isn't a valid
  pass/fail test:** the default pointwise-style band compared against a
  simulation-calibrated simultaneous band. It's a general point, relevant beyond this paper.
- **Stranded-chain detection via log-posterior gap** *(done)*. r-hat and ESS can pass while
  one or more chains sit in a much lower-posterior basin. Tested directly, BFMI does not catch
  this either: a chain hundreds of nats below its siblings had a BFMI in the healthy range.
  The mean log-posterior gap between chains does catch it (~100 µs per call). It was validated
  on two real cases: one rogue chain 533 nats below its siblings, and a 4-of-8 split 6,975
  nats apart. A posterior-predictive check split by chain group shows the excluded mode
  visibly failing to fit the data. The known-answer toy (3.1) checks the converse: two
  genuinely equal-mass modes must *not* be excluded, and must be flagged by r-hat instead.
- **Chain-count diagnostic power** *(done)*: how many chains are needed before r-hat and
  rank-ECDF have enough resolution to be trusted. It is measured by splitting a 64-chain run
  into disjoint 4/8/16/32-chain groups and comparing the diagnostics across groups. This is a
  correctness question, not a speed benchmark.
- **ODE solver tolerance robustness:** one Tier-1 fit repeated at a tighter `rtol`, showing
  the posterior is materially unchanged. Section 2.2's selection procedure supplies the
  supporting numbers.
- **`target_accept` / divergence check:** divergence counts at the chosen setting, and
  confirmation that results don't shift at 0.95.
- **Mass-matrix structure (diagonal vs. dense):** this bears on 3.2/3.3's correlation claims,
  since the default sampler assumes no correlation. It is resolved with the `d1`+`d2`
  known-answer pair (3.3) fit both ways. The preliminary evidence is the `a1`+`c2` pilot
  (3.1), which the diagonal matrix can't sample and a dense one doesn't rescue in 12 h.
- **Prior width and form:** the prior is wide enough not to presuppose the answer. The
  evidence is the ladder's shrinkage (3.1) and the prior-misspecification curve (Fig 4).
