# Efficiency logic map: "I put this model in Python, now I want efficient Bayesian inference"

Purpose: trace the entire chain of reasoning from that starting premise to where the
project currently stands, to where it needs to get to. Companion to
`full_audit_request_v3_annotated.md` (that file tracks the audit process and its open
questions; this file is the actual content map it was asked to produce for the
efficiency thread specifically). Built 2026-09-09 from this session's work plus
project memory — sections marked **[gap]** are genuinely unaddressed, not just
unwritten.

---

## 0. The premise

Python Model exists to do one thing the MATLAB model and the early Restructured
Framework pipeline couldn't: run Bayesian inference on the FAS kinetic model fast
enough to be practical. Every decision below is either (a) something that had to be
true for inference to run *at all* on this hardware/queue, or (b) something chosen to
make it faster without breaking correctness. The two are hard to fully separate in
practice — several "efficiency" decisions (checkpointing, the lp-gap fix) turned out
to be load-bearing for correctness too.

---

## 1. Foundational choices (settled, not revisited this session)

**Inference engine: PyMC front-end + BlackJAX NUTS, not PyMC's own sampler or nutpie.**
Chosen specifically for checkpointing control — Blanca's preemptable QOS can kill and
requeue a job mid-run, and neither PyMC's built-in sampler nor nutpie exposes the
low-level state needed to resume a NUTS chain mid-warmup or mid-sampling. Without
this, a multi-day run on this cluster's queue policy would likely never complete at
all, regardless of how fast any single step is. This choice underlies everything
after it — it's the reason a "resumable sampler" module exists as its own thing.

**ODE solving: JAX + diffrax, Kvaerno5 implicit solver, PIDController adaptive
step-size.** Chosen for differentiability (NUTS needs gradients through the ODE
solve) and to handle a stiff kinetic system. Correctness-driven, but every later
efficiency question (floor/no-floor, tolerances, startup cost) is downstream of this
solver choice.

**float32 tried and rejected.** Single precision breaks the ODE forward solve
outright (the custom_vjp TypeError that results is a masked exception, not a clean
failure) — this is closed; don't revisit without first fixing whatever makes the
forward solve numerically fragile at that precision. → memory `float32-not-viable`.

---

## 2. Building a scaling axis that actually means something

The first real efficiency-methodology decision: what varies along the test ladder?
An earlier enzyme-count ladder was **78-81% dead species** — testing against it
would have measured almost nothing real, and it's what caused an earlier "floor
inversion" result that made no sense until this was caught. Replaced with the
**chain-length ladder** (C4_NoFB through C20+unsat, 56 to 586 reactions) as the
actual scaling axis. Every GPU-normalization fit, every startup-cost fit, and this
whole session's floor/no-floor comparison depend on this axis being real — it is,
now, but it's worth remembering *why* enzyme count was rejected if the model ever
grows a new dimension to scale along.

---

## 3. Choosing what to actually benchmark (`Notes/benchmark_test_case_decisions.md`)

Before any speed comparison means anything, you need a defensible "which system,
which parameter, which data, which prior, which sampler settings" baseline. This was
done rigorously, by computed metric rather than intuition, and is worth restating
because it's the methodological foundation everything else in this document sits on:

- **System: C8**, not the smaller C4_NoFB — because cost-per-*converged-run* (not
  cost-per-draw) is what matters, and C4_NoFB's single observable can't separate a1
  from a global rate rescaling, making it need *more* draws and *more* leapfrog steps
  despite being the smallest system. Known limitation carried forward: C8 doesn't
  exercise FabF/FabB chain-length specificity (those constants don't diverge until
  C10+), so anything testing that needs a bigger, more expensive system.
- **Parameter: a1**, not c2 (strongest grouping test) or c3 (highest influence) —
  chosen for the *lowest interaction ratio* of the three candidates (so
  single-parameter results are more likely to survive into joint inference later),
  strong identification (82% of the top parameter's influence), and continuity with
  already-validated posteriors.
- **Data: maximize time points (they're free — measured, not assumed: 1100.8ms vs
  1102.2ms per gradient with/without), minimize+order conditions cheapest-first**
  (the real cost driver, ~14% faster at matched count).
- **Prior width: profile log-likelihood, calibrated delta=2.23** — not a step-count
  ceiling (silent about informativeness) or a fixed sensitivity threshold (too
  conservative by ~sqrt(n), would have wrongly truncated 2/3 of a1's real posterior
  mass on C4_NoFB). This is the ancestor of tonight's "tightest" [0.1,10] and
  "narrowest" [0.05,20] widths.
- **Sampler settings held fixed** across the whole grid (chains=8, target_accept=0.8,
  checkpoint_every=5, rhat_check_every=100, 2 consecutive checks, 1 post-convergence
  check) so later comparisons aren't confounded by an incidental settings change.
- **Standing observation, never acted on:** ESS and the rank-ECDF check are
  *currently inert* — r-hat is the binding criterion in every completed run, ESS
  clears its floor well before r-hat does, and rank-ECDF has never bound. This is
  slack that could be traded for speed (relax the ESS floor, check rank-ECDF less
  often) if convergence-check overhead ever becomes worth shaving — **[suggestion,
  not yet pursued]**.

---

## 4. Making the cluster infrastructure itself efficient and reliable

Efficiency isn't just per-step speed — a fast sampler that silently loses progress to
preemption, or burns wall-clock on a job that will never converge, is not efficient
in the way that matters (results per wall-clock-day). This layer of work:

- **Resumable checkpointing** (`progress_log.jsonl`, `checkpoint_meta.json`,
  `draws.zarr`) makes the preemptable QOS usable at all instead of a liability.
- **Predictive, not retrospective, time-budget guard** — checks whether the *next*
  chunk will fit before starting it, not whether the budget is already spent;
  retrospective checking was measured to lose up to 45 minutes of unsaved work per
  overrun at C10.
- **Dead-chain detection with patience=2** — distinguishes "genuinely dead" (0%
  acceptance, 100% divergence) from "merely slow," calibrated against ~1100
  observations where no healthy run ever produced even one fully-dead checkpoint.
  Without this, a broken run (like the one that burned 24h on C10 earlier in the
  project) would silently keep re-queueing itself forever.
- **GPU hardware normalization**, cross-validated three independent ways (dead-run
  cost law, direct overlap, shared-stage-curve regression) rather than trusted from
  one method — necessary because Blanca's shared pool silently mixes A100/H100
  NVL/H100 PCIe MIG/V100, and a bare `--gres=gpu:1` request can be satisfied by a MIG
  slice that isn't a whole card. Fixed by pinning `--gres=gpu:a100:1` +
  `--constraint=A100` everywhere.
- **Self-resubmission**, replacing an earlier pre-chained `--dependency=afterany`
  pattern that queued an unconditional successor even for runs that had already
  converged. **Hardened tonight (2026-09-09):** the TERM-trap-only approach lost 3 of
  9 jobs to a real race (SLURM's SIGKILL beating bash's trap to running); fixed by
  having whatever cancels a job submit its successor directly
  (`cancel_and_resubmit.sh`), with an atomic per-segment lock so the old trap (kept
  as a fallback for genuine unattended preemption) can't double-submit.
- **24h total-run-time cap, added tonight, in two independent places** (the bash
  resubmit gate and the sampler's own loop) so a slow system stops for human review
  instead of silently re-queueing for days. **This is deliberately the same
  mechanism you'll reuse for the eventual full-model production run's multi-week
  budget** — `SamplerSpec.max_total_hours`, settable per-run via
  `solver_params.json`.
- **[Gap, discovered tonight, not yet addressed]:** the account's effective
  concurrent-A100 ceiling on Blanca's shared preemptable pool is empirically ~9 jobs
  right now (3 more sat PENDING on Priority/Resources behind the running 9). This
  doesn't affect any single run's efficiency, but it directly affects **wall-clock
  time to a confident ladder-wide answer** — the near-term success criterion. Left
  to resolve naturally tonight per your call; worth a real decision later about
  whether to just accept the throttling or actively manage which systems get
  priority.

---

## 5. Making the convergence diagnostic itself trustworthy

A speed comparison is worthless if "converged" doesn't reliably mean converged. This
turned out to be a real gap, found *because of* the floor/no-floor speed work, not
independently of it:

- Standard r-hat + ESS: not sufficient alone — a single passing r-hat/ESS check is
  fragile (C8 passed at draw 100, failed at 150 and 300, passed again at 450), so a
  **consecutive-passes streak** requirement was added, plus a **rank-ECDF mixing
  check** as a third criterion.
- **BFMI does not catch a stranded/rogue chain** — tested directly: a chain stuck
  533 nats below its siblings still had a BFMI (1.155) inside the healthy range
  (1.005-1.309). BFMI measures local energy-exploration quality, not where the
  posterior mass actually is.
- **Log-posterior gap is the correct, cheap discriminator** (~96µs/call vs.
  10-100ms for one arviz r-hat call) — implemented, unit-tested against real data,
  and deployed to the cluster tonight. Validated live on two structurally different
  cases: C8 (1 of 8 chains stranded, 533-nat gap) and C10-narrowest (4 of 8 chains,
  6975-nat gap — a much larger and more prevalent split).
- **This session's newest finding: the two "stranded" cases are very likely
  numerical artifacts, not real multimodality** — per your domain confirmation that
  a1 alone shouldn't have two valid solutions without another parameter also
  changing, and confirmed independently by data: C8's *floor* equivalent converges
  perfectly cleanly with zero such mode (lp gap 0.2 nats), while only *no-floor*
  produces the secondary mode. This points at no-floor's negative-concentration
  tolerance as the likely mechanism. See `a1_should_not_bifurcate` in memory and the
  open-questions section of the audit-request file for what's still unresolved here
  (C10-floor's own unrelated warmup instability couldn't be checked the same way; the
  paper-documentation question is still open).
- **Visually confirmed via posterior-predictive checks (2026-09-09) that a1≈9.4
  (C10-narrowest's stranded mode) is a genuinely bad fit, not a plausible
  alternative:** split the finalized netcdf by chain group and ran
  `plot_predictive` on each. The kept group (a1≈1.0) matches the observed FA
  time-courses almost exactly. The stranded group (a1≈9.4) systematically
  *underpredicts* — most dramatically on C10_FA/Total_FA, where by t≈150s the
  data reaches ~6 µM and the stranded-only prediction is still under 1 µM. This
  is the intuitive picture behind the 6975-nat lp gap, not just a number.
- **FIXED, same session: the exclusion wasn't propagating to finalized
  artifacts.** The live sampler already excludes a stranded chain from its own
  r-hat/ESS decision, but `_finalize_resumable_run()` (in `inference_runner.py`,
  called both by the live auto-finalize path and by `finalize_window.py`) was
  still building `posterior_predictive`/log-likelihood/summary metrics from
  *all* chains regardless — confirmed visually: the all-8-chains
  posterior-predictive band for C10-narrowest was measurably wider and biased
  low relative to the kept-chains-only version. Fixed by threading
  `status.json`'s `stranded_chains` list through both call sites; the C8
  (tightest, no-floor) and C10 (narrowest, no-floor) netcdfs were regenerated
  under the fix. `finalize_window.py` excludes by default now — pass
  `--include_stranded` to deliberately see the unfiltered version.
- **[Gap, flagged in section 7 below]:** this whole diagnostic — and the "exclude
  anything >20 nats below the best chain" default it's built around — was designed
  and validated entirely in a single-free-parameter world. It is not yet clear it's
  safe to apply unmodified once a second free parameter is added, where a real
  competing solution becomes physically plausible per your own stated rule.

---

## 6. The floor-vs-no-floor investigation (this session's central efficiency result)

**Question:** does clamping negative ODE concentrations to zero cost speed, relative
to letting the solver see negatives and crash-detect via `EQX_ON_ERROR=nan`?

**Headline result:** under the actual production prior (tightest, [0.1,10]), no-floor
is safe and up to **~9x faster on C10** than floor. This *reverses* an earlier
verdict — under the old, wide [0.001,1000] prior, floor was genuinely needed to
prevent crashes. That's a clean example of a decision that was correct in its
original context and became wrong once the prior was tightened, caught by this
audit rather than carried forward silently.

**CORRECTED 2026-09-09, superseding what this section said earlier tonight.** The
"C6 exception is a competing-power-laws crossover" claim (no-floor startup steep at
`~rxns^2.2-2.8`, floor startup flat at `~rxns^0.6-0.9`, floor per-step cost steep at
`~rxns^2.87` against no-floor's `~rxns^0.83`) does not hold up. Checked directly
against `speedup_analysis.ipynb`'s own live `startup_cost()`/`cost_per_solve()`
functions, not a re-derivation, and both halves were wrong:

- Floor has **no live cost-per-solve data at all past C8** — it's dead everywhere
  above that (see the ladder leaderboard below), so there is no trend to fit a
  `~rxns^2.87` exponent to. That number traced back to a background analysis run
  before this notebook's dead-checkpoint filtering existed, which silently averaged
  failed-attempt seconds into what looked like a cost curve — the notebook's own
  intro cell already warns about exactly this failure mode.
- Floor's startup cost is **larger than no-floor's at nearly every system**, not
  smaller — direct measurement, not a fit. The opposite claim traced to the same
  kind of stale, pre-filtering background analysis.

**What actually explains "floor wins on C6" remains an open mechanistic question.**
What's now confirmed directly: at the three systems where floor is even alive
(C4_NoFB/C6/C8), its per-solve cost is competitive-to-cheaper than no-floor's, and
its startup cost is comparable-to-worse — neither cleanly explains a C6-specific win
in total sampling time (5.9 vs 8.4 s/draw). The C8→C10 transition is best described
as a **cliff** (floor works, at roughly comparable cost, right up until it doesn't —
0% acceptance, frozen position, confirmed directly via `accept`/`n_steps` arrays),
not a gradual scaling crossover. `speedup_analysis.ipynb` now shows this directly as
measured bars (per-solve cost, startup cost) rather than fitted/extrapolated curves.
If the mechanism behind the C6 result specifically still matters for the paper, it
needs new investigation, not a revival of either retracted exponent.

**Side effect, not the original goal:** this comparison is what surfaced the
stranded-chain diagnostic gap (section 5) — a reminder that a rigorous efficiency
comparison forces close-enough inspection of individual chains that correctness bugs
surface as a byproduct.

**[Open, from the audit-request file's questions]:** whether to document no-floor's
spurious-secondary-mode behavior as a known limitation in the paper, given it's
otherwise the recommended faster choice.

**Second notebook-methodology correction, same session:** the "ladder leaderboard"
figure originally drew a hatched bar for any non-converged run, sized to however much
compute it happened to log before being cancelled or found dead — a number with no
meaning (it's "when I killed it," not a measurement of anything), but a bar's length
reads as data regardless of intent. Fixed to a plain "did not converge" label pinned
to the axis instead, for both conditions. Worth checking any other figure that shows
a not-yet-finished run's elapsed time as if it were a completion time before trusting
it — this is the second time a plot has silently implied a number that wasn't real
(the first being the retracted crossover fit above).

---

## 7. Where we currently are (snapshot, 2026-09-09 ~01:30)

- **Tightest/no-floor** (production prior): C4_NoFB, C6, C12 converged; C8 finalizing
  (~3000 sampling draws); C10 just resubmitted after the stranded-chain fix (queued,
  behind the account's current A100 ceiling); C12+unsat running normally; C14 through
  C20+unsat all still mid-warmup, with per-step rates ranging from healthy
  (C14+unsat ~17s/step) to unresolved-and-concerning (C20+unsat pre-restart, 1341s/step
  — not yet re-measured cleanly post-fix, see the audit-request file's open question
  about whether this is a floor/no-floor issue or a separate solver-settings issue).
- **Narrowest/no-floor** (secondary prior width): **far less complete than tightest —
  only C4_NoFB and C10 have been run at all**, both now converged (C10 at exactly
  1205 sampling draws, matching its stranded-chain-fix finalization). The other 12
  systems have no narrowest-width data yet. **This wasn't flagged before — worth
  deciding whether narrowest needs the full ladder too, or whether tightest alone is
  sufficient for the production decision** (the profile-likelihood prior-width
  methodology in section 3 would suggest tightest, calibrated the same way, is
  probably the one that matters).
- Confirmed tonight: all 6 of the large systems currently running are correctly on
  the no-floor sbatch script — no confound there.
- Both infra fixes (resubmit-race, 24h cap) are deployed and apply to future
  segments; the currently-executing 9 jobs are running under whatever code they
  loaded at their own start until their next resubmission.

---

## 8. Where we want to be — and the gap nobody has named yet

**Long-term success criterion:** the full model (586 reactions, "a handful of
parameters," not just a1) converges on a single A100 in under 2 weeks.

**Every efficiency result in sections 3-6 above was produced with exactly one free
parameter.** That is the single biggest unaddressed gap between current work and the
stated goal, and it touches nearly everything already decided:

1. **Mass matrix.** `SamplerSpec.is_mass_matrix_diagonal=True` (cheaper) is
   trivially correct with one parameter — a 1x1 matrix has no off-diagonal to miss.
   With several correlated parameters, a diagonal approximation can badly mis-adapt
   step size and direction, inflating leapfrog-step counts or blocking convergence
   outright. Whether diagonal remains adequate, or a dense mass matrix becomes
   necessary (more expensive per adaptation step, but possibly required), is
   completely untested.
2. **Convergence criteria at multiple parameters.** More free parameters is more
   simultaneous ways to fail r-hat/ESS/rank-ECDF — likely slower wall-clock
   convergence even if per-step cost didn't change at all, which it will.
3. **The lp-gap stranded-chain fix's core assumption may not transfer.** Your own
   rule — a1 alone can't have two valid solutions without another parameter also
   changing — implies that *with* a second free parameter, a real competing
   solution becomes physically plausible. The current default (exclude anything
   >20 nats below the best chain) was validated entirely in the regime where that
   couldn't happen. Applying it unmodified to a multi-parameter run risks discarding
   a real second mode instead of an artifact.
4. **The floor/no-floor conclusion may not transfer.** It was derived from
   single-parameter exploration of a1's range; a multi-parameter run explores
   concentration-space very differently (more directions to wander negative, or
   conversely more constraints holding trajectories near nominal) — "no-floor is up
   to 9x faster" is not guaranteed to hold once more parameters are free.
5. **The prior-width methodology (profile log-likelihood, delta=2.23) was calibrated
   per-parameter.** A multi-parameter prior needs either independent per-parameter
   profiles (assumes weak interaction — already contradicted by a1's own multi-
   constant grouping structure) or a genuinely joint/multivariate calibration that
   doesn't exist yet.
6. **The throughput numbers informing "is 2 weeks realistic" are all
   single-parameter.** Multi-parameter HMC is essentially certain to need more
   leapfrog steps per iteration at a given target_accept (higher-dimensional
   gradient, harder geometry) — current numbers, even the healthy ones, cannot be
   extrapolated to the goal without a real multi-parameter measurement.
7. **The already-concerning large-system throughput (C16+unsat...C20+unsat) is a
   floor under item 6, not a ceiling** — whatever multi-parameter slowdown gets
   layered on top starts from numbers that are already the least-understood and
   least-healthy in the whole ladder.

**Suggestion, not yet acted on:** run a small, deliberate multi-parameter pilot
(e.g. a1 plus one more well-identified, weakly-correlated parameter — c3 was
flagged in section 3 as highest-influence with the weakest grouping test, or the
same low-interaction-ratio logic that picked a1 could pick its next-best partner) on
a small-to-mid system (C8 or C10) *in parallel with* finishing the single-parameter
ladder, rather than sequentially after it. Every week spent only refining the
single-parameter regime is a week not de-risking the actual multi-parameter target,
and items 1-6 above are all things a small pilot would start answering empirically
rather than by further extrapolation.

**Two smaller, cheaper suggestions surfaced along the way, also not yet acted on:**
- A shorter fixed warmup (`n_tune` 600-700 instead of 1000) was identified earlier
  this session as a safer partial win than n_tune=400 (which cuts mid-mass-matrix-
  window under BlackJAX's fixed schedule) — never actually tested. Cheap to try on
  the existing single-parameter ladder before warmup cost becomes even more relevant
  in the multi-parameter regime.
- Section 3's observation that ESS/rank-ECDF are currently inert is free slack that
  could be traded for a small speed gain if convergence-check overhead becomes worth
  shaving.

---

## Suggested next steps (mine to propose, yours to prioritize)

1. Decide whether to start a multi-parameter pilot now, in parallel with the
   single-parameter ladder, given section 8's gap is the largest unaddressed risk to
   the long-term goal.
2. Decide whether the narrowest prior width needs the full 14-system ladder too
   (section 7), or whether tightest alone should be the production decision basis.
3. Resolve C20+unsat's throughput cause (floor/no-floor vs. solver-settings vs.
   both) — already flagged in the audit-request file's open questions, restated here
   because it's also the floor under every multi-parameter throughput question.
4. When bandwidth allows: the two cheap experiments (shorter fixed warmup; relaxed
   ESS/rank-ECDF cadence) are low-risk, low-cost, and could be run on the existing
   single-parameter infrastructure without waiting on anything above.

---

## 9. The multi-parameter pilot: implemented 2026-09-09, not yet submitted

Item 1's decision came back "yes" — full plan at
`~/.claude/plans/i-want-to-test-twinkling-kernighan.md`. Built, not yet run
(explicitly held back from `sbatch` pending your review):

- **`ess_threshold`** added to `SamplerSpec`/`inference_runner.py` (was a hardcoded
  `>=400` literal). `None` drops the ESS requirement, leaving r-hat as the sole
  criterion. Unit-verified default-preserving and correctly permissive when `None`.
- **Prior width for c2/c3, corrected mid-implementation**: the plan's original
  "Stage 1" (reuse `info_vs_conditions.py`'s `delta=2.23` profile as the new prior)
  was wrong — that script approximates the *posterior*, not the prior (its own
  docstring says so; C8's profile is `[0.9798,1.0197]` against a1's actual prior
  `[0.1,10]`, ~115x wider). Traced the real precedent instead: `Outline_v2.md`
  section 2.3's Morris screen already uses a 100-fold `[0.1,10]` window for every
  scaling parameter, c2/c3 included — identical to a1's own prior. **c2 and c3 get
  `[0.1,10]`, matching a1, not a freshly-derived bound.**
- **Stage 2 (joint feasibility, `job_files/bench/joint_feasibility.py`, new)**:
  Latin-hypercube (200 draws) forward-solve check over the joint `[0.1,10]×[0.1,10]`
  box under no-floor. Sanity-checked on a known-safe single-parameter box first
  (0/50 failures). Real checks (a1+c2, a1+c3 × C6/C10/C14): C6 clean on both pairs
  (0/200 failures each); C10/C14 still running as of this writing.
- **`build_multiparam_config.py`** (new) clones a system's existing no-floor config,
  swapping in both free parameters, `tune=600`, `ess_threshold=null`, rank-ECDF
  removed.
- **`multiparam_nofloor.sbatch`** (new) — required extending `resubmit_if_needed.sh`
  to recognize the new `*_no_floor` naming as a complete label rather than
  double-wrapping it in the old `a1 $WIDTH nofloor-eqxnan` template.
- **Side effect of this work**: found and fixed two live infra bugs unrelated to
  the pilot itself, both in `regen_finished_plots.sbatch` (re-finalizing
  C8/C10-narrowest under the stranded-chain exclusion fix, then generating all 3
  standard plots for every converged run). First, a broken conda activation had
  spent 99 minutes running GPU-designed work entirely on CPU, confirmed via
  `nvidia-smi` showing its PID absent from the GPU's compute-apps list — killed,
  fixed to the working `module load miniforge; mamba activate Bayesian` pattern,
  resubmitted (28214704). Second, once that fix let it actually run, it failed on
  *every single system* with `FileNotFoundError`: it hardcoded the pre-rename
  directory suffixes (`"a1 tightest nofloor-eqxnan"` etc.) that Phase 0's renaming
  had already moved. Fixed by giving it the same new-name-first/old-name-fallback
  `resolve_suffix()` pattern used elsewhere, verified against 4 real paths
  (including one, C18, that resolved to the *old* name correctly — converged but
  not yet renamed since nothing re-ran the rename step after it finished), and
  added C18 itself to the run list (missing because it converged after the list
  was written). Resubmitted as 28214750.
- **`warmup_status.py` performance**: profiled at 50s for a full run, 41s of it 34
  separate per-system subprocess spawns of the Bayesian conda env's python just to
  read one small zarr array each, plus 6 redundant `squeue` calls (the query
  ignores its jobids argument and always returns the full job list, so 6 sets meant
  6 identical calls) and 6 separate `sacct` round trips. Batched all three into one
  call each, shared across every set shown instead of refetched per set/system:
  6.8s for the same full run, output-diffed against the pre-fix run to confirm no
  regression (every difference was real progress made in the few minutes between
  the two runs).

**Also: results-folder cleanup executed** (Phase 0 of the same plan) — deleted 19
superseded folders using an explicitly-rejected wide prior that never really ran,
archived ~31 more (an old `[0.01,100]` rejected-prior attempt with real progress, an
early attempt at the outline's actual final settings mistakenly sharing the
"narrowest" name, and the abandoned `a2` parameter exploration), and renamed the 4
canonical sets to prior-value-based names (`a1_0.1-10_floor` etc.) for every
converged/dead system -- still-running systems keep their old name until they
finish. Required making `warmup_status.py` and both export scripts resolve each
system's directory per-system rather than once per set, since a set can now span
both naming generations at once.

**`warmup_status.py` also redesigned** the same session: replaced the 8
milestone-timing columns (W5...S1000 -- genuinely useful for the notebook's analysis,
mostly noise for "how is my job doing right now") with live JOBID, NODE/reason (one
squeue call, not a separate check), R-HAT/ESS and STRANDED count parsed straight from
each job's own logged convergence checks. Independently confirmed correct: it now
shows C8's 1/8 and C10-narrowest's 4/8 stranded chains without any extra lookup,
matching the manual investigation exactly.

**`job_files/bench/multiparam_analysis.ipynb`** (new) covers the four questions the
plan's "What gets checked once these run" section named: cost scaling with parameter
count, mass-matrix adequacy (step-size/acceptance distributions, via the
`step_size` field `export_posterior_series.py` now captures), convergence
draw-count under r-hat-only vs. the 1-parameter baseline (with the explicit caveat
that the two use different stopping criteria), and stranded-chain behavior with a
second free parameter -- flagged there as needing judgment rather than an
assumed-artifact default, since a real second joint solution becomes physically
plausible once `a1` isn't the only thing free to vary. Required generalizing
`export_posterior_series.py` from a hardcoded `a1` to whatever `<name>_log__`
parameters actually exist in a run's zarr store (`params: {name: [...]}` dict, `a1`
kept as a top-level key for backward compatibility with `speedup_analysis.ipynb`).
Executed end-to-end against real cluster data (0 errors): the 1-parameter baseline
(`tightest_nofloor`, C6/C10/C14) populates every table and plot with real numbers;
the two pilot sets (`a1c2_no_floor`, `a1c3_no_floor`) correctly show as
"(none yet)"/`NaN` throughout, since no pilot job has been submitted -- the notebook
was written to degrade gracefully rather than error on the empty side of that
comparison, and will pick up real pilot numbers automatically once something is
submitted and checkpoints.
