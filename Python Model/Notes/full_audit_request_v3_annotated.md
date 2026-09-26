# Full retrospective audit — updated request, annotated (prepared 2026-09-09)

**Status: resumed and actively in progress** as of the 2026-09-09 late-night session.
This file exists so that whenever work picks back up — same session or a new one —
it starts warm instead of cold. It is the rewritten audit prompt (verbatim, below)
plus everything known that bears on each item. Read this whole file first.

---

**See also `Notes/efficiency_logic_map.md`** — the full "Python model → efficient
Bayesian inference → current state → target state" map, built 2026-09-09. It
restates and organizes everything below by theme rather than chronology, and names
the single biggest unaddressed gap: every efficiency result so far (floor/no-floor,
GPU factors, benchmark methodology, convergence diagnostics) was produced with
exactly one free parameter, and the long-term goal needs "a handful."

## OPEN QUESTIONS — answer these first, next session

Compiled 2026-09-09 ~01:40, when the user signed off for the night with "flag all
your questions in this session so I can answer them in the next but do what you can
on your own." Ordered by how much they block other work.

1. **Given your answer that a1 alone shouldn't have a second solution without
   another parameter also changing** — I checked C8's floor equivalent (see
   "Autonomous follow-up" below): it converges perfectly cleanly, all 8 chains at
   a1≈1.00, lp gap 0.2 nats. Only the *no-floor* C8 run shows the a1≈0.43 secondary
   mode. That's consistent with your answer: the secondary mode looks like a
   no-floor-specific numerical artifact, not a real kinetic alternative. **Do you
   want this written up as a known no-floor limitation** (spurious local minima from
   tolerating negative-concentration excursions) for the paper, given no-floor is
   otherwise the faster, recommended choice? And does that change how much you trust
   the stranded-chain-exclusion fix's *output* — i.e. is discarding those chains
   "correctly recovering the one true posterior" (safe to use downstream) rather
   than "correctly handling real multimodality" (would need more caution about what
   else it might be masking)?
2. **C10's floor equivalent couldn't be checked the same way** — both its floor runs
   (tightest and narrowest-matched) are still stuck in *warmup* after however many
   hours, with an enormous 17,000+ nat lp spread between chains (see below). Is that
   normal early-warmup noise before the mass matrix adapts, or does C10-floor look
   like it might not be converging at all? Worth a look once/if it reaches sampling.
3. **Scope correction, needs your call:** the artifact linked as "your first pass at
   this audit" turned out to be the workflow audit only — items 1-3 of the original
   request (MATLAB→Bernat→Linden→Restructured lineage timeline, Fox Group papers
   cross-check, `Outline_v2.md` gap-check) were never actually produced as a
   deliverable. Do you remember more of that work happening than I have access to
   (e.g. in an earlier session not covered by this file), or should I start those
   three items genuinely from scratch?
4. **Scope of "pause development" during the audit:** tonight's bug fixes (sampler
   stranded-chain exclusion, sbatch resubmit-race fix, 24h total-run-time cap,
   `run_model.ipynb`) all happened *during* the audit because they were blocking a
   trustworthy answer to the floor/no-floor question. The still-open Python Model
   cleanup items (A-01 env-spec pins, A-02 untracked files, B-03 stale cluster
   plotting code, B-04 no cluster git identity, B-05 four remaining notebook call
   sites, C-06 default mismatches, C-07 unused conda envs, D-08/09 backup/archive
   files, D-10/11 docs) are a different kind of thing — pure hygiene, not blocking
   the audit itself. Hold those until the audit concludes, or is fixing them (not
   just flagging them) fair game now too?
5. **C20+unsat's bad throughput** — is it a solver-settings problem (the paused plan
   at `~/.claude/plans/i-want-to-test-twinkling-kernighan.md`, about how its
   training/test conditions are generated) or a floor/no-floor problem (this
   thread), or both? Not yet investigated either way.
6. Once C16+unsat through C20+unsat have run clean for a few hours under the new
   infra fixes (not immediately post-restart), their throughput needs re-measuring
   against the 2-week/A100 long-term goal — tonight's numbers were noisy from the
   just-completed restart and aren't trustworthy yet either way.
7. **New tonight:** the "narrowest" [0.05,20] prior-width ladder is far less
   complete than "tightest" — only C4_NoFB and C10 have been run under it at all (12
   of 14 systems have zero narrowest data). Does narrowest need the full ladder too,
   or is tightest alone the production decision basis?
8. **The biggest structural gap, named in `Notes/efficiency_logic_map.md` section
   8:** every efficiency result so far (floor/no-floor, GPU factors, benchmark
   methodology, the lp-gap fix) was validated with exactly one free parameter (a1).
   The long-term goal needs "a handful." Worth starting a small multi-parameter
   pilot now, in parallel with the single-parameter ladder, rather than only after
   it finishes?

---

## Verbatim request

> I want to pause active development and do a full retrospective audit of this
> project's evolution — not add new code. Your job is to help me reconstruct and
> evaluate every major decision that got us from the original problem to where we are
> now, so I can be confident this is the strongest approach for my paper.
>
> Background / lineage of the work, in chronological order:
> 0. Original MATLAB model — the starting point, at: `.../Matlab Current Projects`.
>    This is the full model as it originally existed. The goal was a file-based,
>    flexible, automatic ODE builder that could eventually be shared for use on other
>    projects.
> 1. Bernat 2024 — a collaborator wrote this to help generalize the model into
>    something more flexible for new chemical systems than the MATLAB version. It
>    introduced building ODEs programmatically from YAML files instead of hardcoding
>    them.
> 2. Linden 2025 Simplified — I took a Bayesian inference framework for ODEs from
>    Linden's paper and stripped it down to the essentials I actually needed. This was
>    my starting point for doing Bayesian inference in Python at all.
> 3. Restructured framework (simple models) — I merged (1) and (2) into a working
>    pipeline: reaction files -> YAML-built ODEs -> Bayesian inference, running on a
>    simplified FAS model.
> 4. Python Model (this folder) — where I've been iterating since, mainly trying to
>    improve inference quality/performance. Contains `_staging_new_schema/`, a mockup
>    built using ideas from Tim Bernat that pulls FAS-specific templates (saturation,
>    chain length) out of the model into a more general, customizable framework.
>
> All four folders (1)-(4) are siblings under one git repo root ("Bayesian Kinetic
> Model"); folder 0 is a separate, older git repo.
>
> Two more reference sources, not code: the Fox Group papers at
> `.../Papers/Fox Group`, and the in-progress paper outline at
> `.../Outlines/Bayesian Framework/Outline_v2.md`. Both live outside this git repo —
> read them from their filesystem paths, not git history.
>
> What I need from you:
> 1. Reconstruct a timeline of what changed at each stage and why, from git
>    history/file structure of all five locations. Confirm or complicate the claim
>    that the original MATLAB model was too expensive — what made it expensive, and
>    how each later stage addressed that.
> 2. Read the Fox Group papers; cross-check which modeling choices in the current code
>    are carried over vs. new. Flag anything that contradicts or silently diverges
>    from published work without a clear reason.
> 3. Read the paper outline; check it against the current Python Model codebase —
>    does the code produce what the outline needs? Flag gaps both directions.
> 4. Ask directly wherever the reasoning isn't clear — don't guess.
> 5. Within Python Model specifically, flag redundant/dead/superseded/leftover code.
>    `_staging_new_schema/` is active and purposeful — evaluate on its own merits.
> 6. Flag anything across the whole lineage that's internally inconsistent,
>    undocumented, or a decision that doesn't hold up.
> 7. Produce one written summary mapping the full process end-to-end, documenting key
>    decisions/justifications, and listing cleanup items/inconsistencies/open
>    questions from steps 2-3 and 5-6. Go step by step, ask questions as you go — a
>    real dialogue, not everything saved for the end.
>
> --- STATUS UPDATE — please read before proceeding ---
>
> You already produced a first pass at this audit; the summary is here:
> https://claude.ai/code/artifact/8e2467b7-40d5-4ac8-afcf-bc77a6d48212?via=auto_preview
> You've also already gone through the other folders in the repo and the papers. I'm
> not sure how much of that is still in your working memory right now, so before
> continuing, tell me briefly what you currently recall/have loaded from that prior
> pass — I'd rather you say "I don't have X anymore" than silently re-derive it
> differently.
>
> From there, pick back up specifically on the efficiency-improvement thread: we had
> started tracking the logic behind making the model more efficient and had stepped
> out to investigate the floor tests, which is what's currently in progress. Continue
> that same logic-tracking-and-flagging approach on the floor tests specifically.
>
> Keep these two success criteria in view:
> - **Near-term:** confident that tier-1 tests run on the most efficient version of
>   the model possible.
> - **Long-term:** the full model needs to eventually run on a handful of parameters
>   and converge on an A100 GPU in under 2 weeks.
>
> Flag anything that puts either goal at risk, keep asking questions rather than
> assuming.

---

## What this session actually has loaded, honestly, per your own instruction to say "I don't have X" rather than guess

**Lineage/provenance (items 1, 2, 3, 5, 6 above) — reconstructed from memory, NOT from
re-reading the artifact or the source folders this session:**

- Bernat 2024's rxnfiles are git-tracked; Tim Bernat co-developed the ideas behind
  `reaction_model_builder.py` and is also the source for `_staging_new_schema/`,
  confirmed as active, intentional future work (de-hardcoding FAS specifics), not
  cruft. → memory `bernat-2024-provenance`, `reaction-builder-module-split`.
- Chain-length ladder (C4_NoFB...C20+unsat) replaced an earlier enzyme-count ladder
  because the enzyme ladder was 78–81% dead species — this is *why* an earlier
  "floor inversion" result was confusing before the switch. → memory
  `chain-length-scaling-axis`.
- "NoFB" = no FabF/FabB, a genuinely different model from plain C4 (56 vs 82
  reactions), not a subset. → memory `nofb-means-no-fabf-fabb`.
- Scaling groups must never default to 1 for d-type (additive-in-exp) groups; this
  was caught via C4_NoFB non-convergence and is a real historical bug class. → memory
  `scaling-coeffs-must-be-explicit`. **This bug recurred tonight** — see the C10 fix
  in `ODE Runner/run_model.ipynb`, and it's worth checking whether any of the audit's
  flagged notebook call-sites (item below) have the same silent-default problem.

**What this session does NOT have reloaded:** the MATLAB folder's actual git
history/file structure, the Fox Group papers' content, and `Outline_v2.md`'s current
content. The published artifact at the URL above is the authoritative first pass —
**re-read it via `Artifact` (`action: "read"`) before resuming**, rather than trusting
this session's paraphrase of it. Do not assume the artifact and this file agree on
every point; this file is a supplement, not a replacement.

**Item 5 (redundant/dead/superseded within Python Model) — this session DOES have a
verified, current list**, from a separate 2.5-hour read-only workflow audit conducted
earlier in this same session (published as artifact "FAS Inference Audit" — different
artifact from the one linked above; find it via `Artifact action:"list"` if the URL
isn't at hand). Re-verified as still-accurate as of tonight (2026-09-09):

| # | Finding | Verified status tonight |
|---|---|---|
| A-01 | `environment.yml`/`requirements.txt` uncommitted, pin jax versions nothing runs | still open |
| A-02 | 5 untracked Utilities modules (`generate_synthetic_ground_truth.py`, `grouping_diagnostics.py`, `heldout_predictive.py`, `plot_predictive_check.py`, `plot_trace_diagnostics.py`), 2 undocumented | still open |
| B-03 | Cluster's `inference_plotting.py` stale (1153 lines, Aug 27) vs local (1930 lines, Sep 2) → `plot_trace_diagnostics`/`grouping_diagnostics` presumably still ImportError on cluster | still open — the one-line fix (`Sync/sync_to_cluster.sh Utilities`) has never actually been run |
| B-04 | Cluster deploy has no git identity (rsync-only) | still open |
| B-05 | 16 notebook call sites will raise on the new `scaling_group` signature | 2 of 6 originally-flagged notebooks fixed (`run_model.ipynb` tonight, one other earlier); 4 never touched (`generate_model_error_configs.ipynb` x2, `optimize_diffrax_solver.ipynb`, `optimize_pymc_solver.ipynb`, `guided_bayesian_inference.ipynb`) |
| C-06 | Silent default mismatches: `max_steps` 10k/20k inconsistency, `atol` 1e-8 vs validated 1e-7 | still open |
| D-08/09 | 14 stale `.bak_preC` files, 940 untracked Archive files | still open, unchanged counts |
| D-10/11 | 16 undocumented modules; `_staging_new_schema` now stale | informational |

Also still pending, not part of the numbered findings: renaming/broadening
`generate_scaling_test_data.ipynb` for general use (acknowledged desired, never done).

---

## The efficiency thread: floor vs. no-floor — full state as of tonight

This is the thread you asked to resume "same logic-tracking-and-flagging approach."
Here is everything decided, tested, and still open.

### The question and the headline answer

Masking negative ODE concentrations (`jnp.maximum(y, 0.0)` in the reaction-network
RHS — "floor") was suspected of hurting either accuracy or speed relative to letting
the solver see negatives and crash-detect via `EQX_ON_ERROR=nan` ("no-floor"). Under
the **production prior** ("tightest", a1 ∈ [0.1, 10]): no-floor is looking **safe and
up to ~9x faster on C10** than the floor. This reverses an earlier verdict — under the
old, wide [0.001, 1000] prior, the floor was needed to prevent solver crashes; that
finding does not hold at the tighter, production-relevant prior width. → memory
`floor-does-not-prevent-crash` (marked "still confirming, not yet closed" — the
convergence sweep below is what closes it).

**Caveat you should re-flag if this goes in the paper:** "floor beats no-floor on C6"
is also true and currently unexplained beyond "startup cost scales very differently
between the two" (see below) — the overall speed win is not uniform across the
ladder, and a paper claim needs the *mechanism*, not just the net result.

### What was tested, and how device/hardware effects were isolated

- Full ladder (C4_NoFB through C20+unsat) run under both floor and no-floor, under
  both "tightest" [0.1,10] and "narrowest" [0.05,20] prior widths — 4 conditions x 14
  systems, tracked in `job_files/masking_check/speedup_analysis.ipynb`.
- Runs landed on a mix of A100/H100 NVL/H100 PCIe MIG/V100 variants (Blanca's shared
  pool), so raw wall-clock isn't comparable across runs without a hardware
  normalization factor. Rather than trust one method, **three independent methods**
  were cross-validated:
  1. Dead-run cost law (A100 dead runs fit `rate ∝ rxns^-1.16`, R²=0.988) → back-solve
     other devices' factors from their dead-run residuals.
  2. Direct within-run overlap (only the C14 dead-floor run spans A100 and H100 NVL
     directly).
  3. Shared-binned-stage-curve regression on live runs, divergent windows excluded
     (`log(rate) = run + g(step) + device`).
  - **Adopted factors (A100 = 1.00):** H100 NVL 1.37, H100 PCIe MIG 3g.40gb 0.90,
    V100-SXM2-32GB 0.67, V100-PCIE-32GB 0.75, V100-PCIE-16GB 0.75.
  - A 4th check (dropping V100 windows entirely vs. keeping them scaled) found scaling
    them barely changes the reaction-count scaling fit (R² 0.964 vs 0.949) — the
    factors are trustworthy enough to keep the V100 data in.
- **RETRACTED 2026-09-09 — do not cite this exponent.** This bullet originally
  claimed no-floor's startup cost scales steeply (`rxns^2.2-2.8`) while floor's is
  flat (`rxns^0.6-0.9`), and treated that as the leading explanation for "floor wins
  on C6." Checked directly against `speedup_analysis.ipynb`'s own live
  `startup_cost()` function (not a re-derivation): floor's startup cost is actually
  **larger** than no-floor's at nearly every system, the opposite of this claim. The
  original numbers traced to a background analysis run before this notebook had
  dead-checkpoint filtering, which silently mixed failed-attempt time into what
  looked like "startup." See `Notes/efficiency_logic_map.md` section 6 for the
  corrected picture: there is no clean startup-tax asymmetry, and why floor
  specifically wins on C6 is still an open question — not this one.
- Cost-per-ODE-solve comparison (isolating the floor's own per-step overhead from
  everything else) is only meaningful on C4_NoFB/C6/C8 — those are the only systems
  where BOTH conditions have live (non-dead) data to compare.
- NUTS step-size adaptation: floor's step size collapses to ~7e-13 (numerically
  degenerate) on multiple systems — visible across the *whole* ladder, deliberately
  not capped to the 3-system pairing above since dead runs are the point of that
  panel.

### The stranded-chain bug found and fixed via this thread

While comparing chains for the floor/no-floor decision, found a real MCMC bug
unrelated to floor/no-floor itself but exposed by it: a chain can get permanently
stuck in a much-lower-posterior basin than its siblings, and **neither r-hat, ESS, nor
BFMI detects it** (tested directly: rogue chain's BFMI sat inside the healthy range).
The only reliable, cheap (96 µs/call) discriminator is the **log-posterior gap**
relative to the best chain (>20 nats ≈ e^-20 ≈ negligible relative mass).

- Fixed in `Utilities/resumable_sampler.py`: `_stranded_chains()` runs before the
  r-hat/ESS loop, excludes chains >20 nats below the best from convergence
  diagnostics (not from saved draws), refuses convergence if survivors <
  `min_chains_for_convergence` (default 4).
- **Validated live on the cluster tonight, on two real, different failure shapes:**
  - C8 (job 28213043): excluded **1 of 8** chains, 533 nats below the rest — the
    original single-rogue-chain case that motivated the fix.
  - C10 narrowest (job 28213044): excluded **4 of 8** chains, all ~6975 nats below —
    a much larger gap and a much larger fraction of chains than the C8 case.
    **This is new information, not yet interpreted**: a 4-way even split at a huge
    posterior gap looks less like "one chain wandered" and more like "the narrower
    a1 prior is exposing a real second solution branch that half the chains find."
    If that's true, it's directly relevant to the paper (identifiability claim) and
    to the efficiency question (a genuinely bimodal posterior costs more to sample
    correctly, not just longer chains). **Open question for you:** is a1 physically
    expected to have two solution branches under a narrower prior, or should this
    read as a pathology specific to the narrowest-width run?
- All production jobs were cancelled at their next checkpoint and rotated onto this
  fixed code tonight (see next section for a wrinkle discovered doing that).

**Checked where the stranded chains actually land (2026-09-09) — not a boundary
artifact, and not the same secondary mode in both cases:**

| Run | Prior bounds | Primary mode (survivors) | Secondary mode (stranded) | Gap | Fraction stranded |
|---|---|---|---|---|---|
| C8 tightest | [0.1, 10] | a1 ≈ 1.000 (std 0.011) | a1 ≈ **0.433** (std 0.006) | 533 nats | 1/8 |
| C10 narrowest | [0.05, 20] | a1 ≈ 0.999 (std 0.017) | a1 ≈ **9.41** (std 0.04) | 6975 nats | 4/8 |

Both secondary modes are tight, well-converged, and nowhere near either prior edge —
this rules out "chain stuck at the boundary" as the explanation in both cases. But
the two secondary values are **not the same alternate solution reappearing**: one
sits well below the primary (0.43x), the other well above (9.4x), at different chain
lengths, with a much weaker/rarer signal at C8 (1 chain, smaller gap) than at C10 (4
of 8 chains — half the run — much bigger gap). **Open question for you, not
guessable from the code:** does a1 have a plausible alternate kinetic regime at
either of these values (e.g. a different rate-limiting step becoming locally
competitive), and if the two chain lengths landing on different secondary values is
expected? Also concretely relevant to the production prior: **C10's secondary mode
at 9.41 sits just under the *tightest* prior's own upper bound of 10** — if C10 is
ever run under tightest instead of narrowest, this second mode may get truncated or
distorted right at the boundary rather than explored cleanly, which could produce a
misleading "chain hit the wall" read instead of a legible second mode.

**User's domain answer (2026-09-09 ~01:35): "a1 shouldn't [have a second solution]
without changing another parameter too."** Since a1 is the *only* free parameter in
these runs (everything else pinned at nominal), this reframes the finding: a clean
bimodal a1-only posterior is therefore NOT expected to be physical. That reading is
also what the numbers already say on their own terms — the "secondary" mode is 533
and 6975 nats *below* the primary in C8 and C10 respectively (e^-533, e^-6975 of the
relative mass), i.e. these were never remotely competitive solutions. The real
question was always "why does NUTS get trapped in a region that far below the true
posterior," not "is this a competing physical answer" — and the answer to that
narrower question turned out to be checkable directly.

**Autonomous follow-up (2026-09-09 ~01:45), done without waiting for the domain
answer above since it only needed data already on disk:** does the FLOOR version of
either run show the same secondary mode?

| Run | Phase | Per-chain a1 means | lp gap |
|---|---|---|---|
| C8 tightest **floor** | sampling (400 draws) | all 8 chains 0.9991–1.0002 | **0.2 nats** — clean, single mode |
| C10 narrowest-matched **floor** | still warmup (50 steps) | 1.38–5.47 (wildly scattered) | 17,664 nats |
| C10 tightest **floor** | still warmup (155 steps) | 0.86–3.39 (wildly scattered) | 17,078 nats |

**C8 is a clean, direct answer: floor shows zero sign of the secondary mode that
no-floor produces.** This is consistent with the secondary mode being a
no-floor-specific numerical artifact — plausibly a spurious local likelihood basin
created by the solver tolerating negative-concentration excursions rather than
clamping them, which is exactly the mechanism this whole investigation exists to
scrutinize. **C10 floor could not be checked the same way — it hasn't reached
sampling yet**, and its warmup-phase per-chain lp spread (17,000+ nats) is far
larger than anything seen elsewhere in this project. That number alone isn't
necessarily alarming (warmup is non-stationary adaptation, per the sampler's own
docstring, and large early spread is somewhat expected before the mass matrix
settles) — but it's worth a direct look once/if C10 floor reaches sampling, both to
complete the floor-vs-no-floor comparison and because a spread that large, that long
into warmup, is itself a mild yellow flag for C10-floor's own convergence.

### A live infra bug found *tonight*, directly relevant to the near-term success criterion

While rotating the 9 running jobs onto the fixed sampler code, **3 of 9 (C10, C14,
C20+unsat) silently lost their self-resubmission** — the job was cancelled and no
successor was ever queued, so each sat with **zero compute happening for 12-25
minutes** before this was caught by hand (not by any automated check) and manually
patched by resubmitting them directly. Root cause: the sbatch script's
`trap '_do_resubmit ...' TERM` only runs once bash's foreground `wait()` on the
Python child returns; if Python doesn't exit before SLURM's kill-escalation window,
SIGKILL takes the whole process group out before the trap can call `sbatch`. This is
a race, not a logic bug — 6/9 jobs won the race, 3/9 didn't.

**FIXED 2026-09-09, deployed and verified on the cluster.** You chose (a): whatever
cancels a job now submits its successor directly rather than trusting the doomed
job's own trap. `job_files/masking_check/cancel_and_resubmit.sh` (new) does
`scancel` then calls `resubmit_if_needed.sh` itself; that script gained an atomic
per-segment `mkdir` lock so the old TERM trap (left in place as a fallback for real
unattended preemption) and the new direct call can't both submit a successor for the
same cancellation. Any future checkpoint-then-cancel cycle should call the wrapper,
not a bare `scancel`.

**Also added, same session:** a hard **24h total-wall-clock-since-first-checkpoint**
cap, enforced in *two* independent places so it can't be silently skipped:
- `resubmit_if_needed.sh` refuses to queue a new segment once 24h (configurable via
  `RESUB_MAX_TOTAL_HOURS`) has passed since the run's first `progress_log.jsonl` row.
- `SamplerSpec.max_total_hours` (new field, `Utilities/resumable_sampler.py`,
  default 24.0) makes the *running* sampler itself stop at the next checkpoint once
  crossed — `stopped_reason="total_time_budget"` — rather than running out its full
  12h segment first. Read from `solver_params.json`'s `posterior_sampling.max_total_hours`
  if present, else 24.0. Confirmed additive: not part of `config_signature`, so it
  does not invalidate any already-running checkpoint.
- **This is the same knob you'll reuse for the eventual full-model production run's
  multi-week budget** (e.g. `max_total_hours=336.0` for a 2-week cap) — just set it
  per-run in that run's `solver_params.json`. No new mechanism needed later.
- Pushed to the cluster and import-verified (`max_total_hours default: 24.0`
  confirmed via a live import on the Bayesian env). Applies to *new* segments only —
  the 9 jobs already executing keep running under the code they loaded at their own
  start until their next resubmission.

### Long-term success criterion: current signal is concerning, not yet triaged

As of tonight (2026-09-09, ~00:53), live per-step rates on the largest systems, freshly
resumed onto the fixed code, under the tightest/no-floor condition:

| System | warmup progress | rate |
|---|---|---|
| C16+unsat | 360/1000 | 36s/step (ETA 6.4h to finish warmup alone) |
| C18 | 390/1000 | 24s/step (ETA 4.0h) |
| C18+unsat | 160/1000 | 112s/step (ETA 26.0h) |
| C20 | 275/1000 | 61s/step (ETA 12.4h) |
| C20+unsat | 105/1000 | 1341s/step (ETA **333.5h**, i.e. ~14 days, for warmup alone) |

**Do not take the C20+unsat number at face value yet** — it's computed from only two
checkpoints immediately after a resubmission (small-sample rate estimate, and this
system is one of the three that just sat idle for 25 minutes, so the timing window
spans a cold restart). It needs to be re-measured once C20+unsat has run cleanly for
a while. But even discounting that number heavily, **C18+unsat at 112s/step for 1000
tune + thousands of draws, times 8 chains, does not obviously fit inside a 2-week
budget**, and this is before even considering the "handful of parameters" (i.e.
higher-dimensional, presumably more expensive) target of the long-term goal. This
combination — largest systems, worst rates, least scrutiny so far — is the highest-
leverage place to point the efficiency audit once the floor/no-floor question is
closed out.

**Also relevant and unresolved:** a separate, currently-parked plan exists at
`~/.claude/plans/i-want-to-test-twinkling-kernighan.md` for redesigning how
C20+unsat's training/test conditions are generated (self-referential strict-tolerance
reference, decoupled from a forced 9-condition count). That plan was paused
mid-design, before this floor/no-floor detour, and touches the same system
(C20+unsat) that's now showing the worst throughput. **Worth deciding explicitly:
is C20+unsat's bad rate a solver-settings problem (which that paused plan was already
investigating) or a floor/no-floor problem (this thread), or both?** Don't assume
either without checking.

---

## Suggested order of operations for whenever this resumes

1. **DONE.** Re-read the linked artifact fresh — confirmed it's the workflow audit
   only (A-01..E-13); the lineage/papers/outline work (items 1-3 of the original
   request) was never actually produced as a deliverable, contrary to what the
   status update assumed. Starting from memory fragments only when step 6 comes up.
2. **DONE.** Infra-race fixed (canceller-submits-successor + dedup lock) and a 24h
   total-run-time cap added in both the bash resubmit gate and the sampler itself —
   see the infra-bug section above for full detail. Queue contention (3 jobs stuck
   PENDING behind the other 9) was left to resolve naturally, per your call.
3. **RETRACTED as originally framed, still open as a question.** The proposed
   "startup-cost-asymmetry" explanation for why floor wins on C6 did not survive a
   direct check (see the retraction above) — the mechanism is unexplained again, not
   closed. It's the one open mechanistic question standing between "no-floor
   is faster" and "no-floor is faster and we understand why," which matters for a
   paper claim.
4. Interpret the C10-narrowest 4-way stranded split — ask whether a second solution
   branch is physically plausible for a1 before treating it as pure pathology.
5. Re-measure C16+unsat through C20+unsat throughput once they've run clean for a few
   hours (not immediately post-restart), then assess against the 2-week/A100 goal
   with real numbers rather than the noisy ones above.
6. Only then return to the five-location lineage reconstruction (items 1-3, 6 of the
   original request) and the Python Model cleanup list (item 5, table above) — those
   don't have a clock on them the way the live cluster runs do.
