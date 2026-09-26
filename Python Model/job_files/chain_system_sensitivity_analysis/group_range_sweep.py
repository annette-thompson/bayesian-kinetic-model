"""Find prior bounds for a3, c3 and a2, the same way the a1 range was found.

The a1 range was set by walking outward from 1.0 logarithmically until the parameter
stopped mattering, then refining between decades to 2 significant figures. This does
the same for the three groups the ranking put ahead of a1 on scaling reach, using an
explicit information criterion rather than an eyeballed plateau.

TWO INDEPENDENT CRITERIA -- deliberately NOT merged

An earlier version of this walked outward and stopped at whichever of "signal died"
or "solver blew up" happened first, then reported that as "the range". Those are
different questions with different consequences and merging them hid which one was
binding: an a2 bound of [1, 2] looked like a statement about information when it was
purely the step ceiling firing, with the sensitivity test never reaching that region
at all.

So every multiplier is now evaluated with a GENEROUS ceiling, guaranteeing the
sensitivity measurement completes, and the step count is recorded alongside it as a
separate curve. Two boundaries come out per (group, system):

  INFORMATIVE bound -- where a further factor of two stops moving the observables
      above the noise floor. Beyond it the likelihood is flat, so prior mass there is
      dead sampling cost. A property of model + data + noise.

  SOLVABLE bound -- where the solve cost exceeds STEP_CEIL_FACTOR x the system's own
      nominal baseline. Beyond it solves fail or crawl, producing NaN gradients and
      divergences. Partly a policy choice, since the ceiling is chosen not derived.

The usable prior is their intersection, and the report says which one binds.

ORIGINAL CRITERION NOTE

At each candidate multiplier m the local sensitivity is measured as the observable
change between m and 2m:

    S(m) = median over conditions of ( max over observables of |y(2m) - y(m)| / |y(m)| )

and compared against the 10% relative noise these datasets carry. Where S(m) < 0.10 a
further factor-of-two move in the parameter is invisible to the data: the likelihood
is flat, the posterior there is just the prior, and any prior mass beyond that point is
dead sampling cost. The outermost m with S(m) >= 0.10 is the bound. The same test runs
inward (m vs m/2) for the lower bound.

A second, independent stopping condition is numerical. Extreme multipliers drive the
ODE into stiff regimes, so each solve is capped at STEP_CEIL_FACTOR x the system's own
nominal baseline step count, with max_steps set to that cap so a bad region fails fast
rather than grinding. A multiplier whose solves blow the cap ends the walk even if the
observables were still responding -- an unsolvable region is not usable prior mass
regardless of how informative it would be.

Bounds are reported per system and then reduced to the MOST RESTRICTIVE across all 14,
since one prior has to serve the whole ladder. Output is rounded to 2 significant
figures, matching how the a1 bounds were reported.

Usage: python group_range_sweep.py [--groups a3,c3,a2] [SYSTEM ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")

import numpy as np
import pandas as pd
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd
from reaction_model_builder import set_scaling_group_values

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

ALL_SYSTEMS = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
               "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]
DEFAULT_GROUPS = ["a3", "c3", "a2"]

NOISE = 0.10              # relative noise model on these datasets
STEP_CEIL_FACTOR = 5.0    # same 5x-baseline ceiling used for the a1 range
MEASURE_CAP = 20_000      # generous ceiling used WHILE MEASURING, so a region
                          # that merely exceeds STEP_CEIL_FACTOR is still
                          # evaluated for informativeness instead of aborting
HARD_CAP = 20_000
OBS_FLOOR = 1e-9          # ignore observables too small to be measured; a species at
                          # 1e-12 uM produces enormous relative changes that are
                          # numerically real and physically meaningless
# Coarse ladder outward from 1.0, then refined between the last good and first bad rung.
COARSE_UP = [2, 5, 10, 20, 50, 100, 300, 1000]
COARSE_DN = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.003, 0.001]


def sig2(x):
    if x == 0:
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + 1)


def build(system):
    cfg = json.loads((ROOT / "Results/Chain Scaling Tests" / f"Chain {system} - a1 tightest"
                      / "solver_params.json").read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                           icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"],
                           scaling_group_overrides=sg)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN)
    return sys_, targets, sg, ctrl


def conditions(sys_, system):
    df = pd.read_csv(ROOT / "Data" / f"Chain_{system}" / "init_vs_final_conc.csv")
    sweep = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
    y0s = []
    for _, r in df.iterrows():
        y0 = np.array(sys_.y0(), dtype=float)
        for n in sweep:
            col = f"{n} (uM)"
            if col in df.columns:
                y0[sys_.index_of[n]] = float(r[col])
        y0s.append(y0)
    return y0s


def solve_set(sys_, theta, y0s, idx, ctrl, cap):
    """Observables, the MAX step count over conditions, and whether any solve failed
    outright at this (generous) cap. Cost is reported rather than used to abort, so
    the caller can judge the two criteria independently."""
    outs, blew, max_steps = [], False, 0
    for y0 in y0s:
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
            t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
            saveat=dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
            max_steps=int(cap), throw=False)
        n = int(sol.stats["num_steps"])
        max_steps = max(max_steps, n)
        ok = bool(sol.result == dfrx.RESULTS.successful) and n < cap
        if not ok:
            blew = True
            outs.append(None)
            continue
        final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
        outs.append(np.array([float(final[i]) for i in idx]))
    return outs, max_steps, blew


def sensitivity(a, b):
    """Median over conditions of max over measurable observables of |b-a|/|a|."""
    vals = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        mask = np.abs(x) > OBS_FLOOR
        if not mask.any():
            continue
        vals.append(float(np.max(np.abs(y[mask] - x[mask]) / np.abs(x[mask]))))
    return float(np.median(vals)) if vals else None


def group_value_for(group, mult):
    """Group value inducing rate multiplier `mult`.

    Ordinary groups multiply a rate constant, so the value IS the multiplier. The
    d-prefixed groups enter additively inside exp() (TesA's 1/exp(12*d1+d2)) with a
    nominal of 0.0, so setting them to `mult` directly would be meaningless -- d1=10
    is not "10x", it is exp(-120). Solving 12*d1 + d2 = -ln(mult) puts them on the
    same multiplier axis as everything else, matching morris_screen.py.
    """
    if group == "d1":
        return -np.log(mult) / 12.0
    if group == "d2":
        return -np.log(mult)
    return mult


def theta_at(sys_, sg, group, m):
    o = dict(sg)
    o[group] = group_value_for(group, m)
    return set_scaling_group_values(sys_.theta, sys_.params, o)


def walk(sys_, sg, group, y0s, idx, ctrl, cap, direction):
    """Walk outward recording BOTH criteria independently at every rung.

    `cap` is the STEP_CEIL_FACTOR budget and is only compared against, never used to
    abort: solves run at MEASURE_CAP so the sensitivity number always exists. Returns
    the last multiplier that was still informative, the last that was still within
    budget, and the per-rung trace.
    """
    ladder = COARSE_UP if direction > 0 else COARSE_DN
    last_informative, last_solvable = 1.0, 1.0
    informative_ended, solvable_ended = None, None
    trace = []
    for m in ladder:
        base, steps_a, hard_fail_a = solve_set(sys_, theta_at(sys_, sg, group, m),
                                               y0s, idx, ctrl, MEASURE_CAP)
        probe_m = m * 2 if direction > 0 else m / 2
        probe, steps_b, hard_fail_b = solve_set(sys_, theta_at(sys_, sg, group, probe_m),
                                                y0s, idx, ctrl, MEASURE_CAP)
        s_val = None if (hard_fail_a or hard_fail_b) else sensitivity(base, probe)
        within_budget = (not hard_fail_a) and steps_a <= cap
        if within_budget:
            last_solvable = m
        elif solvable_ended is None:
            solvable_ended = (f"steps {steps_a} > budget {cap}" if not hard_fail_a
                              else f"solve failed even at {MEASURE_CAP} steps")
        if s_val is not None and s_val >= NOISE:
            last_informative = m
        elif informative_ended is None:
            informative_ended = (f"sensitivity {s_val*100:.1f}% < {NOISE*100:.0f}%"
                                 if s_val is not None else "unsolvable, sensitivity unknown")
        trace.append(dict(m=m, sensitivity=s_val, steps=steps_a,
                          within_budget=bool(within_budget)))
        if informative_ended and solvable_ended:
            break
    return dict(informative=last_informative, solvable=last_solvable,
                informative_ended=informative_ended or "reached end of ladder",
                solvable_ended=solvable_ended or "reached end of ladder",
                trace=trace)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gsel = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--groups=")]
    groups = gsel[0].split(",") if gsel else DEFAULT_GROUPS
    systems = args or ALL_SYSTEMS

    results = {}
    for system in systems:
        sys_, targets, sg, ctrl = build(system)
        idx = [sys_.index_of[t] for t in targets]
        y0s = conditions(sys_, system)
        # size the ceiling off this system's own nominal cost, as the a1 sweep did
        base_steps = 0
        for y0 in y0s:
            sol = dfrx.diffeqsolve(
                dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
                t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
                y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
                saveat=dfrx.SaveAt(t1=True),
                stepsize_controller=dfrx.PIDController(
                    rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                    icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
                max_steps=HARD_CAP, throw=False)
            base_steps = max(base_steps, int(sol.stats["num_steps"]))
        cap = max(50, int(np.ceil(STEP_CEIL_FACTOR * base_steps)))
        print(f"\n=== {system} === rtol={ctrl['rtol']:g}  baseline max steps={base_steps} "
              f"-> step cap {cap} ({STEP_CEIL_FACTOR:g}x)", flush=True)

        for g in groups:
            if g not in sg:
                print(f"  {g:<4} not present in this system", flush=True)
                continue
            up = walk(sys_, sg, g, y0s, idx, ctrl, cap, +1)
            dn = walk(sys_, sg, g, y0s, idx, ctrl, cap, -1)
            results.setdefault(g, {})[system] = dict(
                informative=[dn["informative"], up["informative"]],
                solvable=[dn["solvable"], up["solvable"]],
                usable=[max(dn["informative"], dn["solvable"]),
                        min(up["informative"], up["solvable"])],
                why_inf_lo=dn["informative_ended"], why_inf_hi=up["informative_ended"],
                why_slv_lo=dn["solvable_ended"], why_slv_hi=up["solvable_ended"],
                trace_up=up["trace"], trace_dn=dn["trace"])
            r = results[g][system]
            # Which criterion actually binds is the point of separating them: an
            # informative range far wider than the solvable one means the parameter
            # is worth inferring but the integrator cannot follow the sampler there.
            binds_lo = "solver" if dn["solvable"] > dn["informative"] else "signal"
            binds_hi = "solver" if up["solvable"] < up["informative"] else "signal"
            print(f"  {g:<4} informative [{sig2(r['informative'][0]):g}, {sig2(r['informative'][1]):g}]"
                  f"   solvable [{sig2(r['solvable'][0]):g}, {sig2(r['solvable'][1]):g}]"
                  f"   -> usable [{sig2(r['usable'][0]):g}, {sig2(r['usable'][1]):g}]"
                  f"   binding: lo={binds_lo} hi={binds_hi}", flush=True)

    print("\n" + "=" * 96)
    print("MOST RESTRICTIVE RANGE ACROSS ALL SYSTEMS (one prior serves the ladder)")
    print("Reported separately, because a collapsed USABLE range means different things")
    print("depending on which criterion caused it: a narrow INFORMATIVE range says the")
    print("parameter is not worth inferring, while a narrow SOLVABLE one says it is worth")
    print("inferring but the integrator cannot follow the sampler there -- a tractability")
    print("problem that a looser step ceiling or a per-system prior could fix.")
    print("=" * 96)
    summary = {}
    for g, per in results.items():
        def isect(key):
            lo = max(v[key][0] for v in per.values())
            hi = min(v[key][1] for v in per.values())
            blo = [s for s, v in per.items() if v[key][0] == lo]
            bhi = [s for s, v in per.items() if v[key][1] == hi]
            return lo, hi, blo, bhi

        ilo, ihi, iblo, ibhi = isect("informative")
        slo, shi, sblo, sbhi = isect("solvable")
        ulo, uhi, ublo, ubhi = isect("usable")
        collapsed = ulo >= uhi
        cause = ("n/a" if not collapsed else
                 "SOLVER" if (slo >= shi) else "SIGNAL" if (ilo >= ihi) else
                 "DISAGREEMENT ACROSS SYSTEMS")
        summary[g] = dict(informative=[sig2(ilo), sig2(ihi)],
                          solvable=[sig2(slo), sig2(shi)],
                          usable=[sig2(ulo), sig2(uhi)],
                          collapsed=bool(collapsed), collapse_cause=cause,
                          binding_usable_lower=ublo, binding_usable_upper=ubhi)
        print(f"\n  {g}")
        print(f"    informative [{sig2(ilo):g}, {sig2(ihi):g}]"
              f"   (lo set by {iblo[:3]}, hi by {ibhi[:3]})")
        print(f"    solvable    [{sig2(slo):g}, {sig2(shi):g}]"
              f"   (lo set by {sblo[:3]}, hi by {sbhi[:3]})")
        print(f"    USABLE      [{sig2(ulo):g}, {sig2(uhi):g}]"
              + (f"   *** COLLAPSED -- cause: {cause} ***" if collapsed else ""))

    # Name the output for the groups and systems it covers. Two concurrent
    # invocations (e.g. a ladder-wide sweep and a benchmark-set sweep) would
    # otherwise both write "group_range_sweep.json" and the second would silently
    # destroy the first's results.
    slug = "-".join(groups) + "_" + ("all" if len(systems) > 5 else
                                     "-".join(s.replace("+", "") for s in systems))
    out = HERE / f"group_range_{slug}.json"
    out.write_text(json.dumps(dict(groups=groups, systems=systems,
                                   per_system=results, summary=summary), indent=2))
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
