"""How much does each scaling group actually move the observables?

The static footprint (scaling_footprint.py) says how many rate constants a group
multiplies. That is necessary but not sufficient: a group can touch 13 constants and
still be unidentifiable if the observables barely respond, and a group touching one
constant can dominate if that constant gates the whole pathway. This measures the
response directly.

Method, per system:

  * build the system ONCE at nominal scaling values, then vary one group at a time by
    rewriting theta via set_scaling_group_values. theta keeps its shape, so no JAX
    recompilation happens between points and the sweep is just forward solves.
  * solve the system's own 10 endpoint conditions (the ones its training data uses)
    at nominal to get the reference observables
  * for each group, for each target RATE MULTIPLIER m, re-solve all 10 and record the
    relative change in the observables

Making multiplicative and additive groups comparable is the subtle part. Ordinary
groups scale a rate constant directly, so the group value IS the multiplier m. The
d-groups enter TesA as 1/exp(12*d1 + d2) with nominal 0.0, so to induce the same rate
multiplier m they need
    12*d1 + d2 = -ln(m)
i.e. d1 = -ln(m)/12 (with d2 at nominal) or d2 = -ln(m) (with d1 at nominal).
Sweeping by induced multiplier rather than by raw parameter value is what lets d1 be
ranked on the same axis as a1 instead of against an arbitrary offset scale.

Reported per (system, group):

  detect_mult   smallest |log10 m| whose MEDIAN change exceeds the 10% noise floor.
                Low is good: the group is visible against noise near its nominal value.
  saturate_mult multiplier at which the median change first reaches ~100% and stops
                growing. This is the "does a narrow prior capture the full range"
                number the ranking cares about -- past saturation the likelihood is
                flat and extra prior width is dead sampling cost.
  span_decades  decades between detection and saturation: the width of the region
                where the data can actually distinguish values. Wider is better for
                identifiability; it is also the honest prior width.

Usage: python scaling_sensitivity.py [SYSTEM ...]      (default: C6 C14 C20+unsat)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import numpy as np
import pandas as pd
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd
from reaction_model_builder import set_scaling_group_values

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

DEFAULT_SYSTEMS = ["C6", "C14", "C20+unsat"]
HARD_CAP = 20_000
NOISE = 0.10                      # relative noise model on these datasets

# Induced rate multipliers, log-spaced and symmetric about 1.
MULTIPLIERS = [1e-3, 1e-2, 1e-1, 0.5, 2.0, 1e1, 1e2, 1e3]


def group_value_for(group, mult):
    """Group value that induces rate multiplier `mult`. See module docstring."""
    if group == "d1":
        return -np.log(mult) / 12.0
    if group == "d2":
        return -np.log(mult)
    return mult


def build(system):
    cfg_path = (ROOT / "Results" / "Chain Scaling Tests" / f"Chain {system} - a1 tightest"
                / "solver_params.json")
    cfg = json.loads(cfg_path.read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"],
                           pcoeff=ctrl["pcoeff"], icoeff=ctrl["icoeff"],
                           dcoeff=ctrl["dcoeff"], scaling_group_overrides=sg)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN)
    return sys_, targets, sg, ctrl


def conditions_for(sys_, targets, system):
    """The system's own 10 training conditions, as y0 vectors."""
    csv = ROOT / "Data" / f"Chain_{system}" / "init_vs_final_conc.csv"
    df = pd.read_csv(csv)
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


def solve_all(sys_, theta, y0s, idx, ctrl):
    """Observables for every condition under this theta; None where a solve fails."""
    outs = []
    for y0 in y0s:
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
            t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
            y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
            saveat=dfrx.SaveAt(t1=True),
            stepsize_controller=dfrx.PIDController(
                rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
            max_steps=HARD_CAP, throw=False)
        ok = bool(sol.result == dfrx.RESULTS.successful)
        final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
        outs.append(np.array([float(final[i]) for i in idx]) if ok else None)
    return outs


def rel_change(ref, got):
    """Per-condition max relative change, ignoring reference values at/near zero."""
    if got is None:
        return None
    denom = np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.abs(got - ref) / denom
    return float(np.nanmax(v)) if not np.all(np.isnan(v)) else None


def analyse(curve):
    """curve: {mult: median_rel_change}. Returns detect/saturate/span."""
    pts = sorted(curve.items(), key=lambda kv: abs(np.log10(kv[0])))
    detect = None
    for m, v in pts:
        if v is not None and v > NOISE:
            detect = m
            break
    sat = None
    ordered = sorted(curve.items(), key=lambda kv: np.log10(kv[0]))
    for m, v in ordered:
        if v is not None and v >= 0.99:
            if m < 1:
                sat = m
            elif sat is None or m > 1:
                sat = m
                break
    span = None
    if detect and sat:
        span = abs(np.log10(sat) - np.log10(detect))
    return detect, sat, span


def main():
    systems = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_SYSTEMS
    results = {}

    for system in systems:
        print(f"\n{'=' * 78}\n=== {system} ===\n{'=' * 78}", flush=True)
        sys_, targets, sg, ctrl = build(system)
        idx = [sys_.index_of[t] for t in targets]
        y0s = conditions_for(sys_, targets, system)
        print(f"  {len(sys_.species)} species, targets={targets}, "
              f"{len(y0s)} conditions, rtol={ctrl['rtol']:g}", flush=True)

        ref = solve_all(sys_, sys_.theta, y0s, idx, ctrl)
        n_ref_bad = sum(1 for r in ref if r is None)
        if n_ref_bad:
            print(f"  WARNING: {n_ref_bad} nominal solves failed; those conditions are skipped")

        sysres = {}
        for group in sorted(sg):
            curve = {}
            n_fail = 0
            for m in MULTIPLIERS:
                override = dict(sg)
                override[group] = group_value_for(group, m)
                theta = set_scaling_group_values(sys_.theta, sys_.params, override)
                got = solve_all(sys_, theta, y0s, idx, ctrl)
                vals = [rel_change(r, g) for r, g in zip(ref, got)
                        if r is not None]
                good = [v for v in vals if v is not None]
                n_fail += sum(1 for v in vals if v is None)
                curve[m] = float(np.median(good)) if good else None
            detect, sat, span = analyse(curve)
            sysres[group] = dict(curve=curve, detect=detect, saturate=sat,
                                 span_decades=span, n_failed_solves=n_fail)
            cur = "  ".join(f"{m:g}:{'--' if curve[m] is None else f'{curve[m]*100:.0f}%'}"
                            for m in MULTIPLIERS)
            print(f"  {group:<4} {cur}   detect={detect}  sat={sat}  "
                  f"span={'-' if span is None else f'{span:.1f}dec'}  fails={n_fail}",
                  flush=True)
        results[system] = sysres

    out = HERE / "scaling_sensitivity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
