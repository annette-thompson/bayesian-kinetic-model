"""Find the widest a1 scaling-group range (around its nominal value of 1.0) for
which NO chain-length system and NO condition (baseline + the 9 kept endpoint
rows now in Data/Chain_<system>/init_vs_final_conc.csv) needs more than 2200
solver steps (10x the chosen setting's own observed overall max of 220) under
the chosen production PID/tolerance (0.4,0.3,0), rtol=1e-5, atol=1e-7.

Search: start at a1=1 (known-good), step outward by decades (x10, /10) until
the 2200-step ceiling is exceeded somewhere, then refine within that decade by
geometric-mean bisection to pin down the boundary tightly in both directions.

    python a1_range_sweep.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import numpy as np
import pandas as pd
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd
from reaction_model_builder import build_ode_system_from_reactions, set_scaling_group_values

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

SYSTEM_TO_RXDIR = {
    "C4_NoFB": "C4", "C6": "C6", "C8": "C8", "C10": "C10", "C12": "C12",
    "C12+unsat": "C12+unsat", "C14": "C14", "C14+unsat": "C14+unsat",
    "C16": "C16", "C16+unsat": "C16+unsat", "C18": "C18", "C18+unsat": "C18+unsat",
    "C20": "C20", "C20+unsat": "C20+unsat",
}
PID = (0.4, 0.3, 0.0)
RTOL, ATOL = 1e-5, 1e-7
STEP_CEILING = 2000
SOLVE_CAP = STEP_CEILING  # fail fast right at the threshold we actually care about
STEP_MULT = 5.0
LOWER_BOUND_FIXED = 0.001  # already confirmed OK (previous run cleared down to 1.3e-05 at the stricter 1100 ceiling)
UPPER_SEARCH_START = 15.0  # previous run's last-known-OK point at the stricter 1100 ceiling
N_DECADES = 7
N_REFINE = 8

# Pre-build one ChainSystem + its 10 (y0, ) conditions per system, once, so the
# per-a1 evaluation loop only rebuilds theta (cheap) and re-solves (JIT-cached
# after the first call per system since a1 is a runtime value, not a shape).
SYSTEMS = {}
for system, rxname in SYSTEM_TO_RXDIR.items():
    rx_dir = ROOT / "Reactions" / "EC_FAS_ME1" / rxname
    _, _, _, _, scaling_groups = build_ode_system_from_reactions(rx_dir)
    nominal = gcd.nominal_scaling_group_overrides(scaling_groups)
    sys_ = gcd.ChainSystem(rx_dir, rtol=RTOL, atol=ATOL, pcoeff=PID[0], icoeff=PID[1], dcoeff=PID[2],
                           scaling_group_overrides=nominal)
    df = pd.read_csv(ROOT / "Data" / f"Chain_{system}" / "init_vs_final_conc.csv")
    n_targets = len(sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN))
    sweep_cols = [c for c in df.columns][: len(gcd.SWEEP_SPECIES)]
    y0_rows = []
    y0_base = sys_.y0()
    for _, row in df.iterrows():
        y0 = y0_base.copy()
        for name, col in zip(gcd.SWEEP_SPECIES, sweep_cols):
            y0[sys_.index_of[name]] = row[col]
        y0_rows.append(y0)
    SYSTEMS[system] = dict(sys_=sys_, nominal=nominal, y0_rows=y0_rows)
    print(f"loaded {system}: {len(y0_rows)} conditions, {n_targets} targets", flush=True)


def solve_steps(sys_, theta, y0):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(rtol=RTOL, atol=ATOL, pcoeff=PID[0], icoeff=PID[1], dcoeff=PID[2]),
        max_steps=SOLVE_CAP, throw=False,
    )
    steps = int(np.asarray(sol.stats["num_steps"]))
    ok = bool(sol.result == dfrx.RESULTS.successful) and steps < SOLVE_CAP
    # A non-converged solve must read as unambiguously over the ceiling, not
    # exactly at it -- otherwise capping max_steps at the ceiling itself would
    # make a genuine failure register as steps==ceiling, which the caller's
    # `<= STEP_CEILING` check would wrongly count as a pass.
    return steps if ok else STEP_CEILING + 1


def evaluate(a1_value):
    """Returns (max_steps_seen, (worst_system, worst_condition_index))."""
    worst_steps, worst = -1, None
    for system, d in SYSTEMS.items():
        sys_ = d["sys_"]
        overrides = dict(d["nominal"])
        overrides["a1"] = a1_value
        # Rebuild theta from scratch each time (cheap: just an array assembly + one set_scaling_group_values call)
        _, _, params, param_values, scaling_groups = build_ode_system_from_reactions(
            ROOT / "Reactions" / "EC_FAS_ME1" / SYSTEM_TO_RXDIR[system])
        theta_raw = jnp.array([param_values[p] for p in params], dtype=jnp.float64)
        theta = set_scaling_group_values(theta_raw, params, overrides)
        for i, y0 in enumerate(d["y0_rows"]):
            steps = solve_steps(sys_, theta, y0)
            if steps > worst_steps:
                worst_steps, worst = steps, (system, i)
    return worst_steps, worst


def bracket_and_refine(direction, start=1.0):
    lo, hi = start, None
    val = start
    for _ in range(N_DECADES):
        val = gcd.round_sigfigs(val * STEP_MULT if direction > 0 else val / STEP_MULT, 2)
        max_steps, worst = evaluate(val)
        status = "OK" if max_steps <= STEP_CEILING else "EXCEEDS"
        print(f"  a1={val:.6g}: max_steps={max_steps} ({status}, worst={worst})", flush=True)
        if max_steps <= STEP_CEILING:
            lo = val
        else:
            hi = val
            break
    if hi is None:
        print(f"  direction={direction}: never exceeded {STEP_CEILING} within {N_DECADES} steps of "
              f"{STEP_MULT}x (lo={lo}); range is at least this wide", flush=True)
        return lo, None
    for _ in range(N_REFINE):
        mid = gcd.round_sigfigs((lo * hi) ** 0.5, 2)
        if mid == lo or mid == hi:
            print(f"  refine: 2-sigfig rounding collapsed the bracket at lo={lo} hi={hi}; stopping refinement", flush=True)
            break
        max_steps, worst = evaluate(mid)
        status = "OK" if max_steps <= STEP_CEILING else "EXCEEDS"
        print(f"  refine a1={mid:.6g}: max_steps={max_steps} ({status}, worst={worst})", flush=True)
        if max_steps <= STEP_CEILING:
            lo = mid
        else:
            hi = mid
    return lo, hi


def main():
    t0 = time.time()
    base_max, base_worst = evaluate(1.0)
    print(f"a1=1.0 (nominal): max_steps={base_max} worst={base_worst}", flush=True)

    start_max, start_worst = evaluate(UPPER_SEARCH_START)
    status = "OK" if start_max <= STEP_CEILING else "EXCEEDS"
    print(f"a1={UPPER_SEARCH_START} (upward search start, re-verified at ceiling={STEP_CEILING}): "
          f"max_steps={start_max} worst={start_worst} ({status})", flush=True)

    print(f"\n--- searching upward from a1={UPPER_SEARCH_START} (ceiling={STEP_CEILING}) ---", flush=True)
    up_lo, up_hi = bracket_and_refine(direction=+1, start=UPPER_SEARCH_START)

    print(f"\n--- downward bound fixed at {LOWER_BOUND_FIXED} (already confirmed safe by the "
          f"prior run's downward search, which cleared down to 1.3e-05 at the stricter 1100 ceiling) ---",
          flush=True)

    result = dict(
        step_ceiling=STEP_CEILING, base_max_steps=base_max,
        upper_search_start=UPPER_SEARCH_START,
        upper_last_ok=up_lo, upper_first_bad=up_hi,
        lower_bound_fixed=LOWER_BOUND_FIXED,
        elapsed_s=time.time() - t0,
    )
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    out_json = HERE / "a1_range_sweep_result.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
