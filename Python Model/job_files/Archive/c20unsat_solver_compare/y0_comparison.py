"""For each of the 13 non-C20+unsat ladder rungs, solve the baseline at the
tight reference tolerance (rtol=1e-10, atol=1e-12, PID=(0.4,0.3,0)) under TWO
initial conditions -- the normal fixed INITIAL_CONDITIONS, and that same
vector with every (nonzero) entry halved -- and report which needs fewer
total steps. Motivation: INITIAL_CONDITIONS is deliberately held IDENTICAL
across every rung (same absolute enzyme/substrate load regardless of chain
length), which may push smaller truncated systems into sharper, stiffer
transients than the full model ever sees -- independent of which candidate
PID/tolerance is later tested on that rung. This is a diagnostic-only run:
picks are reported for the user to confirm, not applied automatically.

    python y0_comparison.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import numpy as np
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd
from reaction_model_builder import build_ode_system_from_reactions

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

SYSTEM_TO_RXDIR = {
    "C4_NoFB": "C4", "C6": "C6", "C8": "C8", "C10": "C10", "C12": "C12",
    "C12+unsat": "C12+unsat", "C14": "C14", "C14+unsat": "C14+unsat",
    "C16": "C16", "C16+unsat": "C16+unsat", "C18": "C18", "C18+unsat": "C18+unsat",
    "C20": "C20", "C20+unsat": "C20+unsat",
}
REF_PID = (0.4, 0.3, 0.0)
REF_RTOL, REF_ATOL = 1e-10, 1e-12
MAX_STEPS = 300_000


def solve_steps(sys_, y0):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(
            rtol=REF_RTOL, atol=REF_ATOL, pcoeff=REF_PID[0], icoeff=REF_PID[1], dcoeff=REF_PID[2]),
        max_steps=MAX_STEPS, throw=False,
    )
    steps = int(np.asarray(sol.stats["num_steps"]))
    rejected = int(np.asarray(sol.stats["num_rejected_steps"]))
    converged = bool(sol.result == dfrx.RESULTS.successful) and steps < MAX_STEPS
    final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
    return dict(steps=steps, rejected=rejected, converged=converged,
                any_nan=bool(np.isnan(final).any()), any_negative=bool((final < 0).any()),
                max_abs=float(np.max(np.abs(final))) if np.isfinite(final).all() else None)


def main():
    results = {}
    for system, rxname in SYSTEM_TO_RXDIR.items():
        rx_dir = ROOT / "Reactions" / "EC_FAS_ME1" / rxname
        _, _, _, _, scaling_groups = build_ode_system_from_reactions(rx_dir)
        sys_ = gcd.ChainSystem(rx_dir, rtol=REF_RTOL, atol=REF_ATOL, pcoeff=REF_PID[0],
                               icoeff=REF_PID[1], dcoeff=REF_PID[2],
                               scaling_group_overrides=gcd.nominal_scaling_group_overrides(scaling_groups))
        y0_normal = sys_.y0()
        y0_half = y0_normal * 0.5

        t0 = time.time()
        r_normal = solve_steps(sys_, y0_normal)
        r_half = solve_steps(sys_, y0_half)
        elapsed = time.time() - t0

        if r_normal["converged"] and r_half["converged"]:
            pick = "normal" if r_normal["steps"] <= r_half["steps"] else "half"
        elif r_normal["converged"]:
            pick = "normal"
        elif r_half["converged"]:
            pick = "half"
        else:
            pick = "NEITHER_CONVERGED"

        results[system] = dict(normal=r_normal, half=r_half, pick=pick)
        print(f"[{system}] normal: steps={r_normal['steps']} rejected={r_normal['rejected']} "
              f"converged={r_normal['converged']}  |  half: steps={r_half['steps']} "
              f"rejected={r_half['rejected']} converged={r_half['converged']}  "
              f"-> pick={pick}  ({elapsed:.1f}s)", flush=True)

    out_json = HERE / "y0_comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
