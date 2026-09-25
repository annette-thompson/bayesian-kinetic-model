"""Compare the floor against a "robust linear solver" alternative on C4_NoFB.

The floor's crash's own error message named a specific fix: diffrax's Kvaerno5
uses AutoLinearSolver(well_posed=None) by default; setting well_posed=False
tells it up front to use a more robust (least-squares-style) algorithm instead
of assuming the system is well-posed and choking when it isn't. This is not a
custom solver -- it is the same equinox.tree_at one-field edit used in
truncated_warmup_test.py's "robust" mode, applied here directly (no monkeypatch
needed since this script builds the diffeqsolve call itself).

Three configurations run on the same y0 and the same 100 extreme-parameter
draws (same seed as forward_solve_test.py, via its own sample_extreme_overrides):
  floor        -- production: floored network, default Kvaerno5.
  nofloor      -- unclamped network, default Kvaerno5 (same as before).
  robust       -- unclamped network, Kvaerno5 with well_posed=False.

Unlike forward_solve_test.py's Part B (which ran floor and no-floor as two
separate loops), all three run inside the SAME loop iteration on the SAME
theta_i here, so final states can be compared pairwise per draw -- not just
via separate aggregate counts.

Scoped to C4_NoFB only: fast (the full three-system sweep took ~2.3 hours for
two configurations), and it's the system the original crash was reproduced on.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/projects/anth4580/Bayesian")
UTILITIES_DIR = PROJECT_ROOT / "Utilities"
JOB_DIR = PROJECT_ROOT / "job_files" / "chain_scaling_tests"
sys.path.insert(0, str(UTILITIES_DIR))
sys.path.insert(0, str(JOB_DIR))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfrx
import equinox as eqx

import forward_solve_test as fst  # reuse reactions_for, unclamp, classify_result, etc.
from generate_chain_data import ChainSystem, nominal_scaling_group_overrides
from reaction_model_builder import discover_scaling_groups, set_scaling_group_values

SYSTEM = "C4_NoFB"
N_EXTREME_DRAWS = fst.N_EXTREME_DRAWS
SEED = fst.SEED

ROBUST_SOLVER = eqx.tree_at(
    lambda s: s.root_finder.linear_solver.well_posed,
    dfrx.Kvaerno5(), False, is_leaf=lambda x: x is None,
)
DEFAULT_SOLVER = dfrx.Kvaerno5()


def run_once(network, solver, theta, y0, label: str) -> dict:
    """Same as forward_solve_test.run_once, but with the solver as a parameter."""
    t_wall0 = time.perf_counter()
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(network), solver,
        t0=0.0, t1=fst.T1, dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=theta,
        saveat=dfrx.SaveAt(steps=True),
        stepsize_controller=dfrx.PIDController(
            rtol=fst.RTOL, atol=fst.ATOL, pcoeff=fst.PCOEFF, icoeff=fst.ICOEFF, dcoeff=fst.DCOEFF),
        max_steps=fst.MAX_STEPS, throw=False,
    )
    jax.block_until_ready(sol)
    wall_s = time.perf_counter() - t_wall0

    total = int(np.asarray(sol.stats["num_steps"]))
    accepted = int(np.asarray(sol.stats["num_accepted_steps"]))
    ys_valid = np.asarray(sol.ys[:accepted]) if accepted > 0 else np.zeros((0,))
    finite = np.isfinite(ys_valid)
    crashed = (not bool(finite.all())) if ys_valid.size else True
    min_conc = float(np.min(ys_valid[finite])) if finite.any() else float("nan")
    result_name = fst.classify_result(sol.result)
    success = (result_name == "successful") and not crashed and total < fst.MAX_STEPS
    final_state = ys_valid[-1] if (accepted > 0 and not crashed) else None
    return dict(label=label, success=success, crashed=crashed, result_name=result_name,
               num_steps=total, num_rejected=total - accepted, wall_seconds=wall_s,
               min_concentration=min_conc, final_state=final_state)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rx = fst.reactions_for(SYSTEM)
    scaling_groups = discover_scaling_groups(rx)
    sys_ = ChainSystem(rx, fst.RTOL, fst.ATOL, fst.PCOEFF, fst.ICOEFF, fst.DCOEFF,
                      scaling_group_overrides=nominal_scaling_group_overrides(scaling_groups))
    y0 = sys_.y0()
    network_floor = sys_.network
    network_nofloor = fst.unclamp(sys_.network)

    configs = [
        ("floor", network_floor, DEFAULT_SOLVER),
        ("nofloor", network_nofloor, DEFAULT_SOLVER),
        ("robust", network_nofloor, ROBUST_SOLVER),
    ]

    print(f"=== {SYSTEM}: floor vs nofloor vs robust (Kvaerno5 well_posed=False) ===", flush=True)
    print("-- Part A: baseline (nominal theta), 720s --", flush=True)
    baseline = {}
    for name, network, solver in configs:
        r = run_once(network, solver, sys_.theta, y0, f"baseline/{name}")
        baseline[name] = r
        print(f"  {name:<10} success={r['success']!s:<5} crashed={r['crashed']!s:<5} "
             f"steps={r['num_steps']:>6} rejected={r['num_rejected']:>6} "
             f"wall_s={r['wall_seconds']:>8.3f} min_conc={r['min_concentration']:.6g}", flush=True)
    if baseline["floor"]["final_state"] is not None and baseline["robust"]["final_state"] is not None:
        diff = float(np.max(np.abs(baseline["floor"]["final_state"] - baseline["robust"]["final_state"])))
        print(f"  max |final_state difference| floor vs robust: {diff:.6g}", flush=True)

    print(f"-- Part B: {N_EXTREME_DRAWS} extreme-parameter draws, same y0, 720s --", flush=True)
    counts = {name: Counter() for name, _, _ in configs}
    max_steps = {name: 0 for name, _, _ in configs}
    wall_total = {name: 0.0 for name, _, _ in configs}
    worst_min_conc = {name: 0.0 for name, _, _ in configs}
    max_diff_floor_vs_robust = 0.0
    n_both_converged = 0
    jsonl_path = JOB_DIR / f"robust_solver_draws_{SYSTEM}.jsonl"
    with open(jsonl_path, "w") as jf:
        for i in range(N_EXTREME_DRAWS):
            overrides = fst.sample_extreme_overrides(rng, scaling_groups)
            theta_i = set_scaling_group_values(sys_.theta, sys_.params, overrides)
            results = {}
            for name, network, solver in configs:
                r = run_once(network, solver, theta_i, y0, f"extreme[{i}]/{name}")
                results[name] = r
                counts[name][r["result_name"]] += 1
                max_steps[name] = max(max_steps[name], r["num_steps"])
                wall_total[name] += r["wall_seconds"]
                if np.isfinite(r["min_concentration"]):
                    worst_min_conc[name] = min(worst_min_conc[name], r["min_concentration"])
            diff = None
            if results["floor"]["final_state"] is not None and results["robust"]["final_state"] is not None:
                diff = float(np.max(np.abs(results["floor"]["final_state"] - results["robust"]["final_state"])))
                max_diff_floor_vs_robust = max(max_diff_floor_vs_robust, diff)
                n_both_converged += 1
            jf.write(json.dumps(dict(
                draw=i, overrides=overrides,
                floor_result=results["floor"]["result_name"], floor_steps=results["floor"]["num_steps"],
                nofloor_result=results["nofloor"]["result_name"], nofloor_steps=results["nofloor"]["num_steps"],
                robust_result=results["robust"]["result_name"], robust_steps=results["robust"]["num_steps"],
                robust_wall_s=results["robust"]["wall_seconds"],
                floor_vs_robust_final_state_diff=diff,
            )) + "\n")
            jf.flush()

    for name, _, _ in configs:
        print(f"  {name:<10} {dict(counts[name])}, max_steps={max_steps[name]}, "
             f"total_wall_s={wall_total[name]:.2f}, worst_min_conc={worst_min_conc[name]:.6g}", flush=True)
    print(f"  floor vs robust: max final-state difference over {n_both_converged} jointly-converged "
         f"draws = {max_diff_floor_vs_robust:.6g}", flush=True)
    print(f"  per-draw detail written to {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
