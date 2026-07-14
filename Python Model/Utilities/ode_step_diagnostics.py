"""Profile diffrax ODE solver step counts to size ``ODE_solver.max_steps``.

Runs the ODE solve over every experimental condition at a spread of parameter
values (nominal, the prior-box corners, and random log-uniform draws from each
free parameter's prior bounds) and reports the distribution of accepted/rejected/
total internal steps, plus how often the solve hits ``max_steps``.

Why this matters
----------------
``max_steps`` caps diffrax's internal adaptive loop. Too low -> stiff-region
solves fail (``result = max_steps_reached``), returning wrong/NaN trajectories,
which become NaN gradients and NUTS divergences. Too high only costs a little
memory (adaptive solves stop at t1 regardless of the cap), so set it a safe
multiple above the worst case the sampler will actually visit.

Per-solve step counts are impractical to print during MCMC (thousands of jitted,
vmapped solves), so this profiles representative parameter values up front.

Usage (from the "Python Model" directory):
    python Utilities/ode_step_diagnostics.py --solver_params_file "Results/<run>/solver_params.json"
    python Utilities/ode_step_diagnostics.py --solver_params_file ... --n-samples 40
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import diffrax as dfrx
import equinox as eqx
import jax
import jax.numpy as jnp

import inference_runner as ir
from inference_runner import (
    _build_solver,
    build_ode_system_from_reactions,
    get_free_parameter_names,
    import_solver_params,
    load_experiment_bundle,
    validate_experiment_config,
)


def _free_param_bounds(solver_params) -> dict[str, tuple[float, float]]:
    bounds = {}
    for spec in solver_params.get("free_kinetic_params", []):
        p = spec["prior_dist_params"]
        bounds[spec["param_name"]] = (float(p["lower"]), float(p["upper"]))
    return bounds


def profile_ode_steps(solver_params_file: str, n_samples: int = 20, seed: int = 0) -> dict:
    imported = import_solver_params(solver_params_file)
    sp = imported.solver_params
    ode_system, species_names, param_names, param_values, _scaling = build_ode_system_from_reactions(
        imported.reactions_source
    )
    validate_experiment_config(
        solver_params=sp, solver_params_file=str(imported.solver_params_file), species_names=species_names
    )
    experiment = load_experiment_bundle(
        solver_params=sp, solver_params_file=str(imported.solver_params_file), species_names=species_names
    )
    ir.CONDITION_MATRIX = experiment.condition_matrix_jax
    ir.EXPERIMENT = experiment

    ode_cfg = sp.get("ODE_solver", {})
    controller_cfg = sp.get("ODE_stepsize_controller", {})
    dt0 = ode_cfg.get("dt0", None)
    max_steps = int(ode_cfg.get("max_steps", 10_000))
    solver = _build_solver(sp)
    controller = dfrx.PIDController(**controller_cfg)
    t0, t1 = 0.0, float(experiment.simulation_times_np[-1])
    saveat = dfrx.SaveAt(ts=jnp.asarray(experiment.simulation_times_jax, dtype=jnp.float64))
    condition_matrix = jnp.asarray(experiment.condition_matrix_jax, dtype=jnp.float64)
    rhs = dfrx.ODETerm(ode_system)

    @eqx.filter_jit
    def solve_stats(params):
        def one(y0):
            sol = dfrx.diffeqsolve(
                rhs, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=params, saveat=saveat,
                stepsize_controller=controller, max_steps=max_steps, throw=False,
            )
            s = sol.stats
            return jnp.array(
                [s["num_steps"], s["num_accepted_steps"], s["num_rejected_steps"],
                 (sol.result == dfrx.RESULTS.successful).astype(jnp.int32)],
                dtype=jnp.int32,
            )
        return jax.vmap(one)(condition_matrix)  # (n_conditions, 4)

    bounds = _free_param_bounds(sp)
    free = get_free_parameter_names(sp)
    rng = np.random.default_rng(seed)

    def make_params(free_values: dict[str, float]):
        return tuple(
            jnp.asarray(float(free_values.get(name, param_values[name])), dtype=jnp.float64)
            for name in param_names
        )

    param_sets = [("nominal", make_params({}))]
    if free:
        param_sets.append(("all-lower", make_params({n: bounds[n][0] for n in free})))
        param_sets.append(("all-upper", make_params({n: bounds[n][1] for n in free})))
        for i in range(n_samples):
            draw = {n: float(np.exp(rng.uniform(np.log(bounds[n][0]), np.log(bounds[n][1])))) for n in free}
            param_sets.append((f"prior-draw-{i}", make_params(draw)))

    n_cond = int(condition_matrix.shape[0])
    print(f"Config: {imported.results_save_dir.name}")
    print(f"  solver={ode_cfg.get('solver_name', 'Kvaerno5')}  max_steps={max_steps}  "
          f"rtol={controller_cfg.get('rtol')}  atol={controller_cfg.get('atol')}")
    print(f"  profiling {len(param_sets)} parameter sets x {n_cond} conditions "
          f"({len(param_sets) * n_cond} solves)...")

    stats = np.stack([np.asarray(solve_stats(p)) for _, p in param_sets])  # (n_sets, n_cond, 4)
    num_steps, accepted, rejected, success = stats[..., 0], stats[..., 1], stats[..., 2], stats[..., 3].astype(bool)

    total = num_steps.size
    n_fail = int((~success).sum())
    ok = num_steps[success]

    def pct(a, q):
        return int(np.percentile(a, q)) if a.size else 0

    report = {
        "config": imported.results_save_dir.name,
        "max_steps": max_steps,
        "solver": ode_cfg.get("solver_name", "Kvaerno5"),
        "rtol": controller_cfg.get("rtol"),
        "atol": controller_cfg.get("atol"),
        "n_param_sets": len(param_sets),
        "n_conditions": n_cond,
        "total_solves": total,
        "n_hit_max_steps": n_fail,
        "successful_num_steps": {
            "min": int(ok.min()) if ok.size else None,
            "median": pct(ok, 50),
            "p95": pct(ok, 95),
            "p99": pct(ok, 99),
            "max": int(ok.max()) if ok.size else None,
        },
        "max_rejected_steps": int(rejected.max()),
    }

    print("\n=== ODE Step Profile ===")
    print(f"  total solves: {total}   hit max_steps: {n_fail} ({100 * n_fail / total:.1f}%)")
    if ok.size:
        s = report["successful_num_steps"]
        print(f"  successful num_steps  min={s['min']}  median={s['median']}  "
              f"p95={s['p95']}  p99={s['p99']}  max={s['max']}")
        print(f"  max rejected steps in one solve: {report['max_rejected_steps']}")

    # Which conditions are stiffest (max steps across param sets, per condition).
    per_cond_max = num_steps.max(axis=0)
    worst = np.argsort(per_cond_max)[::-1][:5]
    print("  stiffest conditions (index: max steps across param sets):")
    for c in worst:
        print(f"    condition {int(c)}: {int(per_cond_max[c])} steps")

    # Which parameter set drove the worst solve.
    set_max = num_steps.max(axis=1)
    worst_set = int(np.argmax(set_max))
    print(f"  worst parameter set: '{param_sets[worst_set][0]}' -> {int(set_max[worst_set])} steps")

    print("\n=== Recommendation ===")
    if n_fail > 0:
        print(f"  {n_fail}/{total} solves hit max_steps={max_steps} and FAILED "
              f"(result=max_steps_reached).")
        print("  Those return wrong/NaN trajectories -> NaN gradients -> NUTS divergences.")
        print(f"  RAISE max_steps (e.g. {max_steps * 4}) AND/OR reduce stiffness: check that "
              f"atol/rtol aren't too tight for near-zero species, and verify units/scales.")
        report["recommended_max_steps"] = max_steps * 4
    else:
        worst_steps = int(ok.max())
        rec = int(max(1000, np.ceil(worst_steps * 3 / 100.0) * 100))
        head = max_steps / max(worst_steps, 1)
        print(f"  No solve hit the cap. Worst successful solve used {worst_steps} steps "
              f"(current max_steps={max_steps}, {head:.0f}x headroom).")
        print(f"  A safe setting is ~{rec} (3x the worst observed) to leave room for parameter "
              f"regions the sampler visits beyond this prior-range sample.")
        print("  NOTE: raising max_steps above this is only a safety cap -- adaptive solves stop "
              "at t1 regardless, so it costs a little memory, not runtime.")
        report["recommended_max_steps"] = rec

    out_path = imported.results_save_dir / "ode_step_profile.json"
    imported.results_save_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Wrote {out_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True)
    parser.add_argument("--n-samples", type=int, default=20,
                        help="random prior-range draws to profile, on top of nominal + corners (default 20)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    profile_ode_steps(args.solver_params_file, args.n_samples, args.seed)
