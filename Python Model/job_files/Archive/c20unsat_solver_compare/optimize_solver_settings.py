"""Find PID/tolerance settings for the C20+unsat baseline condition, trading off
step count against accuracy relative to a tight-tolerance reference solve.

PID candidates are diffrax's OWN documented recommendations (PIDController
docstring), not arbitrary: the current inference config (0.3,0.3,0) and the
current data-generation config (0.2,0.4,0) are both already members of
diffrax's "moderate difficulty / mildly stiff" recommended set -- this sweep
adds the third member of that set plus the smooth-problem and SDE-style
brackets, to see if either currently-used choice is actually the best member
of the set it was drawn from.
"""
import argparse
import json
import sys
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
import equinox as eqx

from reaction_model_builder import build_ode_system_from_reactions, set_scaling_group_values
from inference_runner import import_solver_params
from experiment_framework import load_experiment_bundle, validate_experiment_config
import generate_chain_data as gcd

PATH = "/projects/anth4580/Bayesian/Results/Chain Scaling Tests/Chain C20+unsat - a1/solver_params.json"

ap = argparse.ArgumentParser()
ap.add_argument("--pid_index", type=int, required=True, help="0-4, selects one PID_CANDIDATES entry")
a = ap.parse_args()

# diffrax's own documented recommendations (PIDController docstring):
PID_CANDIDATES = [
    ("current inference (0.3,0.3,0)", 0.3, 0.3, 0.0),
    ("current data-gen (0.2,0.4,0)", 0.2, 0.4, 0.0),
    ("third moderate-difficulty option (0.4,0.3,0)", 0.4, 0.3, 0.0),
    ("SDE-style insensitive (0.1,0.3,0)", 0.1, 0.3, 0.0),
    ("diffrax default I-controller (0,1,0)", 0.0, 1.0, 0.0),
]

TOL_CANDIDATES = [
    (f"rtol={rtol:.0e} atol={atol:.0e}", rtol, atol)
    for rtol in (1e-3, 1e-4, 1e-5, 1e-6)
    for atol in (1e-6, 1e-7, 1e-8)
]

REFERENCE_RTOL, REFERENCE_ATOL = 1e-10, 1e-12
GROUND_TRUTH_ERROR_FLOOR = 1e-6  # ignore relative error on species below this absolute concentration

imported = import_solver_params(PATH)
sp = imported.solver_params
ode_system, species_names, param_names, param_values, scaling_groups = build_ode_system_from_reactions(imported.reactions_source)
validate_experiment_config(solver_params=sp, solver_params_file=str(imported.solver_params_file), species_names=species_names)
experiment = load_experiment_bundle(solver_params=sp, solver_params_file=str(imported.solver_params_file), species_names=species_names)

t0, t1 = 0.0, float(experiment.simulation_times_np[-1])
saveat = dfrx.SaveAt(ts=jnp.asarray(experiment.simulation_times_jax, dtype=jnp.float64))
y0 = jnp.asarray(experiment.condition_matrix_jax[0], dtype=jnp.float64)  # baseline condition
solver = dfrx.Kvaerno5()
rhs = dfrx.ODETerm(ode_system)
theta_raw = jnp.array([param_values[name] for name in param_names], dtype=jnp.float64)
nominal_overrides = gcd.nominal_scaling_group_overrides(scaling_groups)
print(f"applying nominal scaling-group overrides (d-type=0, else=1): {nominal_overrides}")
theta = tuple(set_scaling_group_values(theta_raw, param_names, nominal_overrides))  # a1=1, all nominal, d1=d2=0


def solve_with(rtol, atol, pcoeff, icoeff, dcoeff, max_steps=200_000, dt0=1e-6):
    controller = dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    sol = dfrx.diffeqsolve(rhs, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=theta,
                            saveat=saveat, stepsize_controller=controller, max_steps=max_steps, throw=False)
    return sol


print("Computing ground-truth reference solve (very tight tolerance)...")
ref_sol = solve_with(REFERENCE_RTOL, REFERENCE_ATOL, 0.4, 0.3, 0.0, max_steps=1_000_000)
ref_ok = bool(ref_sol.result == dfrx.RESULTS.successful) and int(ref_sol.stats["num_steps"]) < 1_000_000
print(f"  reference: steps={int(ref_sol.stats['num_steps'])}  successful={ref_ok}")
if not ref_ok:
    raise SystemExit("Reference solve itself did not converge -- loosen REFERENCE_RTOL/ATOL or raise max_steps.")
ref_ys = np.asarray(ref_sol.ys)  # (n_times, n_species)


def relative_error(ys):
    ys = np.asarray(ys)
    mask = np.abs(ref_ys) > GROUND_TRUTH_ERROR_FLOOR
    if not mask.any():
        return float("nan")
    rel = np.abs(ys[mask] - ref_ys[mask]) / np.abs(ref_ys[mask])
    return float(np.max(rel))


pid_name, pcoeff, icoeff, dcoeff = PID_CANDIDATES[a.pid_index]
print(f"\nTesting PID candidate [{a.pid_index}]: {pid_name}")
print(f"{'PID':45s} {'tol':22s} {'steps':>7s} {'rejected':>9s} {'success':>8s} {'max_rel_err':>12s}")
results = []
for tol_name, rtol, atol in TOL_CANDIDATES:
    sol = solve_with(rtol, atol, pcoeff, icoeff, dcoeff)
    steps = int(sol.stats["num_steps"])
    rejected = int(sol.stats["num_rejected_steps"])
    success = bool(sol.result == dfrx.RESULTS.successful) and steps < 200_000
    err = relative_error(sol.ys) if success else float("nan")
    results.append(dict(pid_name=pid_name, tol_name=tol_name, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                        rtol=rtol, atol=atol, steps=steps, rejected=rejected, success=success,
                        max_rel_err=err))
    print(f"{pid_name:45s} {tol_name:22s} {steps:7d} {rejected:9d} {str(success):>8s} {err:12.2e}")

out_path = f"/projects/anth4580/Bayesian/job_files/c20unsat_solver_compare/pid_sweep_results/pid{a.pid_index}.json"
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(dict(reference_steps=int(ref_sol.stats["num_steps"]), results=results), f, indent=2)
print(f"\nWrote {out_path}")
