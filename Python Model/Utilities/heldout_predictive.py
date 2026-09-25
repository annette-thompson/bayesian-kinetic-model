"""Score a fitted posterior against a HELD-OUT dataset it was never fit to.

This is the comparison method Section 3.6 (data-informativeness) needs: LOO
is invalid across differently-scoped datasets (a timeseries-only fit and an
endpoint-only fit have different observations, so their elpd_loo values are
not comparable), but scoring every measurement-type variant's posterior
against the SAME held-out set puts them on equal footing.

Reuses the exact same forward-solve + observable-mapping path the real
likelihood uses (compute_observation_prediction), evaluated at posterior
draws rather than traced for gradients -- this is a plain forward pass, not
part of the inference model itself.

Usage (from the "Python Model" directory):
    python Utilities/heldout_predictive.py \
        --fitted_solver_params_file "Results/.../variant-A/solver_params.json" \
        --heldout_solver_params_file "Results/.../heldout_data/solver_params.json" \
        --n_draws 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfrx
import equinox as eqx
from scipy.special import logsumexp

from reaction_model_builder import build_ode_system_from_reactions
from inference_runner import _build_solver, get_free_parameter_names, import_solver_params
from experiment_framework import (
    compute_observation_prediction,
    load_experiment_bundle,
    validate_experiment_config,
)
import arviz as az


def _make_solver_fn(solver_params: dict, reactions_source, param_names: list[str]):
    """Returns solve(full_param_vector) -> (n_conditions, n_times, n_species),
    for the HELD-OUT experiment's own conditions/times/reactions."""
    ode_system, species_names, _, _, _ = build_ode_system_from_reactions(
        reactions_source, scaling_group=solver_params.get("scaling_groups"))
    validate_experiment_config(solver_params=solver_params,
                               solver_params_file="<heldout>", species_names=species_names)
    experiment = load_experiment_bundle(solver_params=solver_params,
                                        solver_params_file="<heldout>", species_names=species_names)

    ode_cfg = solver_params.get("ODE_solver", {})
    controller_cfg = solver_params.get("ODE_stepsize_controller", {})
    dt0 = ode_cfg.get("dt0", None)
    max_steps = int(ode_cfg.get("max_steps", 20_000))
    solver = _build_solver(solver_params)
    controller = dfrx.PIDController(**controller_cfg)
    t0, t1 = 0.0, float(experiment.simulation_times_np[-1])
    saveat = dfrx.SaveAt(ts=jnp.asarray(experiment.simulation_times_jax, dtype=jnp.float64))
    condition_matrix = jnp.asarray(experiment.condition_matrix_jax, dtype=jnp.float64)
    rhs = dfrx.ODETerm(ode_system)

    @eqx.filter_jit
    def solve(params):
        def one(y0):
            sol = dfrx.diffeqsolve(rhs, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=params,
                                    saveat=saveat, stepsize_controller=controller,
                                    max_steps=max_steps, throw=False)
            return sol.ys
        trajectories = jax.vmap(one)(condition_matrix)  # (n_conditions, n_times, n_species)
        return compute_observation_prediction(experiment, trajectories, species_names)

    return solve, experiment, species_names


def score_posterior_on_heldout(
    fitted_solver_params_file: str,
    heldout_solver_params_file: str,
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    fitted = import_solver_params(fitted_solver_params_file)
    fitted_sp = fitted.solver_params
    nc_path = fitted.results_save_dir / fitted.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")
    inf_data = az.from_netcdf(nc_path)
    free_names = get_free_parameter_names(fitted_sp)

    heldout = import_solver_params(heldout_solver_params_file)
    heldout_sp = heldout.solver_params
    _, _, param_names, param_values, _ = build_ode_system_from_reactions(
        heldout.reactions_source, scaling_group=heldout_sp.get("scaling_groups"))
    missing = [n for n in free_names if n not in param_names]
    if missing:
        raise SystemExit(f"Fitted free params {missing} do not exist in the held-out model's "
                          "reaction set -- the two configs must share the same underlying network.")

    solve, experiment, _ = _make_solver_fn(heldout_sp, heldout.reactions_source, param_names)
    observed = jnp.asarray(experiment.observed_values.reshape(-1), dtype=jnp.float64)
    sigma = jnp.asarray(experiment.observed_sigma.reshape(-1), dtype=jnp.float64)

    stacked = {name: np.asarray(inf_data.posterior[name].values, dtype=float).reshape(-1) for name in free_names}
    n_available = min(len(v) for v in stacked.values())
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_available, size=min(n_draws, n_available), replace=False)

    log_liks = []
    for i in idx:
        full_params = tuple(
            jnp.asarray(float(stacked[name][i]) if name in stacked else param_values[name], dtype=jnp.float64)
            for name in param_names
        )
        prediction = solve(full_params)
        # Normal log-density, elementwise, matching _build_pymc_model's own likelihood exactly.
        resid = (observed - prediction) / sigma
        log_lik = float(jnp.sum(-0.5 * resid**2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)))
        log_liks.append(log_lik)

    log_liks = np.asarray(log_liks)
    n = log_liks.size
    # Posterior predictive log-density: log(mean(likelihood)) via logsumexp, not mean(log-likelihood) --
    # these differ, and the former is the quantity that's actually comparable to elpd.
    heldout_lpd = float(logsumexp(log_liks) - np.log(n))

    return {
        "fitted_run": str(fitted.results_save_dir),
        "heldout_config": str(heldout.solver_params_file),
        "free_params": free_names,
        "n_draws_used": int(n),
        "per_draw_log_lik": log_liks,
        "heldout_log_predictive_density": heldout_lpd,
        "heldout_log_predictive_density_se": float(np.std(log_liks) / np.sqrt(n)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fitted_solver_params_file", required=True,
                        help="The config+posterior netcdf of the run being scored.")
    parser.add_argument("--heldout_solver_params_file", required=True,
                        help="A config pointing at the held-out data to score against "
                        "(same reactions_source/model, different data_file(s)).")
    parser.add_argument("--n_draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = score_posterior_on_heldout(
        args.fitted_solver_params_file, args.heldout_solver_params_file, args.n_draws, args.seed
    )
    print(f"Fitted run: {result['fitted_run']}")
    print(f"Scored against: {result['heldout_config']}")
    print(f"Free params: {result['free_params']}")
    print(f"Draws used: {result['n_draws_used']}")
    print(f"Held-out log predictive density: {result['heldout_log_predictive_density']:.3f} "
          f"+/- {result['heldout_log_predictive_density_se']:.3f} (SE)")


if __name__ == "__main__":
    main()
