from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import diffrax as dfrx
import equinox as eqx
import jax
import jax.numpy as jnp
import json
import pymc as pm
import preliz as pz

from pytensor.link.jax.dispatch import jax_funcify

from experiment_framework import (
    compute_observation_prediction,
    format_experiment_summary,
    load_experiment_bundle,
    validate_experiment_config,
)
from reaction_model_builder import build_ode_system_from_reactions
from pytensor_ode_ops import SolOp, VJPSolOp
from inference_plotting import (
    plot_inference_diagnostics,
    plot_parameter_marginals,
    plot_posterior_trace_diagnostics,
    plot_predictive,
    plot_priors,
)


_JAXIFY_REGISTERED = False


@dataclass(frozen=True)
class InferenceRunResult:
    posterior: az.InferenceData
    prior_predictive: az.InferenceData
    posterior_predictive: az.InferenceData
    summary: Any
    free_params: list[str]
    species_names: list[str]
    param_names: list[str]
    param_values: dict[str, float]
    solver_params: dict[str, Any]
    experiment: Any
    experiment_summary: str
    prior_file: str
    results_file: str
    loo: Any
    waic: Any
    trace_plot_file: str | None = None
    parameter_density_plot_file: str | None = None
    prior_density_plot_file: str | None = None
    data_vs_results_plot_files: dict[str, str] | None = None
    data_vs_results_legend_plot_files: dict[str, str] | None = None


def get_free_parameter_names(solver_params: dict[str, Any]) -> list[str]:
    """Return configured native-scale free parameter names from solver params."""
    prior_specs = solver_params.get("free_kinetic_params", [])
    free_params = [spec["param_name"] for spec in prior_specs if "param_name" in spec]
    if not free_params:
        raise ValueError("No free parameter names found in solver_params['free_kinetic_params'].")
    return free_params


def summarize_inference_metrics(
    inf_data: az.InferenceData,
    free_params: list[str],
) -> dict[str, Any]:
    """Compute summary, LOO, and WAIC for provided free parameters."""
    available_vars = set(inf_data.posterior.data_vars)
    selected_free_params = [name for name in free_params if name in available_vars]
    if not selected_free_params:
        raise ValueError("None of the requested free parameters were found in posterior variables.")

    summary = az.summary(inf_data, var_names=selected_free_params, round_to=4)
    loo = az.loo(inf_data)
    waic = az.waic(inf_data)
    return {
        "free_params": selected_free_params,
        "summary": summary,
        "loo": loo,
        "waic": waic,
    }


def run_bayesian_inference(
    solver_params_file: str,
    reactions_source: str,
    savedir: str = "./Results",
    prior_samples_file: str = "prior_samples_pm.nc",
    posterior_samples_file: str = "posterior_samples_pm.nc",
    make_trace_plot: bool = True,
    trace_plot_file: str = "trace_plot.png",
    use_log_param_axis: bool = False,
) -> InferenceRunResult:
    """Run full Bayesian inference pipeline from config/reactions and return artifacts."""
    solver_params_path = Path(solver_params_file).expanduser().resolve()
    with open(solver_params_path, "r") as file:
        solver_params = json.load(file)

    savedir_path = Path(savedir).expanduser().resolve()
    savedir_path.mkdir(parents=True, exist_ok=True)

    ode_system, species_names, param_names, param_values = build_ode_system_from_reactions(
        reactions_source
    )

    validate_experiment_config(
        solver_params=solver_params,
        solver_params_file=str(solver_params_path),
        species_names=species_names,
    )

    experiment = load_experiment_bundle(
        solver_params=solver_params,
        solver_params_file=str(solver_params_path),
        species_names=species_names,
    )
    experiment_summary = format_experiment_summary(experiment)

    simulator = _build_simulator(
        ode_system=ode_system,
        species_names=species_names,
        solver_params=solver_params,
        experiment=experiment,
    )
    sol_op = _build_pytensor_sol_op(simulator)

    pm_model, free_params = _build_pymc_model(
        solver_params=solver_params,
        sol_op=sol_op,
        observed_data=experiment.observed_values,
        observed_sigma=experiment.observed_sigma,
    )

    prior_pred = _sample_prior(pm_model=pm_model, solver_params=solver_params)
    posterior = _sample_posterior(pm_model=pm_model, solver_params=solver_params)
    with pm_model:
        post_pred = pm.sample_posterior_predictive(posterior, progressbar=False)

    prior_file = savedir_path / prior_samples_file
    results_file = savedir_path / posterior_samples_file
    _safe_write_idata(prior_pred, prior_file)

    combined = posterior.copy()
    combined.extend(prior_pred)
    combined.extend(post_pred)
    _safe_write_idata(combined, results_file)

    metrics = summarize_inference_metrics(inf_data=combined, free_params=free_params)
    summary = metrics["summary"]
    free_params = metrics["free_params"]
    loo = metrics["loo"]
    waic = metrics["waic"]

    trace_plot_path = None
    parameter_density_plot_file = None
    prior_density_plot_file = None
    data_vs_results_plot_files = None
    data_vs_results_legend_plot_files = None
    if make_trace_plot:
        trace_plot_path = savedir_path / trace_plot_file
        plot_artifacts = plot_inference_diagnostics(
            inf_data=combined,
            free_params=free_params,
            experiment=experiment,
            save_path=str(trace_plot_path),
            show=False,
            use_log_param_axis=use_log_param_axis,
        )
        parameter_density_plot_file = plot_artifacts.get("parameter_density_plot_file")
        prior_density_plot_file = plot_artifacts.get("prior_density_plot_file")
        data_vs_results_plot_files = plot_artifacts.get("data_vs_results_plot_files")
        data_vs_results_legend_plot_files = plot_artifacts.get(
            "data_vs_results_legend_plot_files"
        )

    return InferenceRunResult(
        posterior=combined,
        prior_predictive=prior_pred,
        posterior_predictive=post_pred,
        summary=summary,
        free_params=free_params,
        species_names=species_names,
        param_names=param_names,
        param_values=param_values,
        solver_params=solver_params,
        experiment=experiment,
        experiment_summary=experiment_summary,
        prior_file=str(prior_file),
        results_file=str(results_file),
        loo=loo,
        waic=waic,
        trace_plot_file=str(trace_plot_path) if trace_plot_path is not None else None,
        parameter_density_plot_file=parameter_density_plot_file,
        prior_density_plot_file=prior_density_plot_file,
        data_vs_results_plot_files=data_vs_results_plot_files,
        data_vs_results_legend_plot_files=data_vs_results_legend_plot_files,
    )


def _build_simulator(ode_system, species_names, solver_params, experiment):
    ode_solver_config = solver_params.get("ODE_solver", {})
    solver_name = ode_solver_config.get("solver_name", "Kvaerno5")
    dt0 = float(ode_solver_config.get("dt0", 1.0e-12))
    max_steps = int(ode_solver_config.get("max_steps", 10_000_000))

    ode_controller_config = solver_params.get("ODE_stepsize_controller", {})

    solver = getattr(dfrx, solver_name)()
    stepsize_controller = dfrx.PIDController(**ode_controller_config)
    t0 = 0.0
    t1 = float(experiment.simulation_times_np[-1])
    saveat = dfrx.SaveAt(ts=experiment.simulation_times_jax)

    condition_matrix = experiment.condition_matrix_jax
    rhs = dfrx.ODETerm(ode_system)

    def solve_single_initial_condition(y0_local, params):
        sol = dfrx.diffeqsolve(
            rhs,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0_local,
            args=params,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            throw=True,
        )
        concentrations = jnp.asarray(sol.ys)
        if concentrations.ndim == 1:
            concentrations = concentrations[jnp.newaxis, :]
        return concentrations

    def simulator(params):
        n_conditions = condition_matrix.shape[0]
        if n_conditions == 1:
            all_concentrations = solve_single_initial_condition(
                condition_matrix[0], params
            )[jnp.newaxis, :, :]
        else:
            all_concentrations = jax.vmap(
                lambda y0_local: solve_single_initial_condition(y0_local, params)
            )(condition_matrix)

        return compute_observation_prediction(
            experiment=experiment,
            concentrations=all_concentrations,
            species_names=species_names,
        )

    return simulator


def _build_pytensor_sol_op(simulator):
    _register_jaxify_handlers()

    def sol_op_jax(*params):
        return simulator(params)

    sol_op_jax_jitted = eqx.filter_jit(sol_op_jax)

    def vjp_sol_op_jax(gz, *params):
        _, vjp_fn = jax.vjp(sol_op_jax, *params)
        return vjp_fn(gz)

    vjp_sol_op_jax_jitted = eqx.filter_jit(vjp_sol_op_jax)

    vjp_sol_op = VJPSolOp(vjp_sol_op_jax_jitted)
    return SolOp(sol_op_jax_jitted, vjp_sol_op)


def _register_jaxify_handlers():
    global _JAXIFY_REGISTERED
    if _JAXIFY_REGISTERED:
        return

    @jax_funcify.register(SolOp)
    def sol_op_jax_funcify(op, **kwargs):
        return op.sol_op_jax_jitted

    @jax_funcify.register(VJPSolOp)
    def vjp_sol_op_jax_funcify(op, **kwargs):
        return op.vjp_sol_op_jax_jitted

    _JAXIFY_REGISTERED = True


def _build_pymc_model(solver_params, sol_op, observed_data, observed_sigma):
    prior_specs = solver_params.get("free_kinetic_params", [])
    if not prior_specs:
        raise ValueError("No 'free_kinetic_params' configured in solver_params.json")

    ordered_param_names = [spec["param_name"] for spec in prior_specs]

    with pm.Model() as pm_model:
        priors = {}
        for param_spec in prior_specs:
            param_name = param_spec["param_name"]
            prior_params = param_spec["prior_dist_params"]

            dist_name = prior_params["distribution"]
            dist = getattr(pz, dist_name)()
            result = pz.maxent(
                distribution=dist,
                lower=prior_params["lower"],
                upper=prior_params["upper"],
                mass=prior_params.get("mass", 0.95),
                fixed_stat=prior_params.get("fixed_stat",None),
                plot=False
            )
            pm_dist_cls = getattr(pm, result.__class__.__name__)
            priors[param_name] = pm_dist_cls(
                param_name,
                *[float(value) for value in result.params],
            )

        prediction = pm.Deterministic(
            "prediction",
            sol_op(*[priors[param] for param in ordered_param_names]),
        )
        pm.Normal(
            "llike",
            mu=prediction,
            sigma=observed_sigma,
            observed=observed_data,
        )

    return pm_model, ordered_param_names


def _sample_prior(pm_model, solver_params):
    prior_config = solver_params.get("prior_sampling", {})
    draws = int(prior_config.get("draws", 1000))
    random_seed = prior_config.get("random_seed", 0)
    with pm_model:
        return pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
        )


def _sample_posterior(pm_model, solver_params):
    posterior_config = solver_params.get("posterior_sampling", {})

    draws = int(posterior_config.get("draws", 1000))
    tune = int(posterior_config.get("tune", 1000))
    chains = int(posterior_config.get("chains", 4))
    cores_raw = posterior_config.get("cores", None)
    if cores_raw in (None, "None"):
        cores = None
    else:
        cores = int(cores_raw)

    random_seed = posterior_config.get("random_seed", 0)
    nuts_sampler = posterior_config.get("nuts_sampler", None)

    sample_kwargs = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "random_seed": random_seed,
    }
    if nuts_sampler is not None:
        sample_kwargs["nuts_sampler"] = nuts_sampler

    with pm_model:
        posterior = pm.sample(**sample_kwargs)
        pm.compute_log_likelihood(posterior, model=pm_model, progressbar=False)
    return posterior


def _safe_write_idata(inf_data: az.InferenceData, output_file: Path):
    if output_file.exists():
        output_file.unlink()
    inf_data.to_netcdf(output_file)
