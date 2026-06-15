from __future__ import annotations

import multiprocessing
import os
import sys

# Force forkserver on macOS to prevent deadlocks between PyMC and JAX
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

cores = multiprocessing.cpu_count()
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_num_cpu_devices", cores) # This might be different on not my computer - try "jax_xla_backend_host_device_count" instead if it doesn't work

device_count = jax.local_device_count()
print(f"Total JAX local devices initialized: {device_count}")

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect
import json
import yaml
import numpy as np
import warnings

import arviz as az
import diffrax as dfrx
import equinox as eqx
import jax.numpy as jnp
import lineax
import pymc as pm
import preliz as pz
from pytensor.link.jax.dispatch import jax_funcify

# Suppress known harmless PyTensor/Numba object-mode fallback warnings.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pytensor\.link\.numba\.dispatch\.basic",
    message=r"Numba will use object mode.*",
)

from experiment_framework import (
    compute_observation_prediction,
    format_experiment_summary,
    load_experiment_bundle,
    validate_experiment_config,
)
from reaction_model_builder import build_ode_system_from_reactions
from pytensor_ode_ops import SolOp, VJPSolOp
from inference_plotting import plot_inference_diagnostics

_JAXIFY_REGISTERED = False


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_kv(label: str, value: Any) -> None:
    print(f"- {label}: {value}")

def _build_pytensor_sol_op(simulator):
    _register_jaxify_handlers()

    def sol_op_jax(*params):
        params = tuple(jnp.asarray(param, dtype=jnp.float64) for param in params)
        return simulator(params)

    sol_op_jax_jitted = eqx.filter_jit(sol_op_jax)

    def vjp_sol_op_jax(gz, *params):
        gz = jnp.asarray(gz, dtype=jnp.float64)
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

_register_jaxify_handlers()

@dataclass(frozen=True)
class InferenceRunResult:
    posterior: az.InferenceData
    prior_predictive: az.InferenceData
    posterior_predictive: az.InferenceData | None
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

CONDITION_MATRIX = None
EXPERIMENT = None


def _resolve_solver_relative_path(base_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _get_solver_path_base_dir(solver_params: dict[str, Any], solver_params_file: str | Path) -> Path:
    solver_params_dir = Path(solver_params_file).resolve().parent
    return _resolve_solver_relative_path(solver_params_dir, solver_params.get("path_base", "."))


def run_bayesian_inference(
    solver_params_file: str,
    reactions_source: str,
    savedir: str = "Results",
    prior_samples_file: str = "prior_samples_pm.nc",
    posterior_samples_file: str = "posterior_samples_pm.nc",
    make_trace_plot: bool = True,
    trace_plot_file: str = "trace_plot.png",
    use_log_param_axis: bool = False,
) -> InferenceRunResult:
    """Run full Bayesian inference pipeline from config/reactions and return artifacts."""
    solver_params_path = Path(solver_params_file).expanduser().resolve()
    with open(solver_params_path, "r") as file:
        if solver_params_path.suffix in (".yaml", ".yml"):
            solver_params = yaml.safe_load(file)
        elif solver_params_path.suffix == ".json":
            solver_params = json.load(file)
        else:
            raise ValueError(f"Unsupported solver params file format: {solver_params_path.suffix}")

    base_dir = _get_solver_path_base_dir(solver_params, solver_params_path)
    reactions_source_path = _resolve_solver_relative_path(base_dir, reactions_source)
    savedir_path = _resolve_solver_relative_path(base_dir, savedir)
    savedir_path.mkdir(parents=True, exist_ok=True)

    ode_system, species_names, param_names, param_values, scaling_params = build_ode_system_from_reactions(
        reactions_source_path
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
    _print_section("Experiment Summary")
    print(experiment_summary)

    global CONDITION_MATRIX, EXPERIMENT
    CONDITION_MATRIX = experiment.condition_matrix_jax
    EXPERIMENT = experiment

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
        param_names=param_names,
        param_values=param_values,
        scaling_params=scaling_params,
        observed_data=experiment.observed_values,
        observed_sigma=experiment.observed_sigma,
    )

    _print_section("Sampling")
    print("Running prior predictive sampling...")
    prior_pred = _sample_prior(
        pm_model=pm_model,
        solver_params=solver_params,
        free_params=free_params,
    )

    print("Running posterior MCMC sampling...")
    posterior = _sample_posterior(
        pm_model=pm_model,
        solver_params=solver_params
    )

    print("Running posterior predictive simulation...")
    with pm_model:
        post_pred = pm.sample_posterior_predictive(posterior, progressbar=False)

    _print_section("Artifacts")
    print("Combining inference groups and writing NetCDF outputs...")
    prior_file = savedir_path / prior_samples_file
    results_file = savedir_path / posterior_samples_file
    _safe_write_idata(prior_pred, prior_file)

    combined = posterior.copy()
    combined.extend(prior_pred)
    combined.extend(post_pred)
    _safe_write_idata(combined, results_file)
    _print_kv("Prior file", prior_file.name)
    _print_kv("Posterior file", results_file.name)

    _print_section("Metrics")
    print("Computing posterior summary statistics...")
    metrics = summarize_inference_metrics(
        inf_data=combined,
        free_params=free_params,
    )
    summary = metrics["summary"]
    free_params = metrics["free_params"]
    loo = metrics["loo"]
    waic = metrics["waic"]

    _print_section("Posterior Summary")
    print(summary)
    _print_section("Model Comparison")
    print("LOO")
    print(loo)
    print("WAIC")
    print(waic)

    trace_plot_path = None
    parameter_density_plot_file = None
    prior_density_plot_file = None
    data_vs_results_plot_files = None
    data_vs_results_legend_plot_files = None
    if make_trace_plot:
        _print_section("Diagnostic Plots")
        print("Generating trace diagnostics and predictive plots...")
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
        print("Finished rendering diagnostic views.")

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


def _build_solver(solver_params):
    """Construct the diffrax solver, optionally with a robust (ill-conditioning
    tolerant) implicit root finder.

    Stiff implicit solvers (e.g. Kvaerno5) solve a linear system involving the
    RHS Jacobian at every step. When that Jacobian is singular / very
    ill-conditioned, the default linear solver returns NaN/inf and the solve
    fails. Setting ``robust_linear_solver: true`` in the ODE_solver config
    swaps in ``lineax.AutoLinearSolver(well_posed=False)``, which falls back to
    a least-squares solve instead of producing non-finite output.
    """
    ode_solver_config = solver_params.get("ODE_solver", {})
    solver_name = ode_solver_config.get("solver_name", "Kvaerno5")
    solver_cls = getattr(dfrx, solver_name)

    if not ode_solver_config.get("robust_linear_solver", False):
        return solver_cls()

    controller_config = solver_params.get("ODE_stepsize_controller", {})
    rtol = float(controller_config.get("rtol", 1e-4))
    atol = float(controller_config.get("atol", 1e-6))
    root_finder = dfrx.VeryChord(
        rtol=rtol,
        atol=atol,
        linear_solver=lineax.AutoLinearSolver(well_posed=False),
    )
    return solver_cls(root_finder=root_finder)


def _build_simulator(ode_system, species_names, solver_params, experiment):
    ode_solver_config = solver_params.get("ODE_solver", {})
    dt0 = ode_solver_config.get("dt0", None)
    max_steps = int(ode_solver_config.get("max_steps", 10_000))

    ode_controller_config = solver_params.get("ODE_stepsize_controller", {})

    solver = _build_solver(solver_params)
    stepsize_controller = dfrx.PIDController(**ode_controller_config)

    t0 = 0.0
    t1 = float(experiment.simulation_times_np[-1])

    saveat = dfrx.SaveAt(
        steps=False,
        ts=jnp.asarray(experiment.simulation_times_jax, dtype=jnp.float64),
    )

    condition_matrix = jnp.asarray(experiment.condition_matrix_jax, dtype=jnp.float64)
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

        ys = sol.ys
        if ys.ndim == 1:
            ys = ys[jnp.newaxis, :]
        return ys

    def solve_all_conditions(condition_matrix, params):
        return jax.vmap(
            lambda y0: solve_single_initial_condition(y0, params)
        )(condition_matrix)

    def simulator(params):

        all_concentrations = solve_all_conditions(condition_matrix, params)

        result = compute_observation_prediction(
            experiment=experiment,
            concentrations=all_concentrations,
            species_names=species_names
        )

        return jnp.asarray(result, dtype=jnp.float64)

    simulator = eqx.filter_jit(simulator)

    return simulator


def _build_pymc_model(
    solver_params,
    sol_op,
    param_names,
    param_values,
    scaling_params,
    observed_data,
    observed_sigma,
):
    prior_specs = solver_params.get("free_kinetic_params", [])
    if not prior_specs:
        raise ValueError("No 'free_kinetic_params' configured in solver_params config")

    ordered_param_names = [spec["param_name"] for spec in prior_specs]
    missing_free_params = [name for name in ordered_param_names if name not in param_names]
    if missing_free_params:
        scaling_param_set = set(scaling_params or [])
        rate_params = [name for name in param_names if name not in scaling_param_set]
        raise ValueError(
            "Configured free parameters are missing from the reaction-model parameter vector: "
            f"{missing_free_params}. "
            f"Available rate parameters: {rate_params}. "
            f"Available scaling parameters: {list(scaling_params or [])}."
        )

    missing_nominal_values = [name for name in param_names if name not in param_values]
    if missing_nominal_values:
        raise ValueError(
            "Missing nominal parameter values for reaction-model parameters: "
            f"{missing_nominal_values}."
        )

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
                fixed_stat=prior_params.get("fixed_stat", None),
                plot=False
            )
            pm_dist_cls = getattr(pm, result.__class__.__name__)
            priors[param_name] = pm_dist_cls(
                param_name,
                *[float(value) for value in result.params],
            )

        # Build full parameter tuple in reaction-defined order. Free parameters are
        # sampled; all other parameters are fixed to their nominal values.
        full_parameter_vector = [
            priors[name] if name in priors else float(param_values[name])
            for name in param_names
        ]

        prediction = pm.Deterministic(
            "prediction",
            sol_op(*full_parameter_vector),
        )
        pm.Normal(
            "llike",
            mu=prediction,
            sigma=observed_sigma,
            observed=observed_data,
        )

    return pm_model, ordered_param_names


def _sample_prior(pm_model, solver_params, free_params):
    prior_config = solver_params.get("prior_sampling", {})
    draws = int(prior_config.get("draws", 1000))
    random_seed = prior_config.get("random_seed", 0)

    # Fast default: sample only prior parameter variables.
    # Set include_prediction/include_likelihood true in prior_sampling config
    # when full prior predictive trajectories/likelihood draws are needed.
    include_prediction = bool(prior_config.get("include_prediction", False))
    include_likelihood = bool(prior_config.get("include_likelihood", False))

    var_names = list(free_params)
    if include_prediction:
        var_names.append("prediction")
    if include_likelihood:
        var_names.append("llike")

    with pm_model:
        return pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
            var_names=var_names,
        )


def _sample_posterior(pm_model, solver_params):
    posterior_config = solver_params.get("posterior_sampling", {})

    sample_signature = inspect.signature(pm.sample)
    accepted_keys = set(sample_signature.parameters)

    sample_kwargs = {
        key: value
        for key, value in posterior_config.items()
        if key in accepted_keys and value is not None
    }

    if sample_kwargs.get("nuts_sampler", None) == "numpyro":
        sample_kwargs.update({"chain_method": "vectorized"})

    sample_kwargs["chains"] = int(sample_kwargs.get("chains", cores))

    if sample_kwargs.get("cores") is None:
        sample_kwargs["cores"] = min(cores, sample_kwargs["chains"])

    with pm_model:
        posterior = pm.sample(**sample_kwargs)
        print("Computing log-likelihood values...")
        pm.compute_log_likelihood(posterior, model=pm_model, progressbar=False)
    return posterior


def _safe_write_idata(inf_data: az.InferenceData, output_file: Path):
    def _netcdf_safe_attr(value):
        if isinstance(value, (str, bytes, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return json.dumps({str(key): _netcdf_safe_attr(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            safe_items = [_netcdf_safe_attr(item) for item in value]
            if all(isinstance(item, (str, bytes, int, float, bool)) or item is None for item in safe_items):
                return safe_items
            return json.dumps(safe_items)
        return str(value)

    for group_name in inf_data.groups():
        group = getattr(inf_data, group_name)
        group.attrs = {key: _netcdf_safe_attr(value) for key, value in group.attrs.items()}

    inf_data.attrs = {key: _netcdf_safe_attr(value) for key, value in inf_data.attrs.items()}

    if output_file.exists():
        output_file.unlink()
    inf_data.to_netcdf(output_file)
