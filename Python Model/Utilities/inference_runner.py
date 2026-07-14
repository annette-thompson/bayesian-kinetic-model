from __future__ import annotations

import multiprocessing
import os
import sys
import time

if sys.platform in ("darwin", "linux"):
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
os.environ.setdefault("EQX_ON_ERROR", "nan")

import jax
jax.config.update("jax_enable_x64", True)

device_count = jax.local_device_count()
print(f"Total JAX local devices initialized: {device_count}")

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import yaml

import arviz as az
import diffrax as dfrx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pymc as pm
import preliz as pz
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

from experiment_framework import (
    compute_observation_prediction,
    format_experiment_summary,
    load_experiment_bundle,
    validate_experiment_config,
)
from reaction_model_builder import build_ode_system_from_reactions

_JAXIFY_REGISTERED = False


class SolOp(Op):
    """Pytensor Op for ODE simulation output with an attached gradient Op.

    Only used so ``pm.Model()`` can build a graph around the ODE solve;
    everything BlackJAX actually samples goes through ``get_jaxified_logp`` ->
    ``jax_funcify``, which substitutes ``sol_op_jax_jitted`` directly and never
    calls ``perform``/``pullback`` below.
    """

    def __init__(self, sol_op_jax_jitted, vjp_sol_op):
        self.sol_op_jax_jitted = sol_op_jax_jitted
        self.vjp_sol_op = vjp_sol_op

    def make_node(self, *inputs):
        # Keep all values in float64 to avoid mixed-precision instability in
        # JAX callbacks and VJP computation.
        inputs = [pt.cast(pt.as_tensor_variable(inp), "float64") for inp in inputs]
        outputs = [pt.matrix(dtype="float64")]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result = self.sol_op_jax_jitted(*inputs)
        result_arr = np.asarray(result, dtype="float64")
        if not np.isfinite(result_arr).all():
            raise FloatingPointError(
                "Non-finite values returned by ODE forward solve in SolOp.perform."
            )
        outputs[0][0] = result_arr

    def pullback(self, inputs, outputs, cotangents):
        (gz,) = cotangents
        return self.vjp_sol_op(*inputs, gz)


class VJPSolOp(Op):
    """Pytensor Op for vector-Jacobian products of the ODE simulation Op."""

    def __init__(self, vjp_sol_op_jax_jitted):
        self.vjp_sol_op_jax_jitted = vjp_sol_op_jax_jitted

    def make_node(self, *inputs):
        if len(inputs) < 2:
            raise ValueError("VJPSolOp expects parameter inputs followed by output gradient gz.")

        tensor_inputs = [pt.cast(pt.as_tensor_variable(inp), "float64") for inp in inputs]
        *params, gz = tensor_inputs
        outputs = [pt.tensor(dtype="float64", shape=param.type.shape) for param in params]
        return Apply(self, [*params, gz], outputs)

    def perform(self, node, inputs, outputs):
        *params, gz = inputs
        if any(not np.isfinite(np.asarray(param)).all() for param in params):
            raise FloatingPointError(
                "Non-finite parameter values reached VJPSolOp.perform before JAX VJP evaluation."
            )
        if not np.isfinite(np.asarray(gz)).all():
            raise FloatingPointError(
                "Non-finite output gradient reached VJPSolOp.perform before JAX VJP evaluation."
            )

        result = self.vjp_sol_op_jax_jitted(gz, *params)
        for i, res in enumerate(result):
            res_arr = np.asarray(res, dtype="float64")
            if not np.isfinite(res_arr).all():
                raise FloatingPointError(
                    "Non-finite values returned by ODE VJP solve in VJPSolOp.perform."
                )
            outputs[i][0] = res_arr

def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_kv(label: str, value: Any) -> None:
    print(f"- {label}: {value}")


def _print_config_values(title: str, config: dict[str, Any], keys: list[str]) -> None:
    print(title)
    for key in keys:
        if key in config:
            _print_kv(key, config[key])


def _print_run_configuration(imported: ImportedSolverParams) -> None:
    solver_params = imported.solver_params
    _print_section("Run Configuration")
    _print_kv("Solver params file", imported.solver_params_file)
    _print_kv("Path base", imported.path_base_dir)
    _print_kv("Reactions source", imported.reactions_source)
    _print_kv("Results directory", imported.results_save_dir)
    _print_kv("Prior samples file", imported.prior_samples_file)
    _print_kv("Posterior samples file", imported.posterior_samples_file)
    _print_kv("Free kinetic params", get_free_parameter_names(solver_params))

    _print_config_values(
        "Prior sampling:",
        solver_params.get("prior_sampling", {}),
        ["draws", "samples", "random_seed", "include_prediction", "include_likelihood"],
    )
    _print_config_values(
        "Posterior sampling:",
        solver_params.get("posterior_sampling", {}),
        [
            "draws",
            "tune",
            "chains",
            "cores",
            "target_accept",
            "nuts_sampler",
            "chain_method",
            "random_seed",
            "init",
        ],
    )
    _print_config_values(
        "ODE solver:",
        solver_params.get("ODE_solver", {}),
        ["solver_name", "dt0", "max_steps"],
    )
    _print_config_values(
        "ODE stepsize controller:",
        solver_params.get("ODE_stepsize_controller", {}),
        ["rtol", "atol", "pcoeff", "icoeff", "dcoeff"],
    )

def _build_pytensor_sol_op(simulator):
    _register_jaxify_handlers()

    def _forward_raw(*params):
        params = tuple(jnp.asarray(p, dtype=jnp.float64) for p in params)
        return simulator(params)

    @jax.custom_vjp
    def sol_op_jax(*params):
        return _forward_raw(*params)

    def _sol_fwd(*params):
        # jax.vjp computes the primal AND the pullback closure in one pass, so
        # save the closure as the residual instead of the raw params. The
        # previous version threw the linearization away here and rebuilt it
        # from scratch in _sol_bwd, paying for an extra full ODE forward solve
        # (through the JAX-graph/BlackJAX gradient path) on every single
        # gradient evaluation for no benefit -- same math, ~20% faster.
        primal, vjp_fn = jax.vjp(_forward_raw, *params)
        return primal, vjp_fn

    def _sol_bwd(vjp_fn, gz):
        gz = jnp.asarray(gz, dtype=jnp.float64)
        return vjp_fn(gz)

    sol_op_jax.defvjp(_sol_fwd, _sol_bwd)

    sol_op_jax_jitted = eqx.filter_jit(sol_op_jax)

    def vjp_sol_op_jax(gz, *params):
        gz = jnp.asarray(gz, dtype=jnp.float64)
        _, vjp_fn = jax.vjp(_forward_raw, *params)
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


@dataclass(frozen=True)
class ImportedSolverParams:
    solver_params_file: Path
    solver_params: dict[str, Any]
    path_base_dir: Path
    output_paths: dict[str, Any]
    reactions_source: "Path | list[Path]"
    results_save_dir: Path
    prior_samples_file: str
    posterior_samples_file: str


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
    """Compute summary and LOO for provided free parameters."""
    available_vars = set(inf_data.posterior.data_vars)
    selected_free_params = [name for name in free_params if name in available_vars]
    if not selected_free_params:
        raise ValueError("None of the requested free parameters were found in posterior variables.")

    summary = az.summary(inf_data, var_names=selected_free_params, round_to=4)
    loo = az.loo(inf_data)
    return {
        "free_params": selected_free_params,
        "summary": summary,
        "loo": loo,
    }

CONDITION_MATRIX = None
EXPERIMENT = None


def _resolve_solver_relative_path(
    base_dir: Path, path_value: "str | Path | list[str | Path]"
) -> "Path | list[Path]":
    if isinstance(path_value, list):
        return [_resolve_solver_relative_path(base_dir, entry) for entry in path_value]
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _get_solver_path_base_dir(solver_params: dict[str, Any], solver_params_file: str | Path) -> Path:
    solver_params_dir = Path(solver_params_file).resolve().parent
    return _resolve_solver_relative_path(solver_params_dir, solver_params.get("path_base", "."))


def import_solver_params(solver_params_file: str | Path) -> ImportedSolverParams:
    """Load solver config and resolve JSON/YAML-defined input/output paths."""
    solver_params_path = Path(solver_params_file).expanduser().resolve()
    with open(solver_params_path, "r", encoding="utf-8") as file:
        if solver_params_path.suffix in (".yaml", ".yml"):
            solver_params = yaml.safe_load(file)
        elif solver_params_path.suffix == ".json":
            solver_params = json.load(file)
        else:
            raise ValueError(f"Unsupported solver params file format: {solver_params_path.suffix}")

    if not isinstance(solver_params, dict):
        raise ValueError(f"Solver params file did not load to a dictionary: {solver_params_path}")

    output_paths = solver_params.get("output_paths")
    if not isinstance(output_paths, dict):
        raise KeyError(
            "solver_params config must include an 'output_paths' section. "
            "Regenerate it with guided_solver_config_builder.ipynb."
        )

    required_output_path_keys = [
        "reactions_source",
        "results_save_dir",
        "prior_samples_file",
        "posterior_samples_file",
    ]
    missing_output_path_keys = [key for key in required_output_path_keys if key not in output_paths]
    if missing_output_path_keys:
        raise KeyError(f"Missing output_paths keys: {missing_output_path_keys}")

    base_dir = _get_solver_path_base_dir(solver_params, solver_params_path)
    reactions_source_path = _resolve_solver_relative_path(base_dir, output_paths["reactions_source"])
    savedir_path = _resolve_solver_relative_path(base_dir, output_paths["results_save_dir"])

    return ImportedSolverParams(
        solver_params_file=solver_params_path,
        solver_params=solver_params,
        path_base_dir=base_dir,
        output_paths=output_paths,
        reactions_source=reactions_source_path,
        results_save_dir=savedir_path,
        prior_samples_file=str(output_paths["prior_samples_file"]),
        posterior_samples_file=str(output_paths["posterior_samples_file"]),
    )


@dataclass(frozen=True)
class ModelBundle:
    """ODE-backed PyMC model plus the metadata both sampler paths need."""

    pm_model: Any
    free_params: list[str]
    species_names: list[str]
    param_names: list[str]
    param_values: dict[str, float]
    experiment: Any
    experiment_summary: str


def _build_model_bundle(imported: ImportedSolverParams) -> ModelBundle:
    """Build the ODE simulator, experiment bundle, and PyMC model from config."""
    solver_params = imported.solver_params
    solver_params_path = imported.solver_params_file

    ode_system, species_names, param_names, param_values, scaling_params = build_ode_system_from_reactions(
        imported.reactions_source
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

    return ModelBundle(
        pm_model=pm_model,
        free_params=free_params,
        species_names=species_names,
        param_names=param_names,
        param_values=param_values,
        experiment=experiment,
        experiment_summary=experiment_summary,
    )


def run_bayesian_inference(
    solver_params_file: str | Path,
    max_hours: float | None = None,
    extra_draws: int = 0,
    resume: bool = True,
) -> "InferenceRunResult | dict[str, Any]":
    """Run (or advance) a checkpointed BlackJAX NUTS inference run.

    Every posterior draw is checkpointed, so this is the single entry point for
    both interactive use (call with the defaults; ``max_hours=None`` runs to
    completion in this process, like a classic single-shot sampler call) and
    Alpine's 24h job cap (chain dependent SLURM jobs, each passing
    ``--max_hours`` below the wall limit; each call resumes from
    ``<results_save_dir>/checkpoint/`` and finalizes -- writes the netcdf
    outputs -- only once the draw target is met). ``extra_draws`` raises the
    target so a follow-up job/call can run longer.

    Returns an :class:`InferenceRunResult` once the run is finalized. If a
    prior call already finished this run (e.g. a chained SLURM job that
    overshot), returns the lightweight persisted status dict instead without
    rebuilding the model -- reload ``posterior_samples_pm.nc`` directly (or use
    ``finalize_window.py``) if you need the full result object for an
    already-completed run.
    """
    import shutil

    import resumable_sampler as rs

    imported = import_solver_params(solver_params_file)
    _print_run_configuration(imported)
    solver_params = imported.solver_params
    savedir_path = imported.results_save_dir
    savedir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = savedir_path / "checkpoint"

    posterior_config = solver_params.get("posterior_sampling", {})
    n_tune = int(posterior_config.get("tune", 1000))
    n_draws = int(posterior_config.get("draws", 1000))
    n_chains = int(posterior_config.get("chains", 4))
    target_accept = float(posterior_config.get("target_accept", 0.8))
    seed = int(posterior_config.get("random_seed", 0))
    checkpoint_every = int(posterior_config.get("checkpoint_every_steps", 50))
    is_diag = bool(posterior_config.get("is_mass_matrix_diagonal", True))
    initial_step = float(posterior_config.get("initial_step_size", 1.0))

    # Fast no-op if a prior job already finished this run (SLURM chains overshoot).
    results_file = savedir_path / imported.posterior_samples_file
    status_file = checkpoint_dir / "status.json"
    if resume and not extra_draws and status_file.exists() and results_file.exists():
        prev = json.loads(status_file.read_text())
        if prev.get("is_done"):
            _print_section("Resumable BlackJAX Sampling")
            print(f"Run already complete ({prev.get('sampling_done')} draws) and finalized; nothing to do.")
            return prev

    if not resume and checkpoint_dir.exists():
        print(f"--no_resume: discarding existing checkpoint at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    bundle = _build_model_bundle(imported)

    config_signature = json.dumps(
        {
            "posterior_sampling": posterior_config,
            "free_params": bundle.free_params,
            "reactions_source": str(imported.reactions_source),
        },
        sort_keys=True,
        default=str,
    )

    _print_section("Resumable BlackJAX Sampling")
    _print_kv("Checkpoint dir", checkpoint_dir)
    _print_kv("tune / draws / chains", f"{n_tune} / {n_draws} / {n_chains}")
    _print_kv("target_accept", target_accept)
    _print_kv("mass matrix", "diagonal" if is_diag else "dense")
    _print_kv("checkpoint_every_steps", checkpoint_every)
    _print_kv("max_hours (this segment)", max_hours if max_hours else "unbounded")
    if extra_draws:
        _print_kv("extra_draws requested", extra_draws)

    bridge = rs.prepare_from_pymc(
        bundle.pm_model, n_chains=n_chains, random_seed=seed, config_signature=config_signature
    )
    spec = rs.SamplerSpec(
        n_tune=n_tune,
        n_draws=n_draws,
        n_chains=n_chains,
        target_accept=target_accept,
        is_mass_matrix_diagonal=is_diag,
        initial_step_size=initial_step,
        checkpoint_every=checkpoint_every,
        random_seed=seed,
    )
    sampler = rs.ResumableSampler(
        bridge.logdensity_fn,
        bridge.initial_positions,
        bridge.value_var_names,
        spec,
        checkpoint_dir,
        bridge.config_hash,
    )

    max_seconds = float(max_hours) * 3600.0 if max_hours else None
    stage_start = time.perf_counter()
    status = sampler.run(max_seconds=max_seconds, extra_draws=int(extra_draws))
    segment_seconds = time.perf_counter() - stage_start

    _print_section("Segment Result")
    _print_kv("phase", status.phase)
    _print_kv("warmup_done", f"{status.warmup_done}/{status.n_tune}")
    _print_kv("sampling_done", f"{status.sampling_done}/{status.n_draws}")
    _print_kv("stopped_reason", status.stopped_reason)
    _print_kv("segment_seconds", f"{segment_seconds:.1f}")

    if not status.is_done:
        print("\nSegment incomplete -- checkpoint saved. Submit another dependent job to continue.")
        return status

    _print_section("Finalizing (draw target met)")
    return _finalize_resumable_run(
        imported=imported,
        bundle=bundle,
        draws=sampler.read_draws("sampling"),
        stats=sampler.read_stats("sampling"),
        segment_seconds=segment_seconds,
        warmup_draws=sampler.read_draws("warmup"),
    )


def _finalize_resumable_run(
    imported, bundle, draws, stats, segment_seconds, warmup_draws=None
) -> InferenceRunResult:
    """Convert selected draws into the standard netcdf outputs + metrics.

    ``draws``/``stats`` are {name: (chains, draws, *shape)} in unconstrained
    space (as persisted by the sampler); the caller chooses which draws (e.g.
    the sampling phase, or a retroactively windowed slice). ``warmup_draws``, if
    given, is added as a ``warmup_posterior`` group so trace plots can show the
    tuning phase (inference_plotting reads it for include_tuning).
    """
    from arviz_base import from_dict
    from pymc.backends.arviz import coords_and_dims_for_inferencedata
    from pymc.sampling.jax import get_jaxified_graph, _postprocess_samples
    from pymc.util import get_default_varnames

    pm_model = bundle.pm_model
    free_params = bundle.free_params
    solver_params = imported.solver_params
    savedir_path = imported.results_save_dir

    # Map unconstrained value-var draws -> constrained free params. Use
    # unobserved_value_vars (pure transforms of the value vars) rather than
    # free_RVs, whose graphs carry RNG shared variables that can't be jaxified.
    # Drop deterministics (e.g. the ODE-heavy `prediction`); sample_posterior_
    # predictive supplies those, and inference_plotting reads predictions from the
    # posterior_predictive group.
    constrained_vars = list(
        get_default_varnames(pm_model.unobserved_value_vars, include_transformed=False)
    )
    deterministic_names = {det.name for det in pm_model.deterministics}
    param_vars = [var for var in constrained_vars if var.name not in deterministic_names]
    constrained_fn = get_jaxified_graph(inputs=pm_model.value_vars, outputs=param_vars)

    def _to_constrained(unconstrained_draws):
        raw = [jnp.asarray(unconstrained_draws[value_var.name]) for value_var in pm_model.value_vars]
        result = _postprocess_samples(constrained_fn, raw, postprocessing_vectorize="scan")
        return {var.name: np.asarray(values) for var, values in zip(param_vars, result)}

    posterior_samples = _to_constrained(draws)

    sample_stats = {name: np.asarray(values) for name, values in stats.items()}
    if "diverging" in sample_stats:
        sample_stats["diverging"] = sample_stats["diverging"].astype(bool)

    coords, dims = coords_and_dims_for_inferencedata(pm_model)
    idata = from_dict(
        data={"posterior": posterior_samples, "sample_stats": sample_stats},
        coords=coords,
        dims=dims,
        sample_dims=["chain", "draw"],
    )

    # backend="jax" routes through the same jax_funcify path SolOp already
    # registers a dispatch for; without it, pytensor's default backend tries
    # numba, which has no numba dispatch for SolOp and silently falls back to
    # slow per-call Python object-mode for the ODE node.
    #
    # CAUTION (observed on macOS): creating this SEPARATE JAX/XLA compilation
    # context in the same process right after BlackJAX's own heavy JAX usage
    # intermittently aborted the process (libc++abi/std::__1::system_error --
    # a hard abort(), not a catchable Python exception, so there's no safe
    # try/except fallback around it). Not yet confirmed whether this
    # reproduces on Linux/Alpine. If a finalize step aborts with that error,
    # that's this tradeoff -- rerun, or drop backend="jax" here.
    print("Computing log-likelihood, prior, and posterior predictive...")
    with pm_model:
        pm.compute_log_likelihood(idata, progressbar=False, backend="jax")
    prior_pred = _sample_prior(pm_model=pm_model, solver_params=solver_params, free_params=free_params)
    with pm_model:
        post_pred = pm.sample_posterior_predictive(idata, progressbar=False, backend="jax")

    _print_section("Artifacts")
    prior_file = savedir_path / imported.prior_samples_file
    results_file = savedir_path / imported.posterior_samples_file
    _safe_write_idata(prior_pred, prior_file)
    combined = idata.copy()
    combined.update(prior_pred)
    combined.update(post_pred)
    if warmup_draws:
        warmup_posterior = from_dict(
            data={"posterior": _to_constrained(warmup_draws)},
            coords=coords,
            dims=dims,
            sample_dims=["chain", "draw"],
        ).posterior
        combined["warmup_posterior"] = warmup_posterior
    _safe_write_idata(combined, results_file)
    _print_kv("Prior file", prior_file.name)
    _print_kv("Posterior file", results_file.name)

    timing = {"posterior_sampling_sec": segment_seconds}
    timing_file = savedir_path / "timing.json"
    if timing_file.exists():
        try:
            timing = {**json.loads(timing_file.read_text()), **timing}
        except (json.JSONDecodeError, OSError):
            pass
    with open(timing_file, "w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=4)

    # Artifacts are already written above, so a metrics/LOO failure (e.g. az.loo
    # needs >=5 tail draws, so very short diagnostic runs can't compute it) must
    # not fail the job or its exit code.
    _print_section("Metrics")
    summary = loo = None
    try:
        metrics = summarize_inference_metrics(inf_data=combined, free_params=free_params)
        summary = metrics["summary"]
        free_params = metrics["free_params"]
        loo = metrics["loo"]
        _print_section("Posterior Summary")
        print(summary)
        _print_section("Model Comparison")
        print("LOO")
        print(loo)
    except Exception as exc:  # noqa: BLE001 - metrics are informational only
        print(f"Metrics/LOO skipped ({type(exc).__name__}: {exc}). Artifacts were written.")

    return InferenceRunResult(
        posterior=combined,
        prior_predictive=prior_pred,
        posterior_predictive=post_pred,
        summary=summary,
        free_params=free_params,
        species_names=bundle.species_names,
        param_names=bundle.param_names,
        param_values=bundle.param_values,
        solver_params=solver_params,
        experiment=bundle.experiment,
        experiment_summary=bundle.experiment_summary,
        prior_file=str(prior_file),
        results_file=str(results_file),
        loo=loo,
    )


def _build_solver(solver_params):
    """Construct the diffrax solver.
    """
    ode_solver_config = solver_params.get("ODE_solver", {})
    solver_name = ode_solver_config.get("solver_name", "Kvaerno5")
    solver_cls = getattr(dfrx, solver_name)

    return solver_cls()


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
            throw=False,
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

    # backend="jax": see the note on the compute_log_likelihood/
    # sample_posterior_predictive calls in _finalize_resumable_run.
    with pm_model:
        return pm.sample_prior_predictive(
            draws=draws,
            random_seed=random_seed,
            var_names=var_names,
            backend="jax",
        )


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

    # inf_data.groups holds DataTree-style paths (e.g. "/posterior"), with the
    # tree root itself listed as "/". Skip the root and strip the leading "/"
    # since getattr() needs the bare node name.
    for group_path in inf_data.groups:
        group_name = group_path.lstrip("/")
        if not group_name:
            continue
        group = getattr(inf_data, group_name)
        group.attrs = {key: _netcdf_safe_attr(value) for key, value in group.attrs.items()}

    inf_data.attrs = {key: _netcdf_safe_attr(value) for key, value in inf_data.attrs.items()}

    if output_file.exists():
        output_file.unlink()
    inf_data.to_netcdf(output_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bayesian inference from a solver params JSON/YAML file."
    )
    parser.add_argument(
        "--solver_params_file",
        required=True,
        help="Path to solver_params.json, solver_params.yaml, or solver_params.yml.",
    )
    parser.add_argument(
        "--max_hours",
        type=float,
        default=None,
        help="Wall-clock budget for this segment before checkpointing and exiting. "
        "Omit to run to completion. Set below the SLURM --time limit with margin, "
        "e.g. 23.5 for a 24h cap.",
    )
    parser.add_argument(
        "--extra_draws",
        type=int,
        default=0,
        help="Raise the posterior draw target by this many before running. "
        "Use to extend a finished/underpowered run.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Discard any existing checkpoint and start the run fresh.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_bayesian_inference(
        args.solver_params_file,
        max_hours=args.max_hours,
        extra_draws=args.extra_draws,
        resume=not args.no_resume,
    )
