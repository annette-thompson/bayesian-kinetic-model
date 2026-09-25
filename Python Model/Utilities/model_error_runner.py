"""Monte Carlo propagation of initial-concentration uncertainty into model outputs.

Analogous to the old MATLAB ``Model_Error.m`` workflow: draw many sets of initial
concentrations from a normal distribution around their nominal (mean) values,
solve the ODE system for every draw, and report the resulting spread (mean/std)
in a chosen output observable computed from each draw's final state.

Two ways to spread draws across cores, chosen via the config's ``"parallel": {"mode": ...}``:
- ``"vmap"`` (default): all draws in a chunk are solved in a single batched ``jax.vmap``
  call over ``diffrax.diffeqsolve``. Fast on GPU, but on CPU every draw in the batch is
  locked to the same number of adaptive steps (the slowest draw holds up the rest).
- ``"process"``: each draw is solved independently across a pool of worker processes, one
  draw at a time per worker. No lockstep tax -- a worker just grabs the next draw as soon
  as it's free -- so this is generally the better choice on a many-core CPU node like Alpine.

Designed to be driven either interactively (import the functions in a notebook)
or non-interactively from the command line via a JSON config file:

    python model_error_runner.py --config /path/to/model_error_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# The shared "Bayesian" conda env has jax[cuda12] installed for the GPU inference scripts.
# On a CPU-only node/partition, jax's PJRT plugin discovery still unconditionally tries to
# initialize that CUDA plugin (this happens before JAX_PLATFORMS is even consulted, so that
# env var can't suppress it) and logs the resulting "no CUDA device" failure as an ERROR in
# every process -- harmless, but it floods the log with a traceback per worker. Silence it.
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfrx

try:
    # Works when run directly as a script (CLI / Alpine sbatch): Python puts this file's own
    # directory on sys.path, so its sibling module is importable by its bare name.
    from reaction_model_builder import (
        build_ode_system_from_reactions,
        discover_scaling_groups,
        nominal_scaling_group_values,
        set_scaling_group_values,
    )
except ModuleNotFoundError:
    # Works when imported as "Utilities.model_error_runner" (e.g. from a notebook that did
    # sys.path.insert(0, "../")) -- here only the parent dir is on sys.path, not Utilities/
    # itself, so the sibling module has to be found via the package instead.
    from Utilities.reaction_model_builder import (
        build_ode_system_from_reactions,
        discover_scaling_groups,
        nominal_scaling_group_values,
        set_scaling_group_values,
    )


def _resolve_path(base_dir: str | Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir).expanduser() / path).resolve()


def sample_perturbed_y0(
    y0_mean: np.ndarray,
    species: Sequence[str],
    perturb_species: Sequence[str],
    sigma_frac: float,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_samples`` initial-condition vectors with Gaussian noise on selected species.

    Every draw starts from ``y0_mean``; only the species listed in ``perturb_species``
    are resampled, each as ``Normal(mean=y0_mean[species], std=sigma_frac * y0_mean[species])``,
    clipped at zero since concentrations cannot be negative. All other species are held fixed.
    """
    rng = np.random.default_rng(seed)
    species_idx = {name: i for i, name in enumerate(species)}
    y0_batch = np.tile(np.asarray(y0_mean, dtype=np.float64), (n_samples, 1))

    for name in perturb_species:
        if name not in species_idx:
            raise KeyError(f"Perturbed species '{name}' not found in reaction network species list.")
        idx = species_idx[name]
        mean_val = float(y0_mean[idx])
        draws = rng.normal(loc=mean_val, scale=sigma_frac * mean_val, size=n_samples)
        y0_batch[:, idx] = np.clip(draws, 0.0, None)

    return y0_batch


def run_ensemble(
    network: Any,
    theta: jnp.ndarray,
    y0_batch: np.ndarray,
    time_range: Sequence[float],
    max_steps: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the ODE system for every row of ``y0_batch`` in one batched vmap call.

    Uses diffeqsolve's default ``SaveAt`` (``t1=True``, i.e. keep only the final state)
    instead of an explicit ``ts`` grid -- this also guarantees a static output shape across
    the batch regardless of how many adaptive steps each draw's solve takes, which is what
    lets vmap stack them at all.

    ``throw=False`` is required: diffrax cannot raise on integration failure under vmap,
    so failed draws are instead flagged via ``sol.result`` and filtered by the caller.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)

    def solve_one(y0: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(network),
            dfrx.Kvaerno5(),
            t0=time_range[0], t1=time_range[1], dt0=1e-6,
            y0=y0,
            args=theta,
            stepsize_controller=dfrx.PIDController(
                rtol=rtol, atol=atol, pcoeff=0.2, icoeff=0.4, dcoeff=0,
            ),
            max_steps=max_steps,
            throw=False,
        )
        return sol.ys, sol.result

    ys, results = jax.vmap(solve_one)(jnp.asarray(y0_batch, dtype=jnp.float64))
    # Compare while `results` is still a (batched) diffrax RESULTS enum — equinox's
    # Enumeration.__eq__ only accepts another enum of the same type, so converting to a
    # plain array first (e.g. via np.asarray) breaks the comparison.
    ok = np.asarray(results == dfrx.RESULTS.successful)
    return np.asarray(ys), ok


def run_ensemble_chunked(
    network: Any,
    theta: jnp.ndarray,
    y0_batch: np.ndarray,
    time_range: Sequence[float],
    batch_size: int | None = None,
    max_steps: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``run_ensemble`` in sequential chunks of ``batch_size`` draws to bound peak memory."""
    n_samples = y0_batch.shape[0]
    chunk = n_samples if not batch_size else min(batch_size, n_samples)

    C_chunks: list[np.ndarray] = []
    ok_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, chunk):
        y0_chunk = y0_batch[start:start + chunk]
        C_chunk, ok_chunk = run_ensemble(
            network, theta, y0_chunk, time_range, max_steps=max_steps, rtol=rtol, atol=atol,
        )
        C_chunks.append(C_chunk)
        ok_chunks.append(ok_chunk)
        print(f"  ...solved draws {start + 1}-{start + len(y0_chunk)} of {n_samples}")

    return np.concatenate(C_chunks, axis=0), np.concatenate(ok_chunks, axis=0)


_worker_state: dict[str, Any] = {}


def _init_worker(
    reactions_path: str,
    theta_list: list[float],
    time_range: tuple[float, float],
    max_steps: int,
    rtol: float,
    atol: float,
) -> None:
    """Build one worker's own copy of the network (cheap: YAML parsing, no heavy compute)."""
    # theta (carrying the real scaling values) is passed to the solver separately,
    # so the build itself only needs to be a no-op.
    network, *_ = build_ode_system_from_reactions(
        reactions_path,
        scaling_group=nominal_scaling_group_values(discover_scaling_groups(reactions_path)))
    _worker_state["network"] = network
    _worker_state["theta"] = jnp.array(theta_list, dtype=jnp.float64)
    _worker_state["time_range"] = time_range
    _worker_state["max_steps"] = max_steps
    _worker_state["rtol"] = rtol
    _worker_state["atol"] = atol


def _solve_one_draw(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, bool]:
    idx, y0_row = item
    network = _worker_state["network"]
    theta = _worker_state["theta"]
    time_range = _worker_state["time_range"]
    max_steps = _worker_state["max_steps"]
    rtol = _worker_state["rtol"]
    atol = _worker_state["atol"]

    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(network),
        dfrx.Kvaerno5(),
        t0=time_range[0], t1=time_range[1], dt0=1e-6,
        y0=jnp.asarray(y0_row, dtype=jnp.float64),
        args=theta,
        stepsize_controller=dfrx.PIDController(
            rtol=rtol, atol=atol, pcoeff=0.2, icoeff=0.4, dcoeff=0,
        ),
        max_steps=max_steps,
        throw=False,
    )
    ok = bool(sol.result == dfrx.RESULTS.successful)
    return idx, np.asarray(sol.ys), ok


def run_ensemble_multiprocess(
    reactions_path: str | Path,
    theta: jnp.ndarray,
    y0_batch: np.ndarray,
    time_range: Sequence[float],
    n_workers: int | None = None,
    max_steps: int = 10_000,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve each draw as an independent ``diffeqsolve`` call, load-balanced across worker processes.

    Unlike ``run_ensemble``'s ``jax.vmap`` batching -- which locks every draw in a chunk to the
    same number of adaptive steps, since XLA's batched while-loop can't exit for individual
    draws until the whole batch is done -- each worker here solves one draw at a time and
    immediately picks up the next once free. A handful of stiff draws then only slow down the
    one worker stuck on them, not the whole ensemble.

    The thread-limiting env vars below must be set on *this* (parent) process before the pool
    is spawned: spawned workers inherit the parent's environment at process creation, before
    they import jax/numpy/BLAS, so this is the only point at which capping each worker to a
    single thread reliably takes effect. Without it, every worker would try to multi-thread
    its own solve and they'd all fight over the same cores.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    n_workers = n_workers or (os.cpu_count() or 1)
    n_samples = y0_batch.shape[0]
    theta_list = [float(x) for x in np.asarray(theta)]
    tasks = list(enumerate(np.asarray(y0_batch, dtype=np.float64)))

    ys: list[np.ndarray | None] = [None] * n_samples
    ok = np.zeros(n_samples, dtype=bool)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(str(reactions_path), theta_list, tuple(time_range), max_steps, rtol, atol),
    ) as pool:
        report_every = max(1, n_samples // 20)
        for n_done, (idx, y_i, ok_i) in enumerate(
            pool.imap_unordered(_solve_one_draw, tasks, chunksize=1), start=1
        ):
            ys[idx] = y_i
            ok[idx] = ok_i
            if n_done % report_every == 0 or n_done == n_samples:
                print(f"  ...solved {n_done}/{n_samples} draws")

    return np.stack(ys, axis=0), ok


# Matches terminal fatty-acid product species, e.g. "C16_FA" (saturated) or "C18_FA_unsat"
# (unsaturated, only present in networks that include FabA). Used by the aggregate FAS
# observables below, which reduce over *every* matching species rather than one named one.
_FA_PRODUCT_PATTERN = re.compile(r"^C(\d+)_FA(_unsat)?$")


def _match_fa_species(
    species: Sequence[str], pattern: "re.Pattern[str]" = _FA_PRODUCT_PATTERN
) -> list[tuple[int, int, bool]]:
    """Return ``(species_index, chain_length, is_unsaturated)`` for every species matching ``pattern``."""
    matches = []
    for i, name in enumerate(species):
        m = pattern.match(name)
        if m:
            matches.append((i, int(m.group(1)), bool(m.group(2))))
    return matches


def compute_observable(
    C_batch: np.ndarray,
    species: Sequence[str],
    observable: Mapping[str, Any],
) -> np.ndarray:
    """Reduce a batch of final states (one per draw) to one scalar per draw.

    Each draw is solved with diffeqsolve's default ``SaveAt`` (final state at t1 only), so
    ``C_batch`` has shape ``(n_draws, 1, n_species)`` -- every reduction below indexes that
    lone saved time with ``[:, -1, ...]``.

    ``observable["type"]``:
    - ``"final_conc"``: concentration of ``observable["species"]`` at t1.
    - ``"total_production"``: summed final concentration across every fatty-acid product
      species matching ``observable.get("species_pattern", r"^C(\\d+)_FA(_unsat)?$")``.
    - ``"unsaturated_fraction"``: fraction of ``total_production`` coming from ``*_unsat``
      species (identically 0 on networks with no unsaturation pathway, e.g. no FabA).
    - ``"mean_chain_length"``: concentration-weighted mean carbon count across the same
      matched product species.

    The last three reduce over *all* matched species and take no ``observable["species"]``.
    ``"initial_rate"`` (a slope over a time window) is not supported: each draw only keeps
    its state at t1, not a time series to fit a slope against.
    """
    obs_type = observable["type"]

    if obs_type in ("total_production", "unsaturated_fraction", "mean_chain_length"):
        pattern = re.compile(observable.get("species_pattern", _FA_PRODUCT_PATTERN.pattern))
        fa_species = _match_fa_species(species, pattern)
        if not fa_species:
            raise ValueError(f"No species matched pattern {pattern.pattern!r} for observable {obs_type!r}.")
        idxs = np.array([i for i, _, _ in fa_species])
        chains = np.array([c for _, c, _ in fa_species], dtype=np.float64)
        is_unsat = np.array([u for _, _, u in fa_species])

        final_vals = C_batch[:, -1, idxs]  # (n_draws, n_matched_species)
        total = final_vals.sum(axis=1)

        if obs_type == "total_production":
            return total

        if obs_type == "unsaturated_fraction":
            unsat_total = final_vals[:, is_unsat].sum(axis=1) if is_unsat.any() else np.zeros_like(total)
            return np.divide(unsat_total, total, out=np.zeros_like(total), where=total > 0)

        weighted = final_vals @ chains  # mean_chain_length
        return np.divide(weighted, total, out=np.zeros_like(total), where=total > 0)

    if obs_type == "initial_rate":
        raise ValueError(
            "'initial_rate' is not supported: each draw only saves its state at t1 "
            "(diffeqsolve's default SaveAt), not a time series to fit a slope against."
        )

    species_idx = {name: i for i, name in enumerate(species)}
    if observable["species"] not in species_idx:
        raise KeyError(f"Observable species '{observable['species']}' not found in reaction network.")
    idx = species_idx[observable["species"]]

    if obs_type == "final_conc":
        return C_batch[:, -1, idx]

    raise ValueError(
        f"Unknown observable type: {obs_type!r}. Use 'final_conc', "
        "'total_production', 'unsaturated_fraction', or 'mean_chain_length' "
        "('initial_rate' is no longer supported -- see this function's docstring)."
    )


def compute_fa_chain_profile(
    C_batch: np.ndarray,
    species: Sequence[str],
    pattern: "re.Pattern[str]" = _FA_PRODUCT_PATTERN,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Per-draw saturated/unsaturated mole fraction at each detected FA chain length.

    Same per-draw normalization as the single-run FA profile bar chart in ``run_model.ipynb``
    (each draw's chain-length fractions sum to 1 across sat + unsat), just computed for every
    draw in the ensemble instead of one deterministic run -- so a per-chain-length mean/std
    (error bars) can be taken across draws afterward.

    Returns ``(chains, sat_frac, unsat_frac)``: ``chains`` is the sorted list of carbon counts
    found, and ``sat_frac``/``unsat_frac`` each have shape ``(n_draws, len(chains))``.
    """
    fa_species = _match_fa_species(species, pattern)
    if not fa_species:
        raise ValueError(f"No species matched pattern {pattern.pattern!r} for FA chain profile.")
    chains = sorted(set(c for _, c, _ in fa_species))
    chain_col = {c: i for i, c in enumerate(chains)}

    final_vals = C_batch[:, -1, :]  # (n_draws, n_species)
    n_draws = C_batch.shape[0]
    sat = np.zeros((n_draws, len(chains)))
    unsat = np.zeros((n_draws, len(chains)))
    for i, c, is_unsat in fa_species:
        col = chain_col[c]
        (unsat if is_unsat else sat)[:, col] += final_vals[:, i]

    total = sat.sum(axis=1) + unsat.sum(axis=1)
    sat_frac = np.divide(sat, total[:, None], out=np.zeros_like(sat), where=total[:, None] > 0)
    unsat_frac = np.divide(unsat, total[:, None], out=np.zeros_like(unsat), where=total[:, None] > 0)
    return chains, sat_frac, unsat_frac


def run_from_config(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full Monte Carlo error-propagation workflow from a JSON config file."""
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = json.load(fh)

    path_base = _resolve_path(config_path.parent, config.get("path_base", "."))
    reactions_path = _resolve_path(path_base, config["reactions_path"])

    print(f"==> Loading reactions from {reactions_path}")
    network, species, params, param_values, scaling_groups = build_ode_system_from_reactions(
        reactions_path,
        scaling_group=nominal_scaling_group_values(discover_scaling_groups(reactions_path)))
    species_idx = {name: i for i, name in enumerate(species)}

    theta = jnp.array([param_values[p] for p in params], dtype=jnp.float64)
    theta = set_scaling_group_values(theta, params, config.get("scaling_groups", {}))

    y0_mean = np.zeros(len(species), dtype=np.float64)
    for name, val in {**config.get("y0", {}), **config.get("enzyme_concs", {})}.items():
        if name not in species_idx:
            raise KeyError(f"Species '{name}' in y0/enzyme_concs not found in reaction network.")
        y0_mean[species_idx[name]] = float(val)

    perturb_cfg = config["perturb"]
    perturb_species = list(perturb_cfg["species"])
    sigma_frac = float(perturb_cfg.get("sigma_frac", 0.05))
    n_samples = int(config.get("n_samples", 1000))
    seed = int(config.get("seed", 0))
    batch_size = config.get("batch_size")

    print(f"==> Sampling {n_samples} draws (sigma = {sigma_frac:.1%} of mean) for: {perturb_species}")
    y0_batch = sample_perturbed_y0(y0_mean, species, perturb_species, sigma_frac, n_samples, seed=seed)

    time_range = config.get("time_range", [0.0, 720.0])

    max_steps = int(config.get("max_steps", 10_000))
    tolerance_cfg = config.get("tolerance", {})
    rtol = float(tolerance_cfg.get("rtol", 1e-5))
    atol = float(tolerance_cfg.get("atol", 1e-8))
    parallel_cfg = config.get("parallel", {})
    parallel_mode = parallel_cfg.get("mode", "vmap")

    print(f"==> Solving ODE ensemble over t in {time_range}, "
          f"rtol={rtol:g}, atol={atol:g}, max_steps={max_steps}...")
    if parallel_mode == "process":
        n_workers = parallel_cfg.get("n_workers") or int(
            os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        )
        print(f"==> Process-parallel mode: {n_workers} worker processes, one draw at a time each.")
        C_batch, ok = run_ensemble_multiprocess(
            reactions_path, theta, y0_batch, time_range,
            n_workers=n_workers, max_steps=max_steps, rtol=rtol, atol=atol,
        )
    else:
        C_batch, ok = run_ensemble_chunked(
            network, theta, y0_batch, time_range, batch_size=batch_size,
            max_steps=max_steps, rtol=rtol, atol=atol,
        )

    n_failed = int((~ok).sum())
    if n_failed:
        print(f"Warning: {n_failed}/{n_samples} draws failed to integrate and were excluded from the summary.")
    y0_ok = y0_batch[ok]
    C_ok = C_batch[ok]

    # A single observable dict (back-compatible) or a list of them, computed from the same
    # ensemble -- matches the paper's method of reporting several outputs (e.g. total
    # production, unsaturated fraction, chain length) from one set of 300 model runs.
    observable_cfg = config["observable"]
    observables = observable_cfg if isinstance(observable_cfg, list) else [observable_cfg]

    output_folder = _resolve_path(
        path_base, config.get("output_folder", f"Results/{config.get('folder_name', 'Model Error')}")
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({name: y0_ok[:, species_idx[name]] for name in perturb_species})
    obs_summaries: dict[str, dict[str, Any]] = {}
    for observable in observables:
        obs_vals = compute_observable(C_ok, species, observable)
        obs_label = f"{observable['type']}__{observable['species']}" if "species" in observable else observable["type"]
        df[obs_label] = obs_vals

        mean_val = float(np.mean(obs_vals)) if len(obs_vals) else float("nan")
        std_val = float(np.std(obs_vals, ddof=1)) if len(obs_vals) > 1 else 0.0
        obs_summaries[obs_label] = {
            "observable": observable,
            "mean": mean_val,
            "std": std_val,
            "relative_std": (std_val / mean_val) if mean_val else None,
        }
        print(f"==> {obs_label}: mean={mean_val:.6g}, std={std_val:.6g} (n={int(ok.sum())})")

    samples_path = output_folder / "model_error_samples.csv"
    df.to_csv(samples_path, index=False)

    fa_profile_cfg = config.get("fa_profile", {})
    fa_profile_path = None
    if fa_profile_cfg.get("save", False):
        pattern = re.compile(fa_profile_cfg.get("species_pattern", _FA_PRODUCT_PATTERN.pattern))
        chains, sat_frac, unsat_frac = compute_fa_chain_profile(C_ok, species, pattern)
        profile_df = pd.DataFrame({
            **{f"C{c}_sat_frac": sat_frac[:, j] for j, c in enumerate(chains)},
            **{f"C{c}_unsat_frac": unsat_frac[:, j] for j, c in enumerate(chains)},
        })
        fa_profile_path = output_folder / "fa_chain_profile_samples.csv"
        profile_df.to_csv(fa_profile_path, index=False)
        print(f"==> Saved per-draw FA chain profile ({len(chains)} chain lengths) to {fa_profile_path}")

    summary = {
        "sigma_frac": sigma_frac,
        "perturbed_species": perturb_species,
        "n_samples_requested": n_samples,
        "n_samples_succeeded": int(ok.sum()),
        "n_samples_failed": n_failed,
        "observables": obs_summaries,
    }
    summary_path = output_folder / "model_error_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)

    print(f"==> Saved {len(df)} per-draw samples to {samples_path}")
    print(f"==> Saved summary to {summary_path}")

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo model-error propagation runner.")
    parser.add_argument("--config", required=True, help="Path to a model_error_config.json file.")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
