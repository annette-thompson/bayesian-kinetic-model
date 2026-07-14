"""Monte Carlo propagation of initial-concentration uncertainty into model outputs.

Analogous to the old MATLAB ``Model_Error.m`` workflow: draw many sets of initial
concentrations from a normal distribution around their nominal (mean) values,
solve the ODE system for every draw, and report the resulting spread (mean/std)
in a chosen output observable (a final concentration or an initial rate).

All draws for a chunk are solved in a single batched ``jax.vmap`` call over
``diffrax.diffeqsolve``, so this scales well on a multi-core Alpine node.
Designed to be driven either interactively (import the functions in a notebook)
or non-interactively from the command line via a JSON config file:

    python model_error_runner.py --config /path/to/model_error_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfrx

from reaction_model_builder import (
    build_ode_system_from_reactions,
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
    t_eval: np.ndarray,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the ODE system for every row of ``y0_batch`` in one batched vmap call.

    ``t_eval`` must be a fixed array of evaluation times (shared across all draws) so
    every solve produces the same output shape and can be stacked by vmap. Returns
    ``(C_batch, ok)`` where ``C_batch`` has shape ``(n_samples, len(t_eval), n_species)``
    and ``ok`` is a boolean mask of which draws integrated successfully.

    ``throw=False`` is required: diffrax cannot raise on integration failure under vmap,
    so failed draws are instead flagged via ``sol.result`` and filtered by the caller.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    t_eval = jnp.asarray(t_eval, dtype=jnp.float64)

    def solve_one(y0: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(network),
            dfrx.Kvaerno5(),
            t0=time_range[0], t1=time_range[1], dt0=1e-6,
            y0=y0,
            args=theta,
            saveat=dfrx.SaveAt(ts=t_eval),
            stepsize_controller=dfrx.PIDController(
                rtol=1e-5, atol=1e-8, pcoeff=0.2, icoeff=0.4, dcoeff=0,
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
    t_eval: np.ndarray,
    batch_size: int | None = None,
    max_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``run_ensemble`` in sequential chunks of ``batch_size`` draws to bound peak memory."""
    n_samples = y0_batch.shape[0]
    chunk = n_samples if not batch_size else min(batch_size, n_samples)

    C_chunks: list[np.ndarray] = []
    ok_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, chunk):
        y0_chunk = y0_batch[start:start + chunk]
        C_chunk, ok_chunk = run_ensemble(network, theta, y0_chunk, time_range, t_eval, max_steps=max_steps)
        C_chunks.append(C_chunk)
        ok_chunks.append(ok_chunk)
        print(f"  ...solved draws {start + 1}-{start + len(y0_chunk)} of {n_samples}")

    return np.concatenate(C_chunks, axis=0), np.concatenate(ok_chunks, axis=0)


def compute_observable(
    t_eval: np.ndarray,
    C_batch: np.ndarray,
    species: Sequence[str],
    observable: Mapping[str, Any],
) -> np.ndarray:
    """Reduce a batch of time courses to one scalar per draw.

    ``observable["type"]``:
    - ``"final_conc"``: concentration of ``observable["species"]`` at the last saved time.
    - ``"initial_rate"``: slope of a linear fit to ``observable["species"]`` vs. time over
      ``observable.get("window", [t_eval[0], t_eval[-1]])`` — the Python analog of the
      MATLAB ``Calc_Function`` initial-rate calculation.
    """
    species_idx = {name: i for i, name in enumerate(species)}
    if observable["species"] not in species_idx:
        raise KeyError(f"Observable species '{observable['species']}' not found in reaction network.")
    idx = species_idx[observable["species"]]
    obs_type = observable["type"]

    if obs_type == "final_conc":
        return C_batch[:, -1, idx]

    if obs_type == "initial_rate":
        t0, t1 = observable.get("window", [float(t_eval[0]), float(t_eval[-1])])
        mask = (t_eval >= t0) & (t_eval <= t1)
        if mask.sum() < 2:
            raise ValueError(f"initial_rate window {[t0, t1]} contains fewer than 2 saved time points.")
        t_win = t_eval[mask]
        slopes = np.empty(C_batch.shape[0], dtype=np.float64)
        for i in range(C_batch.shape[0]):
            slopes[i] = np.polyfit(t_win, C_batch[i, mask, idx], 1)[0]
        return slopes

    raise ValueError(f"Unknown observable type: {obs_type!r}. Use 'final_conc' or 'initial_rate'.")


def run_from_config(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full Monte Carlo error-propagation workflow from a JSON config file."""
    config_path = Path(config_path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = json.load(fh)

    path_base = _resolve_path(config_path.parent, config.get("path_base", "."))
    reactions_path = _resolve_path(path_base, config["reactions_path"])

    print(f"==> Loading reactions from {reactions_path}")
    network, species, params, param_values, scaling_groups = build_ode_system_from_reactions(reactions_path)
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
    t_eval_cfg = config.get("t_eval", {"n_points": 50})
    if isinstance(t_eval_cfg, dict):
        t_eval = np.linspace(time_range[0], time_range[1], int(t_eval_cfg.get("n_points", 50)))
    else:
        t_eval = np.asarray(t_eval_cfg, dtype=np.float64)

    print(f"==> Solving ODE ensemble over t in {time_range} ({len(t_eval)} saved points)...")
    C_batch, ok = run_ensemble_chunked(
        network, theta, y0_batch, time_range, t_eval, batch_size=batch_size,
    )

    n_failed = int((~ok).sum())
    if n_failed:
        print(f"Warning: {n_failed}/{n_samples} draws failed to integrate and were excluded from the summary.")
    y0_ok = y0_batch[ok]
    C_ok = C_batch[ok]

    observable = config["observable"]
    obs_vals = compute_observable(t_eval, C_ok, species, observable)

    output_folder = _resolve_path(
        path_base, config.get("output_folder", f"Results/{config.get('folder_name', 'Model Error')}")
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    obs_label = f"{observable['type']}__{observable['species']}"
    df = pd.DataFrame({name: y0_ok[:, species_idx[name]] for name in perturb_species})
    df[obs_label] = obs_vals

    samples_path = output_folder / "model_error_samples.csv"
    df.to_csv(samples_path, index=False)

    mean_val = float(np.mean(obs_vals)) if len(obs_vals) else float("nan")
    std_val = float(np.std(obs_vals, ddof=1)) if len(obs_vals) > 1 else 0.0
    summary = {
        "observable": observable,
        "sigma_frac": sigma_frac,
        "perturbed_species": perturb_species,
        "n_samples_requested": n_samples,
        "n_samples_succeeded": int(ok.sum()),
        "n_samples_failed": n_failed,
        "mean": mean_val,
        "std": std_val,
        "relative_std": (std_val / mean_val) if mean_val else None,
    }
    summary_path = output_folder / "model_error_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)

    print(f"==> Saved {len(df)} per-draw samples to {samples_path}")
    print(f"==> {obs_label}: mean={mean_val:.6g}, std={std_val:.6g} (n={int(ok.sum())})")
    print(f"==> Saved summary to {summary_path}")

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo model-error propagation runner.")
    parser.add_argument("--config", required=True, help="Path to a model_error_config.json file.")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
