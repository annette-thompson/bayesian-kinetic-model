"""Write a short-run copy of a solver_params.json for benchmarking.

draws/tune cannot be set from inference_runner.py's CLI -- they are read only from
the JSON (posterior_sampling.*) -- so a short run means writing a modified copy.
Precedent: Results/CPU Scaling Tests/_finalize_crash_check/gpu_finalize_and_speedup_test.sh

Stdlib only: this is imported by the benchmark driver, which must never import jax.

Three things the precedent script did not have to handle, because it wrote its copy
at the same depth as the original:

1. ``path_base`` must be RECOMPUTED. The 7 scaling configs carry "../../.." -- three
   levels up from Results/GPU Scaling Tests/<X>/. Every datasets[].data_file and the
   calculation_module resolve through it, so a copy at a different depth silently
   resolves to the wrong root and either fails to find data or (worse) finds the
   wrong data.
2. ``output_paths.results_save_dir`` must point at the benchmark directory, or the
   run writes its netcdf over the real results.
3. The directory must NOT be named ``Test*``. compare_sampler_benchmarks.py globs
   ``Results/**/Test*/solver_params.json``, so a benchmark run named that way is
   silently absorbed into the production scaling comparison.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from run_registry import BENCHMARK_PREFIX, config_label, project_root

# Everything this harness generates lives here, so it is trivially separable from
# real results (and .gitignore can exclude the heavy artifacts under it).
BENCHMARK_ROOT = "Results/Benchmarks"


def cell_slug(machine: str, device: str, floor: float, label: str) -> str:
    """Filesystem-safe identity for one matrix cell.

    Deliberately starts with BENCHMARK_PREFIX ("bench__") -- see module docstring.
    """
    floor_s = "0" if not floor else f"{floor:g}".replace("-", "m").replace(".", "p")
    lab = label.replace(" ", "-")
    return f"{BENCHMARK_PREFIX}{machine}__{device}__floor{floor_s}__{lab}"


def write_benchmark_config(
    src_config: Path,
    out_dir: Path,
    *,
    draws: int,
    tune: int,
    chains: int = 4,
    checkpoint_every: int = 10,
    floor: float = 0.0,
    prior_draws: int = 500,
    rtol: float | None = None,
    atol: float | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
) -> Path:
    """Write <out_dir>/solver_params.json derived from src_config. Returns its path.

    Everything not named here is inherited from the source config unchanged.
    """
    src_config = Path(src_config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(src_config.read_text(encoding="utf-8"))
    root = project_root()

    # relpath, not Path.relative_to: only relpath can emit the ".." components that
    # a path_base pointing back up at the project root requires.
    cfg["path_base"] = os.path.relpath(root, out_dir.resolve())

    post = dict(cfg.get("posterior_sampling", {}))
    post.update({
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "checkpoint_every_steps": int(checkpoint_every),
    })
    if seed is not None:
        post["random_seed"] = int(seed)
    cfg["posterior_sampling"] = post

    # Prior sampling is cheap (_sample_prior defaults include_prediction=False, so
    # no ODE solves) but 100k rows of netcdf per cell is pointless.
    prior = dict(cfg.get("prior_sampling", {}))
    prior["draws"] = int(prior_draws)
    cfg["prior_sampling"] = prior

    if rtol is not None or atol is not None:
        ctrl = dict(cfg.get("ODE_stepsize_controller", {}))
        if rtol is not None:
            ctrl["rtol"] = float(rtol)
        if atol is not None:
            ctrl["atol"] = float(atol)
        cfg["ODE_stepsize_controller"] = ctrl
    if max_steps is not None:
        solver = dict(cfg.get("ODE_solver", {}))
        solver["max_steps"] = int(max_steps)
        cfg["ODE_solver"] = solver

    # The floor is a first-class key now, so it reaches inference_runner through
    # experiment_framework.load_experiment_bundle exactly as it reaches the probe.
    if floor and floor > 0:
        cfg["initial_condition_floor"] = float(floor)
    else:
        cfg.pop("initial_condition_floor", None)

    out_paths = dict(cfg.get("output_paths", {}))
    out_paths["results_save_dir"] = os.path.relpath(out_dir.resolve(), root)
    cfg["output_paths"] = out_paths

    dest = out_dir / "solver_params.json"
    dest.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return dest


def benchmark_dir(machine: str, device: str, floor: float, label: str, kind: str) -> Path:
    """Directory for one cell. kind is 'probe' or 'run'.

    Separate directories per kind on purpose: inference_runner's config_signature
    hashes posterior_sampling, so a probe (tune-only) and a run (100 draws) sharing
    a directory would invalidate each other's checkpoint on every alternation.
    """
    return project_root() / BENCHMARK_ROOT / "runs" / f"{cell_slug(machine, device, floor, label)}__{kind}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write a short-run benchmark config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--tune", type=int, default=200)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    print(write_benchmark_config(
        Path(a.config), Path(a.out_dir), draws=a.draws, tune=a.tune, chains=a.chains,
        floor=a.floor, rtol=a.rtol, atol=a.atol, max_steps=a.max_steps,
    ))
