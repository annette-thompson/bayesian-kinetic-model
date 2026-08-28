"""Generate a labeled solver_params.json variant for C4 speed testing.

Loads the base "Chain C4 - a2" config and overrides a handful of keys (ODE
tolerance, initial-condition floor, draw/tune/chain counts), writing the
result under Results/Benchmarks/<label>/solver_params.json. Running and
timing are left to inference_runner.py (writes timing.json) and
ode_step_diagnostics.py (writes ode_step_profile.json) -- this script's only
job is producing the config variant.

Usage (from the "Python Model" directory):
    python Utilities/bench_c4.py --label tol_tight --rtol 1e-6 --atol 1e-10
    python Utilities/bench_c4.py --label floor1 --floor 0.001 --tune 25 --draws 25 --chains 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / "Results" / "Chain Scaling Tests" / "Chain C4 - a2" / "solver_params.json"
BENCH_DIR = PROJECT_ROOT / "Results" / "Benchmarks"


def _path_base_for(run_dir: Path) -> str:
    """Relative path from run_dir back to the project root, for the config's path_base."""
    return os.path.relpath(PROJECT_ROOT, run_dir)


def build_variant(
    *,
    label: str,
    rtol: float | None = None,
    atol: float | None = None,
    floor: float | None = None,
    tune: int | None = None,
    draws: int | None = None,
    chains: int | None = None,
    max_steps: int | None = None,
    base_config: Path = BASE_CONFIG,
) -> dict:
    with open(base_config, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    if rtol is not None:
        config["ODE_stepsize_controller"]["rtol"] = rtol
    if atol is not None:
        config["ODE_stepsize_controller"]["atol"] = atol
    if max_steps is not None:
        config["ODE_solver"]["max_steps"] = max_steps
    if floor is not None:
        config["initial_condition_floor"] = floor
    if tune is not None:
        config["posterior_sampling"]["tune"] = tune
    if draws is not None:
        config["posterior_sampling"]["draws"] = draws
    if chains is not None:
        config["posterior_sampling"]["chains"] = chains

    run_dir = BENCH_DIR / label
    config["output_paths"]["results_save_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
    config["path_base"] = _path_base_for(run_dir)
    return config


def write_variant(config: dict, label: str) -> Path:
    run_dir = BENCH_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "solver_params.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="subdirectory name under Results/Benchmarks/")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--floor", type=float, default=None, help="initial_condition_floor value, e.g. 0.001")
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG, dest="base_config")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    variant = build_variant(
        label=args.label,
        rtol=args.rtol,
        atol=args.atol,
        floor=args.floor,
        tune=args.tune,
        draws=args.draws,
        chains=args.chains,
        max_steps=args.max_steps,
        base_config=args.base_config,
    )
    written_path = write_variant(variant, args.label)
    print(f"Wrote {written_path}")
