"""Plot convergence diagnostics (r-hat/ESS trajectory, cumulative divergences,
BFMI) for a finished (or in-progress) resumable run, from its saved posterior
netcdf.

Usage (from the "Python Model" directory):
    python Utilities/plot_convergence_trajectory.py --solver_params_file "Results/<run>/solver_params.json"

Saves ``convergence_diagnostics.png`` into the run's results_save_dir.
"""
from __future__ import annotations

import argparse

import arviz as az

from inference_plotting import plot_convergence_diagnostics
from inference_runner import import_solver_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True, help="Path to the run's solver_params.json/yaml.")
    parser.add_argument("--step", type=int, default=10, help="Draw-count spacing to recompute r-hat/ESS at (default 10).")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively as well as saving it.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)
    posterior_config = imported.solver_params.get("posterior_sampling", {})
    rhat_threshold = float(posterior_config.get("rhat_threshold", 1.01))
    free_params = [p["param_name"] for p in imported.solver_params.get("free_kinetic_params", [])]

    nc_path = imported.results_save_dir / imported.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")

    inf_data = az.from_netcdf(nc_path)
    save_file = imported.results_save_dir / "convergence_diagnostics.png"
    result = plot_convergence_diagnostics(
        inf_data=inf_data,
        free_params=free_params,
        rhat_threshold=rhat_threshold,
        step=args.step,
        save_file=str(save_file),
        show=args.show,
    )

    print(f"Run: {imported.results_save_dir}")
    print(f"Free params: {free_params}")
    n_draws = result["draw_counts"][-1]
    print(f"Sampling draws available: {n_draws}")
    if result["criteria_met_at"] is not None:
        print(f"Criteria (r_hat<{rhat_threshold}, ess>=400) first met at draw {result['criteria_met_at']}")
    else:
        print(f"Criteria (r_hat<{rhat_threshold}, ess>=400) not yet met within {n_draws} draws.")
    print(f"Total divergences: {result['total_divergences']}")
    if result["bfmi"].size:
        print(f"BFMI per chain: {result['bfmi'].round(3).tolist()}")
    print(f"Saved: {result['plot_file']}")


if __name__ == "__main__":
    main()
