"""Plot posterior-predictive checks -- observed data vs. model predictions
drawn from the posterior -- for a finished resumable run, from its saved
posterior netcdf. Reuses inference_plotting.plot_predictive as-is (same
function the notebook workflow calls); this just rebuilds the (cheap, no
sampling) model bundle needed for dataset/observed-value metadata and points
it at the saved netcdf instead of an in-memory run.

Usage (from the "Python Model" directory):
    python Utilities/plot_predictive_check.py --solver_params_file "Results/<run>/solver_params.json"

Saves a single combined PNG into the run's results_save_dir -- one row per
chain-length observable, time-course panel next to its final-concentration/
table panel -- named predictive_plots_<system>.png.
"""
from __future__ import annotations

import argparse

import arviz as az

from inference_plotting import plot_predictive
from inference_runner import _build_model_bundle, import_solver_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True, help="Path to the run's solver_params.json/yaml.")
    parser.add_argument("--show", action="store_true", help="Display the plots interactively as well as saving them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)

    nc_path = imported.results_save_dir / imported.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")

    inf_data = az.from_netcdf(nc_path)
    if getattr(inf_data, "posterior_predictive", None) is None:
        raise SystemExit(
            f"{nc_path} has no posterior_predictive group -- it was written by an older "
            "inference_runner.py, or finalize didn't complete. Nothing to plot."
        )

    system_name = imported.results_save_dir.name
    print(f"Run: {imported.results_save_dir}")
    print("Rebuilding model bundle (dataset/observed-value metadata only, no sampling)...")
    bundle = _build_model_bundle(imported)

    result = plot_predictive(
        inf_data=inf_data,
        experiment=bundle.experiment,
        save_dir=str(imported.results_save_dir),
        show=args.show,
        system_name=system_name,
    )

    if result["plot_file"] is None:
        print("No predictive figure produced (no posterior_predictive data found for these datasets).")
    else:
        print(f"Saved: {result['plot_file']}")


if __name__ == "__main__":
    main()
