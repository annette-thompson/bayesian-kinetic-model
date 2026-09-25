"""Plot the standard trace diagnostics for a finished (or in-progress)
resumable run, from its saved posterior netcdf -- the same trace_plot.png
artifact the notebook workflow produces, without needing to open the
notebook. (trace_plot_hist.pdf is intentionally not produced here -- the
prior/posterior panels in trace_plot.png already cover that.)

Usage (from the "Python Model" directory):
    python Utilities/plot_trace_diagnostics.py --solver_params_file "Results/<run>/solver_params.json"
"""
from __future__ import annotations

import argparse

import arviz as az

from inference_plotting import (
    compute_convergence_criteria_met_at,
    ess_threshold_for,
    plot_posterior_trace_diagnostics,
)
from inference_runner import import_solver_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True, help="Path to the run's solver_params.json/yaml.")
    parser.add_argument(
        "--no_tuning", action="store_true",
        help="Don't prepend warmup draws to the trace plot (shown by default, to see the chain initial-guess spread).",
    )
    parser.add_argument(
        "--linear_param_axis", action="store_true",
        help="Use a linear parameter-value axis instead of the default log axis -- free kinetic params are "
        "positive scale factors, and a wide prior next to a tight posterior is otherwise crushed into an "
        "invisible sliver on a linear scale.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plots interactively as well as saving them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)
    posterior_config = imported.solver_params.get("posterior_sampling", {})
    rhat_threshold = float(posterior_config.get("rhat_threshold", 1.01))
    free_params = [p["param_name"] for p in imported.solver_params.get("free_kinetic_params", [])]
    system_name = imported.results_save_dir.name
    include_tuning = not args.no_tuning

    nc_path = imported.results_save_dir / imported.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")

    inf_data = az.from_netcdf(nc_path)

    # The criterion is evaluated on SAMPLING draws only -- warmup is adaptation,
    # not draws from the target, and including it makes the number depend on how
    # much warmup was run. When the trace shows the combined timeline the marker
    # is offset by the warmup length so it lands in the right visual place.
    consecutive = int(posterior_config.get("convergence_consecutive_checks", 1))
    # The ESS half of the criterion scales with chain count (see ess_threshold_for);
    # leaving it at the old flat 400 would put this marker earlier than the trace
    # actually earns it on any run with more than four chains.
    ess_threshold, ess_source = ess_threshold_for(posterior_config, int(inf_data.posterior.sizes["chain"]))
    print(f"ESS threshold: {ess_threshold:g}  ({ess_source})")
    sampling_only_at = compute_convergence_criteria_met_at(
        inf_data, free_params, rhat_threshold=rhat_threshold, ess_threshold=ess_threshold,
        consecutive=consecutive
    )
    n_tune = int(inf_data.warmup_posterior.sizes["draw"]) if (
        include_tuning and hasattr(inf_data, "warmup_posterior")
    ) else 0
    criteria_met_at = None if sampling_only_at is None else sampling_only_at + n_tune

    trace_result = plot_posterior_trace_diagnostics(
        inf_data=inf_data,
        free_params=free_params,
        save_file=str(imported.results_save_dir / "trace_plot.png"),
        include_tuning=include_tuning,
        criteria_met_at=criteria_met_at,
        system_name=system_name,
        show=args.show,
        use_log_param_axis=not args.linear_param_axis,
    )

    print(f"Run: {imported.results_save_dir}")
    print(f"Free params: {free_params}")
    if sampling_only_at is not None:
        print(f"Criteria met at {sampling_only_at} sampling draws "
              f"(sustained over {consecutive} consecutive checks; warmup excluded)")
    else:
        print("Criteria never met within the saved sampling draws")
    print(f"Saved: {trace_result['plot_file']}")


if __name__ == "__main__":
    main()
