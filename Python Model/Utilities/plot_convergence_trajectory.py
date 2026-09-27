"""Plot convergence + model-comparison diagnostics for a finished (or
in-progress) resumable run, from its saved posterior netcdf:
convergence.png (r-hat/ESS trajectory, cumulative divergences --
pure "did it converge over draws", nothing per-chain), sampler_energy.png
(BFMI per chain plus the marginal/transition energy distributions it
summarizes), chain_mixing.png (per-chain rank ECDF, a mixing check that's more
informative than overlaid traces once there are multiple correlated
parameters), and leave_one_out.png (Pareto-k + elpd_loo/p_loo; WAIC is
omitted -- this arviz version dropped it in favor of PSIS-LOO).

Usage (from the "Python Model" directory):
    python Utilities/plot_convergence_trajectory.py --solver_params_file "Results/<run>/solver_params.json"
"""
from __future__ import annotations

import argparse

import arviz as az

from inference_plotting import (
    DIAGNOSTIC_FILES,
    ess_threshold_for,
    plot_convergence_diagnostics,
    plot_energy_diagnostics,
    plot_loo_diagnostics,
    plot_rank_diagnostics,
)
from inference_runner import import_solver_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True, help="Path to the run's solver_params.json/yaml.")
    parser.add_argument("--step", type=int, default=10, help="Draw-count spacing to recompute r-hat/ESS at (default 10).")
    parser.add_argument("--show", action="store_true", help="Display the plots interactively as well as saving them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)
    posterior_config = imported.solver_params.get("posterior_sampling", {})
    rhat_threshold = float(posterior_config.get("rhat_threshold", 1.01))
    free_params = [p["param_name"] for p in imported.solver_params.get("free_kinetic_params", [])]
    system_name = imported.results_save_dir.name

    nc_path = imported.results_save_dir / imported.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")

    inf_data = az.from_netcdf(nc_path)
    print(f"Run: {imported.results_save_dir}")
    print(f"Free params: {free_params}")

    # The ESS bar scales with chain count (see ess_threshold_for). Take the chain count
    # from the draws themselves rather than the config, so a run resumed with a different
    # chain count is still plotted against the bar it actually has to clear.
    n_chains = int(inf_data.posterior.sizes["chain"])
    ess_threshold, ess_source = ess_threshold_for(posterior_config, n_chains)
    print(f"ESS threshold: {ess_threshold:g}  ({ess_source})")

    result = plot_convergence_diagnostics(
        inf_data=inf_data,
        free_params=free_params,
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        step=args.step,
        # The run's own stopping rule, as in the trace plot's marker.
        consecutive=int(posterior_config.get("convergence_consecutive_checks", 1)),
        save_file=str(imported.results_save_dir / DIAGNOSTIC_FILES["convergence"]),
        system_name=system_name,
        show=args.show,
    )
    n_draws = result["draw_counts"][-1]
    print(f"Sampling draws available: {n_draws}")
    if result["criteria_met_at"] is not None:
        print(f"Criteria (r_hat<{rhat_threshold}, ess>={ess_threshold:g}) first met at draw {result['criteria_met_at']}")
    else:
        print(f"Criteria (r_hat<{rhat_threshold}, ess>={ess_threshold:g}) not yet met within {n_draws} draws.")
    print(f"Total divergences: {result['total_divergences']}")
    print(f"Saved: {result['plot_file']}")

    loo_result = plot_loo_diagnostics(
        inf_data=inf_data,
        save_file=str(imported.results_save_dir / DIAGNOSTIC_FILES["loo"]),
        system_name=system_name,
        show=args.show,
    )
    if loo_result["loo_result"] is not None:
        print(f"elpd_loo = {loo_result['loo_result'].elpd:.2f} +/- {loo_result['loo_result'].se:.2f}")
    else:
        print(f"LOO unavailable: {loo_result['error_message']}")
    print(f"Saved: {loo_result['plot_file']}")

    energy_result = plot_energy_diagnostics(
        inf_data=inf_data,
        save_file=str(imported.results_save_dir / DIAGNOSTIC_FILES["energy"]),
        system_name=system_name,
        show=args.show,
    )
    if energy_result["bfmi"].size:
        print(f"BFMI per chain: {energy_result['bfmi'].round(3).tolist()}")
    print(f"Saved: {energy_result['plot_file']}")

    try:
        rank_result = plot_rank_diagnostics(
            inf_data=inf_data,
            free_params=free_params,
            save_file=str(imported.results_save_dir / DIAGNOSTIC_FILES["rank"]),
            system_name=system_name,
            show=args.show,
        )
        print(f"Saved: {rank_result['plot_file']}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"Rank plot skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
