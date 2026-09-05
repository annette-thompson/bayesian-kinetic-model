"""Plot convergence + model-comparison diagnostics for a finished (or
in-progress) resumable run, from its saved posterior netcdf:
convergence_diagnostics.png (r-hat/ESS trajectory, cumulative divergences --
pure "did it converge over draws", nothing per-chain), energy_plot.png
(BFMI per chain plus the marginal/transition energy distributions it
summarizes), rank_plot.png (per-chain rank ECDF, a mixing check that's more
informative than overlaid traces once there are multiple correlated
parameters), and loo_diagnostics.png (Pareto-k + elpd_loo/p_loo; WAIC is
omitted -- this arviz version dropped it in favor of PSIS-LOO).

Usage (from the "Python Model" directory):
    python Utilities/plot_convergence_trajectory.py --solver_params_file "Results/<run>/solver_params.json"
"""
from __future__ import annotations

import argparse

import arviz as az

from inference_plotting import plot_convergence_diagnostics, plot_energy_diagnostics, plot_loo_diagnostics
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

    result = plot_convergence_diagnostics(
        inf_data=inf_data,
        free_params=free_params,
        rhat_threshold=rhat_threshold,
        step=args.step,
        save_file=str(imported.results_save_dir / "convergence_diagnostics.png"),
        system_name=system_name,
        show=args.show,
    )
    n_draws = result["draw_counts"][-1]
    print(f"Sampling draws available: {n_draws}")
    if result["criteria_met_at"] is not None:
        print(f"Criteria (r_hat<{rhat_threshold}, ess>=400) first met at draw {result['criteria_met_at']}")
    else:
        print(f"Criteria (r_hat<{rhat_threshold}, ess>=400) not yet met within {n_draws} draws.")
    print(f"Total divergences: {result['total_divergences']}")
    print(f"Saved: {result['plot_file']}")

    loo_result = plot_loo_diagnostics(
        inf_data=inf_data,
        save_file=str(imported.results_save_dir / "loo_diagnostics.png"),
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
        save_file=str(imported.results_save_dir / "energy_plot.png"),
        system_name=system_name,
        show=args.show,
    )
    if energy_result["bfmi"].size:
        print(f"BFMI per chain: {energy_result['bfmi'].round(3).tolist()}")
    print(f"Saved: {energy_result['plot_file']}")

    # az.plot_rank uses arviz's newer PlotCollection-based backend (not a
    # simple matplotlib ax=), so it saves its own standalone figure rather
    # than being composited into the figures above.
    try:
        rank_pc = az.plot_rank(inf_data, var_names=free_params, backend="matplotlib")
        # arviz's rank-ECDF plot leaves the y-axis unlabeled and has no
        # legend by default; the PlotCollection exposes the real matplotlib
        # Axes/Line2D/PolyCollection objects under viz["plot"]/["ecdf_lines"]/
        # ["credible_interval"], so all of these are added directly before
        # saving.
        for var_name in free_params:
            try:
                ax = rank_pc.viz["plot"][var_name].item()
                ax.set_ylabel("ECDF - uniform")
                envelope = rank_pc.viz["credible_interval"][var_name].item()
                envelope.set_label("95% envelope\n(expected if well-mixed)")
                ax.legend(handles=[envelope], loc="upper right", fontsize="small")
                ax.add_artist(ax.get_legend())
                lines = rank_pc.viz["ecdf_lines"][var_name].values
                ax.legend(
                    handles=list(lines),
                    labels=[str(i) for i in range(len(lines))],
                    title="Chains",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=min(len(lines), 12),
                    handlelength=1.0,
                    columnspacing=0.8,
                    handletextpad=0.4,
                    fontsize="small",
                )
            except (KeyError, AttributeError):
                pass
        rank_file = imported.results_save_dir / "rank_plot.png"
        rank_pc.savefig(str(rank_file))
        print(f"Saved: {rank_file}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"Rank plot skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
