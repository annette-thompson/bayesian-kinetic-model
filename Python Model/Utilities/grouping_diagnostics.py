"""Does a set of individually-freed rate constants actually behave like the
single shared scaling parameter they were previously grouped under?

Reuses the reaction files' own declared nominal values (build_ode_system_from_
reactions' param_values, exactly what set_scaling_group_values overwrites) as
the reference point: for a scaling group, "no-op" is theta=1 per constant, so
dividing each constant's posterior by its own nominal value rescales every
constant onto the same unitless "implied group multiplier" axis, directly
comparable across constants even though their absolute nominal values differ.

If the grouping assumption holds, every constant's implied-multiplier
posterior should sit on top of the others (tight correlation, small manifold
deviation). If it doesn't, they spread apart -- letting the constants move
independently reveals structure the grouping masked.

Usage (from the "Python Model" directory):
    python Utilities/grouping_diagnostics.py \
        --solver_params_file "Results/.../Chain ... - split-diag/solver_params.json" \
        --constants g1a g1b
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az

from inference_runner import get_free_parameter_names, import_solver_params
from inference_plotting import _apply_plot_style, _kde_curve, plot_correlation_matrix
from reaction_model_builder import build_ode_system_from_reactions


def compute_implied_scale_factors(
    inf_data: az.InferenceData,
    constant_names: list[str],
    nominal_values: dict[str, float],
) -> dict[str, np.ndarray]:
    """Each constant's posterior, divided by its own nominal (no-op) value --
    puts every constant on the same unitless "implied shared multiplier" axis."""
    implied = {}
    for name in constant_names:
        draws = np.asarray(inf_data.posterior[name].values, dtype=float).reshape(-1)
        draws = draws[np.isfinite(draws)]
        nominal = float(nominal_values[name])
        if nominal <= 0:
            raise ValueError(f"Constant {name!r} has non-positive nominal value {nominal} -- "
                              "can't form a multiplicative implied scale factor from it.")
        implied[name] = draws / nominal
    return implied


def compute_manifold_deviation(
    implied: dict[str, np.ndarray],
    reference_posterior_sd_log: float | None = None,
) -> dict[str, Any]:
    """Per-draw spread of the implied scale factors across constants, in log
    space (so a constant sitting at 2x vs another at 0.5x -- symmetric
    deviations in ratio terms -- contribute equally). Near zero means every
    constant's implied multiplier agrees on every draw, i.e. the grouping
    holds; large means they disagree, i.e. it doesn't.

    ``reference_posterior_sd_log`` (optional): the single-grouped-parameter
    model's own posterior sd, in log space. The deviation only matters
    relative to how precisely a single shared parameter would itself be
    known -- report the ratio, not just the raw number.
    """
    names = list(implied)
    n_draws = min(len(v) for v in implied.values())
    stacked = np.stack([implied[name][:n_draws] for name in names], axis=0)  # (n_constants, n_draws)
    non_positive = stacked <= 0
    if non_positive.any():
        # Real scaling parameters are LogNormal (strictly positive); a non-positive
        # implied value means either a genuinely broken draw or a mismatched nominal
        # value, not something to silently np.log() into a NaN that poisons the mean.
        bad_draws = non_positive.any(axis=0)
        print(f"WARNING: {int(bad_draws.sum())}/{n_draws} draws had a non-positive implied "
              f"scale factor for at least one constant -- dropping those draws.")
        stacked = stacked[:, ~bad_draws]
    log_matrix = np.log(stacked)
    per_draw_spread = np.std(log_matrix, axis=0)  # (n_draws,)

    result: dict[str, Any] = {
        "constant_names": names,
        "n_draws": n_draws,
        "mean_log_spread": float(np.mean(per_draw_spread)),
        "median_log_spread": float(np.median(per_draw_spread)),
        "per_constant_median_implied": {name: float(np.median(implied[name])) for name in names},
    }
    if reference_posterior_sd_log is not None and reference_posterior_sd_log > 0:
        result["relative_to_grouped_precision"] = result["mean_log_spread"] / reference_posterior_sd_log
    return result


def plot_manifold_deviation(
    implied: dict[str, np.ndarray],
    save_file: str | None = None,
    show: bool = False,
    system_name: str | None = None,
):
    """Overlay each constant's implied-scale-factor density on one axis
    (peak-scaled to 1, same convention as the rest of inference_plotting.py)
    -- overlapping curves mean the grouping holds; separated curves mean it
    doesn't."""
    import matplotlib.pyplot as plt

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, draws in implied.items():
        if draws.size > 1 and np.std(draws) > 0:
            x, d = _kde_curve(draws, log_x=True)
            ax.plot(x, d / np.max(d), linewidth=1.8, label=name)
        elif draws.size:
            ax.axvline(float(draws[0]), linewidth=2.0, label=name)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Implied shared multiplier (posterior / own nominal value)")
    ax.set_ylabel("Relative density (each scaled to peak 1)")
    ax.set_title(f"{system_name} — Grouping Manifold Check" if system_name else "Grouping Manifold Check")
    ax.legend(loc="best")
    fig.tight_layout()

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)
    if not show:
        plt.close(fig)
    return {"figure": fig, "axes": ax, "plot_file": plot_file}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True,
                        help="The independently-freed (unpacked/split) model's solver_params.json.")
    parser.add_argument("--constants", nargs="+", default=None,
                        help="Which free parameters to check (default: all free_kinetic_params in the config).")
    parser.add_argument("--reference_solver_params_file", default=None,
                        help="Optional single-grouped-parameter model's solver_params.json, "
                        "for the relative-to-grouped-precision comparison.")
    parser.add_argument("--reference_param_name", default=None,
                        help="The grouped parameter's name in the reference run (required if "
                        "--reference_solver_params_file is given).")
    parser.add_argument("--save_file", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)
    sp = imported.solver_params
    _, _, _, nominal_values, _ = build_ode_system_from_reactions(
        imported.reactions_source, scaling_group=sp.get("scaling_groups"))

    constants = args.constants or get_free_parameter_names(sp)
    nc_path = imported.results_save_dir / imported.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")
    inf_data = az.from_netcdf(nc_path)

    implied = compute_implied_scale_factors(inf_data, constants, nominal_values)

    reference_sd_log = None
    if args.reference_solver_params_file:
        if not args.reference_param_name:
            raise SystemExit("--reference_param_name is required alongside --reference_solver_params_file")
        ref_imported = import_solver_params(args.reference_solver_params_file)
        ref_nc = ref_imported.results_save_dir / ref_imported.posterior_samples_file
        ref_data = az.from_netcdf(ref_nc)
        ref_draws = np.asarray(ref_data.posterior[args.reference_param_name].values, dtype=float).reshape(-1)
        reference_sd_log = float(np.std(np.log(ref_draws[np.isfinite(ref_draws) & (ref_draws > 0)])))

    result = compute_manifold_deviation(implied, reference_posterior_sd_log=reference_sd_log)

    print(f"Run: {imported.results_save_dir}")
    print(f"Constants checked: {constants}")
    print(f"Per-constant median implied multiplier: {result['per_constant_median_implied']}")
    print(f"Mean per-draw log-spread across constants: {result['mean_log_spread']:.4f}")
    print(f"Median per-draw log-spread across constants: {result['median_log_spread']:.4f}")
    if "relative_to_grouped_precision" in result:
        print(f"Relative to grouped model's own log-sd: {result['relative_to_grouped_precision']:.2f}x "
              "(<<1 supports grouping; >>1 argues against it)")

    system_name = imported.results_save_dir.name
    save_file = args.save_file or str(imported.results_save_dir / "grouping_manifold_check.png")
    plot_result = plot_manifold_deviation(implied, save_file=save_file, show=args.show, system_name=system_name)
    print(f"Saved: {plot_result['plot_file']}")

    if len(constants) >= 2:
        matrix_file = str(Path(save_file).with_name(Path(save_file).stem + "_correlation.png"))
        plot_correlation_matrix(inf_data, constants, save_file=matrix_file, system_name=system_name)
        print(f"Saved: {matrix_file}")


if __name__ == "__main__":
    main()
