"""Every diagnostic figure for a finished run in one image, run_summary.png in the run folder.

The figures are the ones plot_trace_diagnostics.py, plot_convergence_trajectory.py and
plot_predictive_check.py save separately, drawn the same way, but each titled by its section
instead of the run; the run name appears once, at the top. Three rows, in the order a run is
checked: the sampler (convergence, beside chain mixing over sampler energy), the posterior
(priors, posteriors and traces, stretched to the page width), then the fit to the data
(posterior predictive checks beside LOO).

Usage (from the "Python Model" directory):
    python Utilities/plot_run_summary.py --solver_params_file "Results/<run>/solver_params.json"
"""
from __future__ import annotations

import argparse
import io

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from inference_plotting import (
    DIAGNOSTIC_TITLES,
    compute_convergence_criteria_met_at,
    ess_threshold_for,
    place_suptitle,
    plot_convergence_diagnostics,
    plot_energy_diagnostics,
    plot_loo_diagnostics,
    plot_posterior_trace_diagnostics,
    plot_predictive,
    plot_rank_diagnostics,
)
from inference_runner import _build_model_bundle, import_solver_params

DPI = 150
GAP_IN = 0.35                    # between sections
SECTION_TITLE = {"fontsize": 20, "fontweight": "bold"}
SECTIONS = DIAGNOSTIC_TITLES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True, help="Path to the run's solver_params.json/yaml.")
    parser.add_argument("--step", type=int, default=10, help="Draw-count spacing to recompute r-hat/ESS at (default 10).")
    return parser.parse_args()


def _render(fig, section: str) -> Image.Image:
    """The figure, retitled with its section heading, as an image at DPI."""
    place_suptitle(fig, SECTIONS[section], **SECTION_TITLE)
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _title(text: str) -> Image.Image:
    fig = plt.figure(figsize=(1, 1))
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=28, fontweight="bold")
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _stack(images: list[Image.Image], horizontal: bool, gap_px: int, align: str = "center") -> Image.Image:
    """Images side by side (or one above another), centred across the other direction, or
    aligned to its start (top / left) with align="start"."""
    if horizontal:
        width = sum(im.width for im in images) + gap_px * (len(images) - 1)
        height = max(im.height for im in images)
    else:
        width = max(im.width for im in images)
        height = sum(im.height for im in images) + gap_px * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for im in images:
        if horizontal:
            canvas.paste(im, (offset, 0 if align == "start" else (height - im.height) // 2))
            offset += im.width + gap_px
        else:
            canvas.paste(im, (0 if align == "start" else (width - im.width) // 2, offset))
            offset += im.height + gap_px
    return canvas


def main() -> None:
    args = _parse_args()
    imported = import_solver_params(args.solver_params_file)
    posterior_config = imported.solver_params.get("posterior_sampling", {})
    rhat_threshold = float(posterior_config.get("rhat_threshold", 1.01))
    consecutive = int(posterior_config.get("convergence_consecutive_checks", 1))
    free_params = [p["param_name"] for p in imported.solver_params.get("free_kinetic_params", [])]
    run_name = imported.results_save_dir.name

    nc_path = imported.results_save_dir / imported.posterior_samples_file
    if not nc_path.exists():
        raise SystemExit(f"No posterior netcdf found at {nc_path}. Has this run finished/finalized yet?")
    inf_data = az.from_netcdf(nc_path)
    ess_threshold, _ = ess_threshold_for(posterior_config, int(inf_data.posterior.sizes["chain"]))

    # As in plot_trace_diagnostics.py: the criterion is met on sampling draws, and the marker
    # sits on the combined warmup + sampling axis.
    sampling_only_at = compute_convergence_criteria_met_at(
        inf_data, free_params, rhat_threshold=rhat_threshold, ess_threshold=ess_threshold, consecutive=consecutive
    )
    n_tune = int(inf_data.warmup_posterior.sizes["draw"]) if hasattr(inf_data, "warmup_posterior") else 0
    criteria_met_at = None if sampling_only_at is None else sampling_only_at + n_tune

    images: dict[str, Image.Image] = {}
    images["convergence"] = _render(plot_convergence_diagnostics(
        inf_data=inf_data, free_params=free_params, rhat_threshold=rhat_threshold, ess_threshold=ess_threshold,
        step=args.step, consecutive=consecutive)["figure"], "convergence")
    try:
        images["rank"] = _render(plot_rank_diagnostics(inf_data=inf_data, free_params=free_params)["figure"], "rank")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"Rank section skipped ({type(exc).__name__}: {exc})")
    images["energy"] = _render(plot_energy_diagnostics(inf_data=inf_data)["figure"], "energy")
    images["loo"] = _render(plot_loo_diagnostics(inf_data=inf_data)["figure"], "loo")
    if getattr(inf_data, "posterior_predictive", None) is not None:
        print("Rebuilding model bundle (dataset/observed-value metadata only, no sampling)...")
        bundle = _build_model_bundle(imported)
        predictive = plot_predictive(inf_data=inf_data, experiment=bundle.experiment)["figure"]
        if predictive is not None:
            images["predictive"] = _render(predictive, "predictive")

    gap = int(GAP_IN * DPI)
    sampler_row = _stack([images["convergence"],
                          _stack([images[k] for k in ("rank", "energy") if k in images], False, gap)], True, gap)
    fit_row = _stack([images[k] for k in ("predictive", "loo") if k in images], True, gap, align="start")

    # The trace figure lays itself out again at the page's width, so the middle row spans it.
    # tight_layout's padding is trimmed on save, so ask for a little more than the page.
    page_in = max(sampler_row.width, fit_row.width) / DPI
    trace = plot_posterior_trace_diagnostics(
        inf_data=inf_data, free_params=free_params, include_tuning=True, criteria_met_at=criteria_met_at,
        use_log_param_axis=True)["figure"]
    trace.set_size_inches(page_in + 0.2, trace.get_size_inches()[1])
    trace.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    images["trace"] = _render(trace, "trace")

    page = _stack([_title(run_name), sampler_row, images["trace"], fit_row], False, gap)
    margin = gap
    framed = Image.new("RGB", (page.width + 2 * margin, page.height + 2 * margin), "white")
    framed.paste(page, (margin, margin))

    out = imported.results_save_dir / "run_summary.png"
    framed.save(out, dpi=(DPI, DPI))
    print(f"Saved: {out}  ({framed.width} x {framed.height} px)")


if __name__ == "__main__":
    main()
