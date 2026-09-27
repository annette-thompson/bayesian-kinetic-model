"""Draft result figures from finished Tier-1 runs, in the diagnostic figures' style (Title
Case, µM, 16 pt, legends below the axes, "<run> — <title>" titles).

  python tier1_result_figures.py fig2 "Tier1 C14+unsat - a1c3"  # posterior vs truth (Fig 2 fallback)
  python tier1_result_figures.py fig4     # robustness to noise level and prior shift (R0 + R5)
  python tier1_result_figures.py r7       # sampled vs predicted shrinkage (R1 + R7 against the grid)
  python tier1_result_figures.py r8       # target_accept 0.95 and rtol 1e-5 against R0
  python tier1_result_figures.py all      # every figure with at least one finished run

Figures go to Results/Tier1/figures/. A run that has not finalized is left out and named in
the figure's footnote, so each figure can be drawn as soon as its first runs finish.

Scores come from recovery_report.analyse(): posterior median and 95% interval, z, shrinkage
on the log scale (independent of where the prior median sits), and bulk ESS for the Monte
Carlo error of a posterior mean (sd / sqrt(ESS)).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inference_plotting import (  # noqa: E402
    PLOT_FONT_SIZE,
    THRESHOLD_STYLE,
    _apply_plot_style,
    _kde_curve,
    _threshold_lines,
    figure_renderer,
    place_suptitle,
)
from recovery_report import RESULTS, analyse  # noqa: E402

OUT = RESULTS / "figures"
PARAM_COLOR = {"a1": "tab:blue", "c3": "tab:green", "a2": "tab:purple", "d1": "tab:blue", "d2": "tab:green"}
TRUTH_STYLE = {"color": "0.35", "linestyle": "-", "linewidth": 1.2}
R0 = "Tier1 C8 - a1c3"
NOISE = [("5%", "Tier1 C8_noise5 - a1c3"), ("10%", R0), ("20%", "Tier1 C8_noise20 - a1c3"),
         ("40%", "Tier1 C8_noise40 - a1c3")]
SHIFT = [(0, R0)] + [(k, f"Tier1 C8 - a1c3 - prior+{k}sd") for k in (1, 2, 3, 4)]
# The paired series, started at the ME1 values instead of each prior's mean (R0 is both).
SHIFT_INIT1 = [(0, R0)] + [(k, f"Tier1 C8 - a1c3 - prior+{k}sd - init1") for k in (1, 2, 3, 4)]
R7_CELLS = [("Full Data", "Tier1 C14+unsat - a1c3", "full"),
            ("Profile Only", "Tier1 C14+unsat - a1c3 - profile", "profile"),
            ("Rates Only", "Tier1 C14+unsat - a1c3 - rates", "rates")]
R7_GRID = HERE / "expected_information_grid_a1c3.json"
R8_VARIANTS = [("Target Accept 0.95", "Tier1 C8 - a1c3 - ta0.95"), ("rtol 1e-5", "Tier1 C8 - a1c3 - rtol1e-5")]


def _scores(runs):
    """{run: analyse() record} for the finalized runs; the names of the rest."""
    done, missing = {}, []
    for run in dict.fromkeys(runs):
        rec = analyse(RESULTS / run) if (RESULTS / run).is_dir() else {"skipped": "no run folder"}
        if rec.get("skipped"):
            missing.append(run)
        else:
            done[run] = rec
    return done, missing


def _footnote(fig, missing, legend=None):
    """Name the runs left out, below the figure legend when there is one."""
    if not missing:
        return
    y = -0.02
    if legend is not None:
        renderer = figure_renderer(fig)
        y = float(fig.transFigure.inverted().transform((0, legend.get_window_extent(renderer).y0))[1]) - 0.01
    import textwrap
    text = "Not finished yet, so not shown: " + ", ".join(m.replace("Tier1 ", "") for m in missing)
    width = int(fig.get_size_inches()[0] * 72 / ((PLOT_FONT_SIZE - 2) * 0.55))   # characters that fit
    fig.text(0.01, y, textwrap.fill(text, width), ha="left", va="top", fontsize=PLOT_FONT_SIZE - 2, color="0.35")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def _prior_median(run):
    cfg = json.loads((RESULTS / run / "solver_params.json").read_text())
    pr = cfg["free_kinetic_params"][0]["prior_dist_params"]
    return math.sqrt(float(pr["lower"]) * float(pr["upper"]))


def fig2(run):
    """Posterior against truth, one panel per parameter, on a log axis."""
    import arviz as az
    rec, missing = _scores([run])
    if missing:
        print(f"{run} has not finalized; nothing to draw.")
        return None
    rec = rec[run]
    post = az.from_netcdf(RESULTS / run / "posterior_samples_pm.nc").posterior
    params = list(rec["params"])
    _apply_plot_style()
    fig, axes = plt.subplots(1, len(params), figsize=(7.0 * len(params), 4.6), squeeze=False)
    for ax, p in zip(axes[0], params):
        s = rec["params"][p]
        x = np.asarray(post[p].values, float).ravel()
        log_x = bool(np.all(x > 0))
        gx, gd = _kde_curve(x, log_x)
        color = PARAM_COLOR.get(p, "tab:blue")
        lo, hi = s["ci95"]
        inside = (gx >= lo) & (gx <= hi)
        ax.fill_between(gx[inside], gd[inside], color=color, alpha=0.3, linewidth=0, label="95% Interval")
        ax.plot(gx, gd, color=color, linewidth=2.0, label="Posterior")
        if s.get("truth") is not None:
            ax.axvline(s["truth"], **TRUTH_STYLE, label="Truth")
        if log_x:
            ax.set_xscale("log")
            from inference_plotting import _plain_ticks_if_narrow_log
            _plain_ticks_if_narrow_log(ax, "x")
        ax.set_ylim(bottom=0)
        ax.set_title(p)
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        text = (f"z = {s['z']:+.2f}\nShrinkage (Log) = {s['shrinkage_log']:.3f}" if "z" in s and "shrinkage_log" in s
                else "")
        ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top", ha="left", fontsize=PLOT_FONT_SIZE - 2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    fig.tight_layout()
    place_suptitle(fig, f"{run} — Posterior vs Truth")
    return _save(fig, f"posterior_vs_truth_{run.replace('Tier1 ', '').replace(' - ', '_').replace(' ', '_')}.png")


def _interval_panel(ax, labels, runs, done, params, shift=0.0, open_markers=False, suffix=""):
    """Posterior median and 95% interval per condition, one offset marker per parameter."""
    offsets = np.linspace(-0.15, 0.15, len(params)) + shift
    for off, p in zip(offsets, params):
        xs, med, lo, hi = [], [], [], []
        for i, run in enumerate(runs):
            s = done.get(run, {}).get("params", {}).get(p)
            if not s:
                continue
            xs.append(i + off)
            med.append(s["median"])
            lo.append(s["median"] - s["ci95"][0])
            hi.append(s["ci95"][1] - s["median"])
        if xs:
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=PARAM_COLOR.get(p), capsize=4, markersize=7,
                        linewidth=2, label=p + suffix, markerfacecolor="white" if open_markers else PARAM_COLOR.get(p),
                        linestyle="none")
    if not shift:
        ax.axhline(1.0, **TRUTH_STYLE, label="Truth")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)


def _log_interval_axis(ax):
    """Log y-axis with plain decimal ticks when narrow; call once, after every series is drawn
    (setting the scale again rescales the axis)."""
    from inference_plotting import _plain_ticks_if_narrow_log
    ax.set_yscale("log")
    _plain_ticks_if_narrow_log(ax, "y")


def _shrink_panel(ax, labels, runs, done, params, open_markers=False, suffix=""):
    for p in params:
        xs, ys = [], []
        for i, run in enumerate(runs):
            s = done.get(run, {}).get("params", {}).get(p)
            if s and "shrinkage_log" in s:
                xs.append(i)
                ys.append(s["shrinkage_log"])
        if xs:
            ax.plot(xs, ys, "o--" if open_markers else "o-", color=PARAM_COLOR.get(p), linewidth=2, markersize=7,
                    label=p + suffix, markerfacecolor="white" if open_markers else PARAM_COLOR.get(p))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylabel("Shrinkage (Log Scale)")


def fig4():
    """Robustness: posterior intervals and log-scale shrinkage against noise level and prior shift."""
    runs = [r for _, r in NOISE] + [r for _, r in SHIFT] + [r for _, r in SHIFT_INIT1]
    done, missing = _scores(runs)
    if not done:
        print("No robustness run has finalized yet.")
        return None
    params = list(next(iter(done.values()))["params"])
    shift_labels = []
    for k, run in SHIFT:
        med = _prior_median(run) if (RESULTS / run / "solver_params.json").exists() else float("nan")
        shift_labels.append(f"+{k} sd\n(prior {med:.3g})" if k else "0\n(prior 1)")
    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), gridspec_kw={"width_ratios": [4, 5]})
    _interval_panel(axes[0, 0], [l for l, _ in NOISE], [r for _, r in NOISE], done, params)
    _interval_panel(axes[0, 1], shift_labels, [r for _, r in SHIFT], done, params, shift=-0.2,
                    suffix=" (Default Start)")
    _interval_panel(axes[0, 1], shift_labels, [r for _, r in SHIFT_INIT1], done, params, shift=0.2,
                    open_markers=True, suffix=" (Started at ME1 Values)")
    axes[0, 1].axhline(1.0, **TRUTH_STYLE, label="Truth")
    for ax in axes[0]:
        _log_interval_axis(ax)
    _shrink_panel(axes[1, 0], [l for l, _ in NOISE], [r for _, r in NOISE], done, params)
    _shrink_panel(axes[1, 1], shift_labels, [r for _, r in SHIFT], done, params, suffix=" (Default Start)")
    _shrink_panel(axes[1, 1], shift_labels, [r for _, r in SHIFT_INIT1], done, params, open_markers=True,
                  suffix=" (Started at ME1 Values)")
    axes[0, 0].set_title("Noise Level")
    axes[0, 1].set_title("Prior Median Shift")
    axes[0, 0].set_ylabel("Posterior Median and 95% Interval")
    axes[1, 0].set_xlabel("Measurement Noise (Sd, % of Value)")
    axes[1, 1].set_xlabel("Prior Median Shift (Prior Sd)")
    for ax in axes[1]:
        vals = [v for line in ax.get_lines() for v in line.get_ydata()]
        ax.set_ylim(min([0.5] + [v - 0.05 for v in vals]), 1.0)
    # One legend for the figure: the prior-shift panel carries every series.
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C8 - a1c3 — Robustness to Noise Level and Prior Shift")
    return _save(fig, "robustness.png")


def r7():
    """Sampled log-scale shrinkage against the expected-information grid's prediction."""
    grid = json.loads(R7_GRID.read_text())["tier1_design_parts"]
    runs = [r for _, r, _ in R7_CELLS]
    done, missing = _scores(runs)
    params = ["a1", "c3"]
    _apply_plot_style()
    fig, axes = plt.subplots(1, len(params), figsize=(14.0, 5.2), sharey=True)
    width = 0.36
    for ax, p in zip(axes, params):
        x = np.arange(len(R7_CELLS))
        pred = [grid[part]["shrinkage_log"][p] for _, _, part in R7_CELLS]
        samp = [done.get(run, {}).get("params", {}).get(p, {}).get("shrinkage_log", np.nan) for _, run, _ in R7_CELLS]
        color = PARAM_COLOR.get(p)
        ax.bar(x - width / 2, pred, width=width, facecolor="white", edgecolor=color, hatch="///", linewidth=1.5,
               label="Predicted (Expected Information)")
        ax.bar(x + width / 2, samp, width=width, color=color, alpha=0.7, label="Sampled")
        for xi, v in zip(x, pred):
            ax.text(xi - width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=PLOT_FONT_SIZE - 4)
        for xi, v in zip(x, samp):
            if np.isfinite(v):
                ax.text(xi + width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=PLOT_FONT_SIZE - 4)
        ax.set_xticks(x, [label for label, _, _ in R7_CELLS])
        ax.set_title(p)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Shrinkage (Log Scale)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    from matplotlib.patches import Patch
    legend = fig.legend(handles=[Patch(facecolor="white", edgecolor="0.3", hatch="///", label="Predicted (Expected Information)"),
                                 Patch(color="0.45", alpha=0.7, label="Sampled")],
                        loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C14+unsat - a1c3 — Data-Type Check: Sampled vs Predicted Shrinkage")
    return _save(fig, "data_type_check.png")


def r8():
    """Do target_accept 0.95 and rtol 1e-5 move the posterior relative to R0?"""
    done, missing = _scores([R0] + [r for _, r in R8_VARIANTS])
    if R0 not in done:
        print("R0 has not finalized; nothing to compare against.")
        return None
    params = list(done[R0]["params"])
    base = done[R0]["params"]
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))
    offsets = np.linspace(-0.12, 0.12, len(params))
    for off, p in zip(offsets, params):
        xs, dz, ratio = [], [], []
        for i, (_, run) in enumerate(R8_VARIANTS):
            s = done.get(run, {}).get("params", {}).get(p)
            if not s:
                continue
            b = base[p]
            mcse = math.sqrt(s["sd"] ** 2 / s["ess_bulk"] + b["sd"] ** 2 / b["ess_bulk"])
            xs.append(i + off)
            dz.append((s["mean"] - b["mean"]) / mcse)
            ratio.append(s["sd"] / b["sd"])
        axes[0].plot(xs, dz, "o", color=PARAM_COLOR.get(p), markersize=9, label=p)
        axes[1].plot(xs, ratio, "o", color=PARAM_COLOR.get(p), markersize=9, label=p)
    h0 = _threshold_lines(axes[0], [-2.0, 2.0])
    axes[0].axhline(0.0, **TRUTH_STYLE)
    _threshold_lines(axes[1], [0.9, 1.1])
    axes[1].axhline(1.0, **TRUTH_STYLE)
    for ax in axes:
        ax.set_xticks(range(len(R8_VARIANTS)), [label for label, _ in R8_VARIANTS])
        ax.set_xlim(-0.5, len(R8_VARIANTS) - 0.5)
    axes[0].set_title("Posterior Mean Shift")
    axes[0].set_ylabel("(Variant − R0) / Monte Carlo Error")
    axes[1].set_title("Posterior Width")
    axes[1].set_ylabel("Posterior Sd, Variant / R0")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    legend = fig.legend(handles + [h0], labels + ["Threshold"], loc="upper center", bbox_to_anchor=(0.5, 0.04),
                        ncol=len(labels) + 1)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C8 - a1c3 — Sampler and Solver Settings Check")
    return _save(fig, "settings_check.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["fig2", "fig4", "r7", "r8", "all"])
    ap.add_argument("run", nargs="?", default="Tier1 C14+unsat - a1c3", help="fig2's run (default R1)")
    a = ap.parse_args()
    if a.figure in ("fig2", "all"):
        fig2(a.run)
    if a.figure in ("fig4", "all"):
        fig4()
    if a.figure in ("r7", "all"):
        r7()
    if a.figure in ("r8", "all"):
        r8()


if __name__ == "__main__":
    main()
