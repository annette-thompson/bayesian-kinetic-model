"""Draft result figures from finished Tier-1 runs, in the diagnostic figures' style (Title
Case, µM, 16 pt, legends below the axes, "<run> — <title>" titles).

  python tier1_result_figures.py fig2 "Tier1 C14+unsat - a1c3"  # posterior vs truth (Fig 2 fallback)
  python tier1_result_figures.py fig4     # robustness to noise level and prior shift (R0 + R5)
  python tier1_result_figures.py fig4_fit    # with fig4: data vs fitted curve at each noise level
  python tier1_result_figures.py fig4_prior  # with fig4: each shifted prior against its posterior
  python tier1_result_figures.py fig4_si  # SI: prior shift from the default start vs the ME1 start
  python tier1_result_figures.py fig5     # c3 grouping test: grouped vs split on standard and 1:3 data (R1 + R6)
  python tier1_result_figures.py r7       # sampled vs predicted contraction (R1 + R7 against the grid)
  python tier1_result_figures.py r8       # target_accept 0.95 and rtol 1e-5 against R0
  python tier1_result_figures.py all      # every figure with at least one finished run

Figures go to Results/Tier1/figures/. A run that has not finalized is left out and named in
the figure's footnote, so each figure can be drawn as soon as its first runs finish.

Scores come from recovery_report.analyse(): posterior median and 95% interval; z and posterior
contraction (1 - posterior variance / prior variance) on the log scale, independent of where
the prior median sits; and bulk ESS for the Monte Carlo error of a posterior mean
(sd / sqrt(ESS)).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))
sys.path.insert(0, str(HERE))

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
NOISE_PCT = [5, 10, 20, 40]        # x positions: the noise axis is spaced by value, not evenly
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
        text = (f"z = {s['z']:+.2f}\nContraction = {s['contraction']:.4f}" if s.get("z") is not None
                and "contraction" in s else "")
        ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top", ha="left", fontsize=PLOT_FONT_SIZE - 2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    fig.tight_layout()
    place_suptitle(fig, f"{run} — Posterior vs Truth")
    return _save(fig, f"posterior_vs_truth_{run.replace('Tier1 ', '').replace(' - ', '_').replace(' ', '_')}.png")


def _positions(labels, positions):
    """x positions for a panel's conditions (evenly spaced unless given), the smallest gap between
    them, and x limits half a gap beyond the ends."""
    pos = np.arange(len(labels), dtype=float) if positions is None else np.asarray(positions, dtype=float)
    gap = float(np.min(np.diff(pos))) if len(pos) > 1 else 1.0
    return pos, gap, (pos[0] - gap / 2, pos[-1] + gap / 2)


def _interval_panel(ax, labels, runs, done, params, shift=0.0, open_markers=False, suffix="", positions=None):
    """Posterior median and 95% interval per condition, one offset marker per parameter."""
    pos, gap, xlim = _positions(labels, positions)
    offsets = (np.linspace(-0.15, 0.15, len(params)) + shift) * gap
    for off, p in zip(offsets, params):
        xs, med, lo, hi = [], [], [], []
        for i, run in enumerate(runs):
            s = done.get(run, {}).get("params", {}).get(p)
            if not s:
                continue
            xs.append(pos[i] + off)
            med.append(s["median"])
            lo.append(s["median"] - s["ci95"][0])
            hi.append(s["ci95"][1] - s["median"])
        if xs:
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="o", color=PARAM_COLOR.get(p), capsize=4, markersize=7,
                        linewidth=2, label=p + suffix, markerfacecolor="white" if open_markers else PARAM_COLOR.get(p),
                        linestyle="none")
    if not shift:
        ax.axhline(1.0, **TRUTH_STYLE, label="Truth")
    ax.set_xticks(pos, labels)
    ax.set_xlim(*xlim)


def _log_interval_axis(ax):
    """Log y-axis with plain decimal ticks when narrow; call once, after every series is drawn
    (setting the scale again rescales the axis)."""
    from inference_plotting import _plain_ticks_if_narrow_log
    ax.set_yscale("log")
    _plain_ticks_if_narrow_log(ax, "y")


def _contraction_panel(ax, labels, runs, done, params, open_markers=False, suffix="", positions=None):
    pos, _, xlim = _positions(labels, positions)
    for p in params:
        xs, ys = [], []
        for i, run in enumerate(runs):
            s = done.get(run, {}).get("params", {}).get(p)
            if s and "contraction" in s:
                xs.append(pos[i])
                ys.append(s["contraction"])
        if xs:
            ax.plot(xs, ys, "o--" if open_markers else "o-", color=PARAM_COLOR.get(p), linewidth=2, markersize=7,
                    label=p + suffix, markerfacecolor="white" if open_markers else PARAM_COLOR.get(p))
    ax.set_xticks(pos, labels)
    ax.set_xlim(*xlim)
    ax.set_ylabel("Posterior Contraction (Log Scale)")


def _z_panel(ax, labels, runs, done, params, open_markers=False, suffix="", positions=None):
    """Log-scale z per condition, one line per parameter."""
    pos, _, xlim = _positions(labels, positions)
    for p in params:
        xs, ys = [], []
        for i, run in enumerate(runs):
            s = done.get(run, {}).get("params", {}).get(p)
            if s and s.get("z") is not None:
                xs.append(pos[i])
                ys.append(s["z"])
        if xs:
            ax.plot(xs, ys, "o--" if open_markers else "o-", color=PARAM_COLOR.get(p), linewidth=2, markersize=7,
                    label=p + suffix, markerfacecolor="white" if open_markers else PARAM_COLOR.get(p))
    ax.set_xticks(pos, labels)
    ax.set_xlim(*xlim)
    ax.set_ylabel("z (Log Scale)")


def _shift_labels():
    labels = []
    for k, run in SHIFT:
        med = _prior_median(run) if (RESULTS / run / "solver_params.json").exists() else float("nan")
        labels.append(f"+{k} sd\n(prior {med:.3g})" if k else "0\n(prior 1)")
    return labels


def _share_row_ylim(axes):
    """One y range for a row of panels (the union of their own)."""
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def fig4():
    """Robustness, from the default start: posterior interval, contraction and z against noise level
    and prior shift. The ME1-start comparison is fig4_si."""
    runs = [r for _, r in NOISE] + [r for _, r in SHIFT]
    done, missing = _scores(runs)
    if not done:
        print("No robustness run has finalized yet.")
        return None
    params = list(next(iter(done.values()))["params"])
    shift_labels = _shift_labels()
    columns = [([l for l, _ in NOISE], [r for _, r in NOISE], NOISE_PCT), (shift_labels, [r for _, r in SHIFT], None)]
    _apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 13.5), gridspec_kw={"width_ratios": [4, 5]})
    for col, (labels, col_runs, positions) in enumerate(columns):
        _interval_panel(axes[0, col], labels, col_runs, done, params, positions=positions)
        _contraction_panel(axes[1, col], labels, col_runs, done, params, positions=positions)
        _z_panel(axes[2, col], labels, col_runs, done, params, positions=positions)
    for ax in axes[0]:
        _log_interval_axis(ax)
    _share_row_ylim(axes[0])
    for ax in axes[0]:
        # narrow log axes read better as plain decimals; reapply once the shared range is set
        from inference_plotting import _plain_ticks_if_narrow_log
        _plain_ticks_if_narrow_log(ax, "y")
    for ax in axes[1]:
        vals = [v for line in ax.get_lines() for v in line.get_ydata()]
        # Contraction sits close to 1 with data this informative: 0.95-1.005 shows the differences,
        # extended down only if a point falls below it.
        ax.set_ylim(min([0.95] + [v - 0.005 for v in vals]), 1.005)
    _share_row_ylim(axes[1])
    zmax = max([2.5] + [abs(v) + 0.3 for ax in axes[2] for line in ax.get_lines() for v in line.get_ydata()])
    handle = None
    for ax in axes[2]:
        ax.set_ylim(-zmax, zmax)
        ax.axhline(0, color="0.6", linewidth=1)
        handle = _threshold_lines(ax, [-2, 2])
    axes[0, 0].set_title("Noise Level")
    axes[0, 1].set_title("Prior Median Shift")
    axes[0, 0].set_ylabel("Posterior Median and 95% Interval")
    axes[1, 1].set_ylabel("")
    axes[2, 1].set_ylabel("")
    axes[0, 1].set_ylabel("")
    axes[2, 0].set_xlabel("Measurement Noise (Sd, % of Value)")
    axes[2, 1].set_xlabel("Prior Median Shift (Prior Sd)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(handle)
    labels.append("Threshold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.025), ncol=len(labels))
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C8 - a1c3 — Robustness to Noise Level and Prior Shift")
    return _save(fig, "robustness.png")


FIT_DRAWS = 128
TRUTH_BAR = "0.86"      # noise-free value as a bar under the profile and rate points
CONDITION_SHORT = {"baseline": "Base-\nline", "FabH 0.1 uM": "FabH\n0.1 µM", "FabB 0 uM": "FabB\n0",
                   "TesA 0.5 uM": "TesA\n0.5 µM", "FabZ 0 uM": "FabZ\n0"}


def _fitted_predictions(runs, n_draws=FIT_DRAWS):
    """Noise-free model predictions for each run's posterior draws, and at the truth, from the
    fit's own simulator (the likelihood's mean function). The runs must share one design (the
    noise levels differ only in their data), so one simulator serves all -- and one compile,
    which the laptop's XLA:CPU needs. Cached per run in figures/.cache by posterior-file mtime.

    Returns ({run: (draws, observations)}, truth (observations,), {run: experiment bundle})."""
    import arviz as az
    import jax
    import jax.numpy as jnp
    import check_model_vs_data as cmd
    cache = OUT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    preds, exps, sim, names, values = {}, {}, None, None, None
    truth = None
    for run in runs:
        cfg_path = RESULTS / run / "solver_params.json"
        cfg = json.loads(cfg_path.read_text())
        imported = cmd.ir.import_solver_params(cfg_path)
        ode, species, run_names, run_values, _ = cmd.build_ode_system_from_reactions(
            imported.reactions_source, scaling_group=cfg["scaling_groups"])
        exps[run] = cmd.load_experiment_bundle(solver_params=cfg, solver_params_file=str(cfg_path),
                                               species_names=species)
        nc = RESULTS / run / "posterior_samples_pm.nc"
        key = f"{nc.stat().st_mtime_ns}-{n_draws}"
        f = cache / f"{run}.npz"
        if f.exists() and str(np.load(f)["key"]) == key:
            preds[run], truth = np.load(f)["pred"], np.load(f)["truth"]
            continue
        if sim is None:
            sim = jax.jit(jax.vmap(cmd.ir._build_simulator(ode, species, cfg, exps[run])))
            names, values = run_names, run_values
        post = az.from_netcdf(nc).posterior
        free = [p for p in post.data_vars if p in names]
        pooled = {p: np.asarray(post[p].values, float).ravel() for p in free}
        idx = np.linspace(0, len(pooled[free[0]]) - 1, n_draws).round().astype(int)
        base = np.array([float(values[n]) for n in names])
        th = np.tile(base, (n_draws + 1, 1))
        for p in free:
            th[:n_draws, names.index(p)] = pooled[p][idx]          # the last row stays at the truth
        out = []
        for i in range(0, len(th), 32):
            chunk = th[i:i + 32]
            pad = np.concatenate([chunk, np.tile(chunk[-1:], (32 - len(chunk), 1))])   # one compiled shape
            out.append(np.asarray(sim(jnp.asarray(pad))).reshape(32, -1)[:len(chunk)])
        out = np.concatenate(out)
        preds[run], truth = out[:n_draws], out[n_draws]
        np.savez(f, key=key, pred=preds[run], truth=truth)
    return preds, truth, exps


def fig4_fit():
    """Companion to fig4's noise column: each noise level's data against the fitted curve (posterior
    draws through the model, without measurement noise) and the noise-free truth."""
    runs = [r for _, r in NOISE]
    done, missing = _scores(runs)
    runs = [r for r in runs if r in done]
    if not runs:
        print("No noise-level run has finalized yet.")
        return None
    from forward_model import CONDITIONS
    preds, truth, exps = _fitted_predictions(runs)
    labels = {r: l for l, r in NOISE}
    rows = [("timeseries", "Time Series", "Time (min)", "C16 Equivalents (µM)"),
            ("profile", "Product Profile (12 min)", "Chain Length", "Concentration (µM)"),
            ("rates", "Initial Rates", "Condition", "Initial Rate (µM C16 Equivalents/min)")]
    _apply_plot_style()
    fig, axes = plt.subplots(3, len(runs), figsize=(5.6 * len(runs), 14.5), squeeze=False)
    obs_color, fit_color = "tab:orange", "tab:blue"
    for col, run in enumerate(runs):
        exp = exps[run]
        sigma_all = np.asarray(exp.observed_sigma, float).ravel()
        lo_all, med_all, hi_all = np.percentile(preds[run], [2.5, 50, 97.5], axis=0)
        offset = 0
        for ds in exp.datasets:
            n = int(np.size(ds.observed_values))
            sl = slice(offset, offset + n)
            offset += n
            kind = next(k for k, *_ in rows if k in ds.name)
            r = [k for k, *_ in rows].index(kind)
            ax = axes[r, col]
            obs = np.ravel(np.asarray(ds.observed_values, float))
            if kind == "timeseries":
                x = np.asarray(ds.time_values, float) / 60.0
                ax.fill_between(x, lo_all[sl], hi_all[sl], color=fit_color, alpha=0.25, linewidth=0)
                ax.plot(x, med_all[sl], color=fit_color, linewidth=2)
                ax.plot(x, truth[sl], color="0.2", linestyle="--", linewidth=1.5)
                ax.errorbar(x, obs, yerr=sigma_all[sl], fmt="o", color=obs_color, capsize=3, markersize=6)
            else:
                if kind == "profile":
                    ticks = [m[1].split("_")[0] for m in ds.observables_mapping]
                else:
                    ticks = [CONDITION_SHORT.get(nm, nm) for nm, _ in CONDITIONS][:n]
                x = np.arange(n, dtype=float)
                ax.bar(x, truth[sl], width=0.7, color=TRUTH_BAR, edgecolor="0.45", linewidth=1.2, zorder=1)
                ax.errorbar(x - 0.14, obs, yerr=sigma_all[sl], fmt="o", color=obs_color, capsize=3, markersize=6,
                            zorder=3)
                ax.errorbar(x + 0.14, med_all[sl], yerr=[med_all[sl] - lo_all[sl], hi_all[sl] - med_all[sl]],
                            fmt="s", color=fit_color, capsize=3, markersize=6, linewidth=2, zorder=3)
                ax.set_xticks(x, ticks)
                ax.set_xlim(-0.6, n - 0.4)
            if col == 0:
                ax.set_ylabel(rows[r][3])
            ax.set_xlabel(rows[r][2])
        err = np.max(np.abs(med_all - truth) / np.abs(truth)) * 100
        half = np.median((hi_all - lo_all) / 2 / np.abs(truth)) * 100
        axes[0, col].set_title(f"Noise {labels[run]}")
        axes[0, col].text(0.97, 0.05, f"Fit vs truth: within {err:.1f}%\n95% band: ±{half:.1f}% (median)",
                          transform=axes[0, col].transAxes, ha="right", va="bottom", fontsize=PLOT_FONT_SIZE - 3)
    for r in range(3):
        _share_row_ylim(axes[r])
        for ax in axes[r]:
            ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=obs_color, marker="o", linestyle="none", markersize=7),
               (plt.Rectangle((0, 0), 1, 1, color=fit_color, alpha=0.25), Line2D([], [], color=fit_color, linewidth=2)),
               (Line2D([], [], color="0.2", linestyle="--", linewidth=1.5),
                plt.Rectangle((0, 0), 1, 1, facecolor=TRUTH_BAR, edgecolor="0.45"))]
    names = ["Observed (± 1σ Noise)", "Posterior Fit (Median, 95% Interval)", "Truth (Noise-Free; Line, Bars)"]
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    legend = fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C8 - a1c3 — Predictive Fit Across Noise Levels")
    return _save(fig, "robustness_predictive_fit.png")


def fig4_prior():
    """Companion to fig4's prior-shift column: each shifted prior against its posterior (default
    start). Left, the full range with each curve scaled to a peak of 1, so a prior 4 sd away and a
    posterior 20x narrower show on one axis; right, the posteriors' own densities near the truth."""
    import arviz as az
    from scipy.stats import gaussian_kde
    runs = [r for _, r in SHIFT]
    done, missing = _scores(runs)
    shifts = [(k, r) for k, r in SHIFT if r in done]
    if not shifts:
        print("No prior-shift run has finalized yet.")
        return None
    params = list(done[shifts[0][1]]["params"])
    ramps = {"a1": plt.cm.Blues, "c3": plt.cm.Greens, "a2": plt.cm.Purples}
    _apply_plot_style()
    fig, axes = plt.subplots(len(params), 2, figsize=(15.0, 5.2 * len(params)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 2]})
    zoom = {}
    for row, p in enumerate(params):
        ramp = ramps.get(p, plt.cm.Greys)
        for i, (k, run) in enumerate(shifts):
            color = ramp(0.4 + 0.6 * i / max(len(shifts) - 1, 1))
            cfg = json.loads((RESULTS / run / "solver_params.json").read_text())
            pr = next(q["prior_dist_params"] for q in cfg["free_kinetic_params"] if q["param_name"] == p)
            lo, hi = float(pr["lower"]), float(pr["upper"])
            mu, sig = 0.5 * math.log(lo * hi), math.log(hi / lo) / (2 * 1.959963984540054)
            lx = np.linspace(mu - 4 * sig, mu + 4 * sig, 400)
            axes[row, 0].plot(np.exp(lx), np.exp(-0.5 * ((lx - mu) / sig) ** 2), color=color, linestyle="--",
                              linewidth=1.8)
            draws = np.log(np.asarray(az.from_netcdf(RESULTS / run / "posterior_samples_pm.nc").posterior[p].values,
                                      float).ravel())
            kde = gaussian_kde(draws)
            gx = np.linspace(draws.min() - 0.1, draws.max() + 0.1, 400)
            gd = kde(gx)
            axes[row, 0].fill_between(np.exp(gx), gd / gd.max(), color=color, alpha=0.5, linewidth=0)
            axes[row, 0].plot(np.exp(gx), gd / gd.max(), color=color, linewidth=1.5)
            # density per unit of the parameter itself: the log-scale density divided by x
            axes[row, 1].plot(np.exp(gx), gd / np.exp(gx), color=color, linewidth=2.2,
                              label=f"+{k} sd (prior {math.exp(mu):.3g})" if k else "0 (prior 1)")
            zoom.setdefault(row, []).append((np.exp(draws.min()), np.exp(draws.max())))
        axes[row, 0].axvline(1.0, **TRUTH_STYLE)
        axes[row, 1].axvline(1.0, **TRUTH_STYLE)
        axes[row, 0].set_xscale("log")
        ticks = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        axes[row, 0].set_xticks(ticks, [f"{t:,g}" if t >= 1 else f"{t:g}" for t in ticks])
        axes[row, 0].set_ylim(0, 1.08)
        axes[row, 0].set_ylabel(f"{p}: Density (Scaled to Peak 1)")
        axes[row, 1].set_ylabel("Posterior Density")
        axes[row, 1].set_ylim(bottom=0)
        lo_z = min(a for a, _ in zoom[row]); hi_z = max(b for _, b in zoom[row])
        axes[row, 1].set_xlim(lo_z, hi_z)
    axes[0, 0].set_title("Priors (Dashed) and Posteriors (Filled)")
    axes[0, 1].set_title("Posteriors Near the Truth")
    axes[-1, 0].set_xlabel("Parameter Value")
    axes[-1, 1].set_xlabel("Parameter Value")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="0.3", linestyle="--", linewidth=1.8), plt.Rectangle((0, 0), 1, 1, color="0.3", alpha=0.5),
               Line2D([], [], **TRUTH_STYLE)]
    names = ["Prior", "Posterior", "Truth"]
    shade = [Line2D([], [], color=plt.cm.Greys(0.4 + 0.6 * i / max(len(shifts) - 1, 1)), linewidth=5) for i in range(len(shifts))]
    shade_names = [f"+{k} sd" if k else "No Shift" for k, _ in shifts]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    legend = fig.legend(handles + shade, names + shade_names, loc="upper center", bbox_to_anchor=(0.5, 0.06),
                        ncol=len(names) + len(shade_names), title="Prior Median Shift: Darker = Further (Blue a1, Green c3)")
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C8 - a1c3 — Prior and Posterior Under Prior Shift")
    return _save(fig, "robustness_prior_posterior.png")


R6_CELLS = {("grouped", "standard"): "Tier1 C14+unsat - a1c3",
            ("split", "standard"): "Tier1 C14+unsat - a1c3sc3l",
            ("grouped", "1:3"): "Tier1 C14+unsat+c3split_c3l3 - a1c3",
            ("split", "1:3"): "Tier1 C14+unsat+c3split_c3l3 - a1c3sc3l"}
MODEL_COLOR = {"grouped": "tab:green", "split": "tab:purple"}


def fig5():
    """The c3 grouping test (R1 + R6): grouped c3 against the split c3s / c3l, on standard data
    (true c3s = c3l = 1) and on off-grouping data (true c3l = 3 x c3s). A: the c3 posteriors
    against the truths. B: the split model's c3l / c3s ratio. C: each model's residuals on the
    1:3 data, in noise sd (observed minus the posterior-predictive mean). D: elpd_loo difference,
    grouped minus split, with its SE (az.compare)."""
    import arviz as az
    import check_model_vs_data as cmd
    runs = list(R6_CELLS.values())
    done, missing = _scores(runs)
    if len(done) < len(runs):
        print("Fig 5 needs all four grouping cells; missing: " + ", ".join(missing))
        return None
    idata = {k: az.from_netcdf(RESULTS / r / "posterior_samples_pm.nc") for k, r in R6_CELLS.items()}
    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 12.0))
    # A: posteriors of the c3 groups, median and 95% interval, against the truths
    ax = axes[0, 0]
    truth = {"standard": {"c3": 1.0, "c3s": 1.0, "c3l": 1.0}, "1:3": {"c3s": 1.0, "c3l": 3.0}}
    series = [("grouped", "c3", "o", "Grouped c3"), ("split", "c3s", "s", "Split c3s (short chains)"),
              ("split", "c3l", "^", "Split c3l (long chains)")]
    for j, data in enumerate(["standard", "1:3"]):
        for i, (model, p, marker, label) in enumerate(series):
            x = j + (i - 1) * 0.22
            s = done[R6_CELLS[(model, data)]]["params"][p]
            ax.errorbar([x], [s["median"]], yerr=[[s["median"] - s["ci95"][0]], [s["ci95"][1] - s["median"]]],
                        fmt=marker, color=MODEL_COLOR[model], markersize=10, capsize=5, linewidth=2,
                        markerfacecolor="white" if p == "c3l" else MODEL_COLOR[model], label=label if j == 0 else None)
            if p in truth[data]:
                ax.plot([x - 0.09, x + 0.09], [truth[data][p]] * 2, color="0.2", linewidth=2.5,
                        label="Truth" if (j, p) == (0, "c3") else None)
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Standard Data\n(c3s = c3l)", "Off-Grouping Data\n(c3l = 3 × c3s)"])
    ax.set_xlim(-0.6, 1.6)
    from inference_plotting import _plain_ticks_if_narrow_log
    _plain_ticks_if_narrow_log(ax, "y")
    ax.set_ylabel("Posterior Median and 95% Interval")
    ax.set_title("A  c3 Posteriors")
    ax.legend(loc="upper left", fontsize=PLOT_FONT_SIZE - 4)
    # B: the split model's c3l / c3s ratio
    ax = axes[0, 1]
    for data, color, true_ratio in (("standard", "0.55", 1.0), ("1:3", MODEL_COLOR["split"], 3.0)):
        post = idata[("split", data)].posterior
        ratio = (np.asarray(post["c3l"].values, float) / np.asarray(post["c3s"].values, float)).ravel()
        gx, gd = _kde_curve(ratio, True)
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        ax.fill_between(gx, gd, color=color, alpha=0.3, linewidth=0)
        ax.plot(gx, gd, color=color, linewidth=2,
                label=f"{'Standard' if data == 'standard' else 'Off-grouping'} data: {np.median(ratio):.2f} "
                      f"[{lo:.2f}, {hi:.2f}], P(> 1) = {np.mean(ratio > 1):.2f}")
        ax.axvline(true_ratio, color=color, linestyle="--", linewidth=1.5)
    ax.set_xscale("log")
    ticks = [0.7, 1, 1.5, 2, 3, 5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("c3l / c3s (Split Model; Dashed = Truth)")
    ax.set_ylabel("Density")
    ax.set_title("B  Does the Split Model See the Difference?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=PLOT_FONT_SIZE - 4)
    # C: residuals on the 1:3 data, in noise sd
    ax = axes[1, 0]
    run = R6_CELLS[("grouped", "1:3")]
    cfg_path = RESULTS / run / "solver_params.json"
    cfg = json.loads(cfg_path.read_text())
    imported = cmd.ir.import_solver_params(cfg_path)
    _, species, _, _, _ = cmd.build_ode_system_from_reactions(imported.reactions_source, scaling_group=cfg["scaling_groups"])
    exp = cmd.load_experiment_bundle(solver_params=cfg, solver_params_file=str(cfg_path), species_names=species)
    sigma = np.asarray(exp.observed_sigma, float).ravel()
    edges, names, offset = [], [], 0
    for ds in exp.datasets:
        n = int(np.size(ds.observed_values))
        edges.append((offset, offset + n))
        names.append(next(t for k, t in (("timeseries", "Time Series"), ("profile", "Profile"), ("rates", "Rates"))
                          if k in ds.name))
        offset += n
    for i, model in enumerate(["grouped", "split"]):
        d = idata[(model, "1:3")]
        obs = np.asarray(d["observed_data"]["llike"], float).ravel()
        pred = np.asarray(d["posterior_predictive"]["llike"], float).reshape(-1, obs.size).mean(axis=0)
        r = (obs - pred) / sigma
        x = np.arange(obs.size) + (i - 0.5) * 0.3
        ax.plot(x, r, "o" if model == "grouped" else "s", color=MODEL_COLOR[model], markersize=8,
                label=f"{model.capitalize()} model: RMS {np.sqrt(np.mean(r ** 2)):.2f}, max |r| {np.max(np.abs(r)):.1f}")
    for lo, hi in edges[1:]:
        ax.axvline(lo - 0.5, color="0.7", linewidth=1)
    for (lo, hi), nm in zip(edges, names):
        ax.text((lo + hi - 1) / 2, 1.02, nm, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=PLOT_FONT_SIZE - 3)
    ax.axhline(0, color="0.6", linewidth=1)
    handle = _threshold_lines(ax, [-2, 2])
    ax.set_xticks([])
    ax.set_xlabel("Observation (Off-Grouping Data)")
    ax.set_ylabel("Residual (Noise Sd)")
    ax.set_title("C  Misfit on the Off-Grouping Data", pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=PLOT_FONT_SIZE - 4)
    # D: elpd_loo difference, grouped minus split
    ax = axes[1, 1]
    for j, data in enumerate(["standard", "1:3"]):
        # pointwise, not az.compare: ArviZ 1.x rounds compare's table to two significant figures
        loo = {m: az.loo(idata[(m, data)], pointwise=True) for m in ("grouped", "split")}
        d_i = np.asarray(loo["grouped"].elpd_i, float).ravel() - np.asarray(loo["split"].elpd_i, float).ravel()
        diff, dse = float(d_i.sum()), float(np.sqrt(d_i.size * d_i.var()))
        kmax = max(float(np.max(np.asarray(l.pareto_k))) for l in loo.values())
        ax.errorbar([j], [diff], yerr=[[2 * dse], [2 * dse]], fmt="o", color="0.25", markersize=10, capsize=6,
                    linewidth=2)
        ax.text(j + 0.08, diff, f"{diff:+.1f} ± {dse:.1f}" + ("\n(max Pareto k %.2f)" % kmax if kmax > 0.7 else ""),
                va="center", fontsize=PLOT_FONT_SIZE - 3)
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xticks([0, 1], ["Standard Data", "Off-Grouping Data"])
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylabel("Δelpd_loo, Grouped − Split (± 2 SE)")
    ax.set_title("D  Which Model Predicts Better?")
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=4.0)
    legend = fig.legend([handle], ["Threshold"], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C14+unsat — c3 Grouping Test")
    return _save(fig, "grouping_test.png")


def fig4_si():
    """SI: the prior-shift runs from the default start (the shifted prior's mean) against the same
    shifts started at the ME1 values. Identical answers mean the start does not matter."""
    runs = [r for _, r in SHIFT] + [r for _, r in SHIFT_INIT1]
    done, missing = _scores(runs)
    if not done:
        print("No prior-shift run has finalized yet.")
        return None
    params = list(next(iter(done.values()))["params"])
    shift_labels = _shift_labels()
    _apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 10.0))
    _interval_panel(axes[0], shift_labels, [r for _, r in SHIFT], done, params, shift=-0.2, suffix=" (Default Start)")
    _interval_panel(axes[0], shift_labels, [r for _, r in SHIFT_INIT1], done, params, shift=0.2, open_markers=True,
                    suffix=" (Started at ME1 Values)")
    axes[0].axhline(1.0, **TRUTH_STYLE, label="Truth")
    _log_interval_axis(axes[0])
    _z_panel(axes[1], shift_labels, [r for _, r in SHIFT], done, params, suffix=" (Default Start)")
    _z_panel(axes[1], shift_labels, [r for _, r in SHIFT_INIT1], done, params, open_markers=True,
             suffix=" (Started at ME1 Values)")
    zmax = max([2.5] + [abs(v) + 0.3 for line in axes[1].get_lines() for v in line.get_ydata()])
    axes[1].set_ylim(-zmax, zmax)
    axes[1].axhline(0, color="0.6", linewidth=1)
    handle = _threshold_lines(axes[1], [-2, 2])
    axes[0].set_ylabel("Posterior Median and 95% Interval")
    axes[1].set_xlabel("Prior Median Shift (Prior Sd)")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    handles, labels = zip(*[(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))])
    handles, labels = list(handles) + [handle], list(labels) + ["Threshold"]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C8 - a1c3 — Prior Shift: Default vs ME1 Start (SI)")
    return _save(fig, "robustness_start_comparison.png")


def r7():
    """Sampled log-scale contraction against the expected-information grid's prediction."""
    grid = json.loads(R7_GRID.read_text())["tier1_design_parts"]
    runs = [r for _, r, _ in R7_CELLS]
    done, missing = _scores(runs)
    params = ["a1", "c3"]
    _apply_plot_style()
    fig, axes = plt.subplots(1, len(params), figsize=(14.0, 5.2), sharey=True)
    width = 0.36
    for ax, p in zip(axes, params):
        x = np.arange(len(R7_CELLS))
        pred = [grid[part]["contraction"][p] for _, _, part in R7_CELLS]
        samp = [done.get(run, {}).get("params", {}).get(p, {}).get("contraction", np.nan) for _, run, _ in R7_CELLS]
        color = PARAM_COLOR.get(p)
        ax.bar(x - width / 2, pred, width=width, facecolor="white", edgecolor=color, hatch="///", linewidth=1.5,
               label="Predicted (Expected Information)")
        ax.bar(x + width / 2, samp, width=width, color=color, alpha=0.7, label="Sampled")
        for xi, v in zip(x, pred):
            ax.text(xi - width / 2, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=PLOT_FONT_SIZE - 4)
        for xi, v in zip(x, samp):
            if np.isfinite(v):
                ax.text(xi + width / 2, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=PLOT_FONT_SIZE - 4)
        ax.set_xticks(x, [label for label, _, _ in R7_CELLS])
        ax.set_title(p)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Posterior Contraction (Log Scale)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    from matplotlib.patches import Patch
    legend = fig.legend(handles=[Patch(facecolor="white", edgecolor="0.3", hatch="///", label="Predicted (Expected Information)"),
                                 Patch(color="0.45", alpha=0.7, label="Sampled")],
                        loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)
    _footnote(fig, missing, legend)
    place_suptitle(fig, "Tier1 C14+unsat - a1c3 — Data-Type Check: Sampled vs Predicted Contraction")
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
    ap.add_argument("figure", choices=["fig2", "fig4", "fig4_fit", "fig4_prior", "fig4_si", "fig5", "r7", "r8", "all"])
    ap.add_argument("run", nargs="?", default="Tier1 C14+unsat - a1c3", help="fig2's run (default R1)")
    a = ap.parse_args()
    if a.figure in ("fig2", "all"):
        fig2(a.run)
    if a.figure in ("fig4", "all"):
        fig4()
    if a.figure in ("fig4_fit", "all"):
        fig4_fit()
    if a.figure in ("fig4_prior", "all"):
        fig4_prior()
    if a.figure in ("fig4_si", "all"):
        fig4_si()
    if a.figure in ("fig5", "all"):
        fig5()
    if a.figure in ("r7", "all"):
        r7()
    if a.figure in ("r8", "all"):
        r8()


if __name__ == "__main__":
    main()
