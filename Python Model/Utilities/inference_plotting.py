from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
from itertools import combinations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


PLOT_FONT_SIZE = 16
TABLE_FONT_SIZE = 11
PLOT_STYLE_RC = {
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "legend.title_fontsize": PLOT_FONT_SIZE,
    "figure.titlesize": PLOT_FONT_SIZE,
    "xtick.major.size": 10.0,
    "ytick.major.size": 10.0,
    "xtick.minor.size": 6.0,
    "ytick.minor.size": 6.0,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.top": False,
    "ytick.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
sns.set_theme(style="whitegrid", rc=PLOT_STYLE_RC)
plt.rcParams.update(PLOT_STYLE_RC)


def _apply_plot_style() -> None:
    """Re-apply plotting style in case caller code reset rcParams."""
    sns.set_theme(style="whitegrid", rc=PLOT_STYLE_RC)
    plt.rcParams.update(PLOT_STYLE_RC)


def _select_free_params(inf_data: az.InferenceData, free_params: list[str]) -> list[str]:
    available_vars = set(inf_data.posterior.data_vars)
    selected = [name for name in free_params if name in available_vars]
    if not selected:
        raise ValueError("None of the provided free_params are present in posterior variables.")
    return selected


def _sanitize_name(name: str, fallback: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name).strip(
        "_"
    )
    return safe or fallback


def _resolve_paths(save_path: str | None) -> dict[str, Path | None]:
    if not save_path:
        return {
            "trace": None,
            "posterior_marginals": None,
            "prior_marginals": None,
            "save_dir": None,
            "stem": None,
        }

    trace_path = Path(save_path).expanduser().resolve()
    return {
        "trace": trace_path,
        "posterior_marginals": trace_path.with_name(f"{trace_path.stem}_hist.pdf"),
        "prior_marginals": trace_path.with_name(f"{trace_path.stem}_prior_hist.pdf"),
        "save_dir": trace_path.parent,
        "stem": trace_path.stem,
    }


def _configure_decade_log_x_axis(ax: Any, values: np.ndarray) -> None:
    """Configure x-axis to strict decade bounds with labels only at powers of 10."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    finite_values = finite_values[finite_values > 0.0]
    if finite_values.size == 0:
        return

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))

    min_exp = int(np.floor(np.log10(min_value)))
    max_exp = int(np.ceil(np.log10(max_value)))

    # Enforce strict bounds: lower bound below min, upper bound above max.
    if np.isclose(min_value, 10.0**min_exp):
        min_exp -= 1
    if np.isclose(max_value, 10.0**max_exp):
        max_exp += 1

    lower = 10.0**min_exp
    upper = 10.0**max_exp

    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _extract_predictive_data(inf_data: az.InferenceData) -> tuple[np.ndarray | None, str | None]:
    posterior_predictive_group = getattr(inf_data, "posterior_predictive", None)
    if posterior_predictive_group is None:
        return None, None

    var_names = list(posterior_predictive_group.data_vars)
    if not var_names:
        return None, None

    preferred = [
        name for name in var_names if any(token in name.lower() for token in ("like", "pred", "obs"))
    ]
    predictive_var_name = preferred[0] if preferred else var_names[0]
    values = posterior_predictive_group[predictive_var_name].values
    return values, predictive_var_name


def _build_table_rows(dataset: Any, experiment: Any, n_points: int) -> list[tuple[str, list[str]]]:
    table_rows: list[tuple[str, list[str]]] = []
    if getattr(dataset, "init_cond_columns", None):
        for species, column in dataset.init_cond_columns:
            table_rows.append(
                (
                    species,
                    [f"{float(dataset.frame.iloc[row_index][column]):.3g}" for row_index in range(n_points)],
                )
            )
        return table_rows

    global_init = experiment.raw_config.get("init_conds", {}) if experiment is not None else {}
    species_candidates = list(global_init.keys())

    for species in species_candidates:
        if species in dataset.frame.columns:
            table_rows.append(
                (
                    species,
                    [f"{float(dataset.frame.iloc[row_index][species]):.3g}" for row_index in range(n_points)],
                )
            )

    if not table_rows and global_init:
        for species, value in global_init.items():
            table_rows.append((species, [f"{float(value):.3g}" for _ in range(n_points)]))

    return table_rows


def _measure_text_width_in(text: str, fontsize: float) -> float:
    """Real glyph-metric text width in inches at the given point size, via
    matplotlib's font machinery -- no renderer/draw pass needed."""
    from matplotlib.textpath import TextPath

    return TextPath((0, 0), str(text), size=fontsize).get_extents().width / 72.0


def _measure_table_column_widths_in(
    table_row_labels: list[str],
    table_cell_text: list[list[str]],
    fontsize: float,
    pad_in: float = 0.16,
) -> tuple[float, list[float]]:
    """Species-name column width and per-data-column widths (inches), each
    sized to its widest rendered string plus a small padding buffer."""
    species_col_width_in = pad_in + max(
        (_measure_text_width_in(label, fontsize) for label in table_row_labels), default=0.0
    )
    n_points = len(table_cell_text[0]) if table_cell_text else 0
    data_col_widths_in = [
        pad_in + max((_measure_text_width_in(row[col_index], fontsize) for row in table_cell_text), default=0.0)
        for col_index in range(n_points)
    ]
    return species_col_width_in, data_col_widths_in


def _aggregate_groupwise_series(
    x_values: np.ndarray,
    observed_values: np.ndarray,
    observed_sigma: np.ndarray,
    posterior_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate repeated x points to group means for groupwise-noise plotting."""
    x_values = np.asarray(x_values, dtype=float)
    observed_values = np.asarray(observed_values, dtype=float)
    observed_sigma = np.asarray(observed_sigma, dtype=float)
    posterior_samples = np.asarray(posterior_samples, dtype=float)

    if x_values.ndim != 1:
        raise ValueError("Expected 1D x_values for groupwise aggregation.")
    if observed_values.shape != x_values.shape or observed_sigma.shape != x_values.shape:
        raise ValueError("Observed values/sigma must match x_values shape for groupwise aggregation.")
    if posterior_samples.ndim != 2 or posterior_samples.shape[1] != x_values.shape[0]:
        raise ValueError("Posterior samples must have shape (n_samples, n_points) for groupwise aggregation.")

    unique_x, first_idx = np.unique(x_values, return_index=True)
    order = np.argsort(first_idx)
    unique_x = unique_x[order]

    if unique_x.size == x_values.size:
        return x_values, observed_values, observed_sigma, posterior_samples

    grouped_obs: list[float] = []
    grouped_sigma: list[float] = []
    grouped_pred: list[np.ndarray] = []

    for x_value in unique_x:
        mask = x_values == x_value
        grouped_obs.append(float(np.nanmean(observed_values[mask])))
        grouped_sigma.append(float(np.nanmean(observed_sigma[mask])))
        grouped_pred.append(np.nanmean(posterior_samples[:, mask], axis=1))

    return (
        unique_x,
        np.asarray(grouped_obs, dtype=float),
        np.asarray(grouped_sigma, dtype=float),
        np.column_stack(grouped_pred),
    )


def _extract_trace_series(
    inf_data: az.InferenceData,
    param_name: str,
    include_tuning: bool,
) -> tuple[np.ndarray, int]:
    """Return trace series with optional warmup prepended and posterior start index."""
    posterior_values = np.asarray(inf_data.posterior[param_name].values, dtype=float)

    if posterior_values.ndim < 2:
        posterior_series = posterior_values.reshape(1, -1)
    else:
        chains = posterior_values.shape[0]
        draws = posterior_values.shape[1]
        posterior_series = posterior_values.reshape(chains, draws, -1)[:, :, 0]

    if not include_tuning:
        return posterior_series, 0

    warmup_group = getattr(inf_data, "warmup_posterior", None)
    if warmup_group is None or param_name not in set(warmup_group.data_vars):
        return posterior_series, 0

    warmup_values = np.asarray(warmup_group[param_name].values, dtype=float)
    if warmup_values.ndim < 2:
        warmup_series = warmup_values.reshape(1, -1)
    else:
        warmup_chains = warmup_values.shape[0]
        warmup_draws = warmup_values.shape[1]
        warmup_series = warmup_values.reshape(warmup_chains, warmup_draws, -1)[:, :, 0]

    n_chains = min(warmup_series.shape[0], posterior_series.shape[0])
    if n_chains == 0:
        return posterior_series, 0

    warmup_series = warmup_series[:n_chains]
    posterior_series = posterior_series[:n_chains]
    return np.concatenate([warmup_series, posterior_series], axis=1), warmup_series.shape[1]


def _kde_curve(values: np.ndarray, log_x: bool, n_grid: int = 512, pad_frac: float = 0.04):
    """KDE evaluated on a grid, fit in log10 space when the axis is logarithmic.

    Fitting in linear space and then displaying on a log axis is what made the
    prior invisible: a LogNormal's linear-space spread is dominated by its long
    right tail, so the KDE bandwidth is huge and the curve is smeared flat
    across the decades where the mass actually lives.
    """
    from scipy.stats import gaussian_kde

    v = np.log10(values) if log_x else values
    kde = gaussian_kde(v)
    lo, hi = float(np.min(v)), float(np.max(v))
    pad = pad_frac * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    density = kde(grid)
    return (np.power(10.0, grid) if log_x else grid), density


def plot_posterior_trace_diagnostics(
    inf_data: az.InferenceData,
    free_params: list[str],
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
    include_tuning: bool = False,
    posterior_start_line: bool = True,
    criteria_met_at: int | None = None,
    system_name: str | None = None,
) -> dict[str, Any]:
    _apply_plot_style()
    selected_free_params = _select_free_params(inf_data, free_params)

    n_params = len(selected_free_params)
    # 3 rows: prior, posterior, trace; one (wide) column per param.
    fig, axes = plt.subplots(3, n_params, figsize=(10.0 * n_params, 12.0), squeeze=False)

    prior_group = getattr(inf_data, "prior", None)

    for col_index, param_name in enumerate(selected_free_params):
        posterior_values = np.asarray(inf_data.posterior[param_name].values, dtype=float)
        posterior_flat = posterior_values[np.isfinite(posterior_values)]

        prior_ax = axes[0, col_index]
        if prior_group is not None and param_name in set(prior_group.data_vars):
            prior_values = np.asarray(prior_group[param_name].values, dtype=float)
            prior_flat = prior_values[np.isfinite(prior_values)]
        else:
            prior_flat = np.array([])

        # Panel 1: the prior over its own full range, with the posterior overlaid.
        # Both curves are scaled to unit peak height. An informative posterior is
        # ~40x taller than a diffuse prior here (measured on C8), so plotting true
        # densities on a shared axis renders the prior as an invisible flat line --
        # which is exactly the bug this replaces. Width, not height, carries the
        # contraction story; the true density is in panel 2.
        prior_log_x = (
            use_log_param_axis
            and prior_flat.size > 0
            and np.all(prior_flat > 0.0)
            and posterior_flat.size > 0
            and np.all(posterior_flat > 0.0)
        )
        drew_prior = False
        if prior_flat.size > 1 and np.nanstd(prior_flat) > 0:
            px, pd_ = _kde_curve(prior_flat, prior_log_x)
            pd_ = pd_ / np.max(pd_)
            prior_ax.fill_between(px, pd_, color="tab:gray", alpha=0.45)
            prior_ax.plot(px, pd_, color="tab:gray", linewidth=1.5, label="Prior")
            drew_prior = True
        elif prior_flat.size == 1:
            prior_ax.axvline(float(prior_flat[0]), linewidth=2.0, color="tab:gray", label="Prior")
            drew_prior = True
        else:
            prior_ax.text(0.5, 0.5, "No prior samples", ha="center", va="center")

        if posterior_flat.size > 1 and np.nanstd(posterior_flat) > 0:
            qx, qd = _kde_curve(posterior_flat, prior_log_x)
            qd = qd / np.max(qd)
            prior_ax.fill_between(qx, qd, color="tab:blue", alpha=0.55)
            prior_ax.plot(qx, qd, color="tab:blue", linewidth=1.5, label="Posterior")
        elif posterior_flat.size == 1:
            prior_ax.axvline(float(posterior_flat[0]), linewidth=2.0, label="Posterior")

        if prior_log_x:
            prior_ax.set_xscale("log")
        # Auto x-limits spanning both curves, so the prior's full width is visible.
        prior_ax.relim()
        prior_ax.autoscale(enable=True, axis="x")
        prior_ax.set_ylim(bottom=0.0)

        prior_ax.set_title(f"{param_name} Prior vs. Posterior")
        prior_ax.set_xlabel("Parameter Value")
        prior_ax.set_ylabel("Relative density (each scaled to peak 1)")
        if drew_prior or posterior_flat.size:
            prior_ax.legend(loc="best")

        # Panel 2: posterior alone, true density, auto-scaled -- necessarily far more
        # zoomed than panel 1 because the posterior is the narrower of the two.
        density_ax = axes[1, col_index]
        post_log_x = use_log_param_axis and posterior_flat.size > 0 and np.all(posterior_flat > 0.0)
        if posterior_flat.size > 1 and np.nanstd(posterior_flat) > 0:
            qx2, qd2 = _kde_curve(posterior_flat, post_log_x)
            density_ax.fill_between(qx2, qd2, color="tab:blue", alpha=0.55)
            density_ax.plot(qx2, qd2, color="tab:blue", linewidth=1.5, label="Posterior density")
            if post_log_x:
                density_ax.set_xscale("log")
            density_ax.set_ylim(bottom=0.0)
        elif posterior_flat.size == 1:
            density_ax.axvline(float(posterior_flat[0]), linewidth=2.0, label="Posterior value")
        else:
            density_ax.text(0.5, 0.5, "No finite samples", ha="center", va="center")

        density_ax.set_title(f"{param_name} Posterior (zoomed)")
        density_ax.set_xlabel("Parameter Value")
        density_ax.set_ylabel("Density")

        trace_ax = axes[2, col_index]
        trace_series, posterior_start_idx = _extract_trace_series(
            inf_data=inf_data,
            param_name=param_name,
            include_tuning=include_tuning,
        )

        x_draws = np.arange(trace_series.shape[1], dtype=int)
        chain_lines = []
        for chain_index in range(trace_series.shape[0]):
            line, = trace_ax.plot(x_draws, trace_series[chain_index], linewidth=1.0, label=str(chain_index))
            chain_lines.append(line)

        marker_lines = []
        if include_tuning and posterior_start_line and posterior_start_idx > 0:
            start_line = trace_ax.axvline(
                posterior_start_idx - 0.5,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label="Posterior start (end of warmup)",
            )
            marker_lines.append(start_line)

        if criteria_met_at is not None:
            # criteria_met_at is an absolute position on whatever timeline
            # this panel is showing (full warmup++sampling when
            # include_tuning, sampling-only otherwise) -- callers compute it
            # to match, so no offset is added here.
            criteria_line = trace_ax.axvline(
                criteria_met_at,
                color="firebrick",
                linestyle=":",
                linewidth=1.8,
                label=f"Criteria met (draw {criteria_met_at})",
            )
            marker_lines.append(criteria_line)

        if marker_lines:
            trace_ax.legend(handles=marker_lines, loc="lower right", fontsize="small")
            trace_ax.add_artist(trace_ax.get_legend())

        # Warmup can span orders of magnitude more than the settled posterior
        # (chains starting from very different initial guesses); a linear
        # y-axis would crush the post-convergence mixing into a flat line.
        finite_trace = trace_series[np.isfinite(trace_series)]
        if include_tuning and finite_trace.size > 0 and np.all(finite_trace > 0.0):
            trace_ax.set_yscale("log")

        trace_ax.set_title(f"{param_name} Trace")
        trace_ax.set_xlabel("Draw")
        trace_ax.set_ylabel("Parameter Value")
        trace_ax.legend(
            handles=chain_lines,
            title="Chains",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=min(len(chain_lines), 12),
            handlelength=1.0,
            columnspacing=0.8,
            handletextpad=0.4,
        )

    fig.suptitle(f"{system_name} — Posterior Trace Diagnostics" if system_name else "Posterior Trace Diagnostics")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.98))

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)

    if not show:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": axes,
        "selected_free_params": selected_free_params,
        "plot_file": plot_file,
    }


def _rhat_ess_trajectory(
    inf_data: az.InferenceData,
    free_params: list[str],
    step: int,
    ess_method: str = "bulk",
    include_tuning: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Worst-case rank-normalized r-hat / min bulk-ESS over growing prefixes of
    the timeline, at every ``step``-th draw count. Mirrors
    ResumableSampler._check_converged's logic exactly (worst r-hat, min ESS
    across free params) when include_tuning=False (sampling-phase only, the
    live early-stop check's own behavior) -- just evaluated on a finer grid
    for a smooth curve instead of the sparse rhat_check_every points actually
    checked live.

    include_tuning=True instead prepends the warmup draws (when the netcdf
    has a warmup_posterior group), matching finalize_window.py's "the whole
    warmup++sampling timeline is one valid chain" philosophy: rank-normalized
    r-hat/ESS assume a single quasi-stationary distribution, which the
    non-stationary tuning process actively violates, so expect (and don't
    read too much into) large r-hat / tiny ESS during early warmup -- that's
    the adaptation process itself, not a mixing failure. Returns n_tune (the
    warmup length actually used, 0 if include_tuning was False or unavailable)
    so callers can mark the phase boundary.
    """
    warmup_group = getattr(inf_data, "warmup_posterior", None) if include_tuning else None
    arrays = {}
    n_tune = 0
    for name in free_params:
        posterior_arr = np.asarray(inf_data.posterior[name].values, dtype=float)
        if warmup_group is not None and name in set(warmup_group.data_vars):
            warmup_arr = np.asarray(warmup_group[name].values, dtype=float)
            n_chains = min(warmup_arr.shape[0], posterior_arr.shape[0])
            arrays[name] = np.concatenate([warmup_arr[:n_chains], posterior_arr[:n_chains]], axis=1)
            n_tune = warmup_arr.shape[1]
        else:
            arrays[name] = posterior_arr
    n_draws = next(iter(arrays.values())).shape[1]

    draw_counts = np.arange(step, n_draws + 1, step, dtype=int)
    if draw_counts.size == 0 or draw_counts[-1] != n_draws:
        draw_counts = np.append(draw_counts, n_draws)

    rhats = np.full(draw_counts.shape, np.nan)
    esses = np.full(draw_counts.shape, np.nan)
    for i, n in enumerate(draw_counts):
        if n < 2:
            continue
        worst_rhat, worst_ess = 0.0, float("inf")
        for name, arr in arrays.items():
            windowed = arr[:, :n]
            worst_rhat = max(worst_rhat, float(az.rhat({name: windowed})[name].values))
            worst_ess = min(worst_ess, float(az.ess({name: windowed}, method=ess_method)[name].values))
        rhats[i] = worst_rhat
        esses[i] = worst_ess
    return draw_counts, rhats, esses, n_tune


def compute_convergence_criteria_met_at(
    inf_data: az.InferenceData,
    free_params: list[str],
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    step: int = 10,
    include_tuning: bool = False,
    consecutive: int = 1,
) -> int | None:
    """Cumulative SAMPLING draw count where r_hat < rhat_threshold and
    ess >= ess_threshold jointly hold (worst-case across free_params) for
    ``consecutive`` grid points in a row, or None if never reached.

    Warmup is excluded by default, which is what Stan/PyMC/ArviZ all do:
    during warmup the step size and mass matrix are still adapting, so those
    draws are not samples from any fixed distribution. Including them also
    makes the answer depend on how much warmup was run (a longer warmup
    dilutes the initial transient, so the same chain appears to "converge"
    sooner), which makes the number incomparable across runs.

    ``consecutive`` > 1 requires a sustained crossing rather than a single
    touch -- mirroring ResumableSampler's convergence_consecutive_checks.
    """
    selected_free_params = _select_free_params(inf_data, free_params)
    draw_counts, rhats, esses, _n_tune = _rhat_ess_trajectory(
        inf_data, selected_free_params, step=step, include_tuning=include_tuning
    )
    ok = (rhats < rhat_threshold) & (esses >= ess_threshold)
    need = max(1, int(consecutive))
    run = 0
    for i, passed in enumerate(ok):
        run = run + 1 if passed else 0
        if run >= need:
            # Report where the sustained run STARTED, not where it completed.
            return int(draw_counts[i - need + 1])
    return None


def plot_convergence_diagnostics(
    inf_data: az.InferenceData,
    free_params: list[str],
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    step: int = 10,
    save_file: str | None = None,
    show: bool = False,
    system_name: str | None = None,
    include_tuning: bool = True,
    consecutive: int = 1,
) -> dict[str, Any]:
    """Convergence diagnostics vs. cumulative draws: worst-case r-hat, min
    bulk-ESS, and cumulative NUTS divergence count share one x-axis, with a
    line marking the first draw count where both r-hat/ESS criteria are
    jointly met (r_hat < rhat_threshold and ess >= ess_threshold).

    include_tuning=True (default) prepends the warmup phase to the r-hat/ESS
    trajectory -- see _rhat_ess_trajectory's docstring for why the numbers
    during early warmup look bad by construction (non-stationary adaptation,
    not a mixing failure) and shouldn't be over-read. Divergence counts stay
    sampling-only regardless (warmup divergence stats aren't persisted to the
    netcdf), so that panel is left blank during the warmup region rather than
    implying zero.
    """
    _apply_plot_style()
    selected_free_params = _select_free_params(inf_data, free_params)
    draw_counts, rhats, esses, n_tune = _rhat_ess_trajectory(
        inf_data, selected_free_params, step=step, include_tuning=include_tuning
    )
    # The criterion is always evaluated on SAMPLING draws only (field standard --
    # warmup is adaptation, not draws from the target). When the trajectory is
    # displayed on the combined warmup++sampling axis, the marker is offset by
    # n_tune so it lands in the right visual place while the reported number
    # stays "k sampling draws".
    criteria_met_sampling = compute_convergence_criteria_met_at(
        inf_data, selected_free_params, rhat_threshold=rhat_threshold, ess_threshold=ess_threshold,
        step=step, include_tuning=False, consecutive=consecutive,
    )
    criteria_met_at = (
        None if criteria_met_sampling is None else criteria_met_sampling + (n_tune if include_tuning else 0)
    )

    diverging = np.asarray(inf_data.sample_stats["diverging"].values, dtype=bool)  # (chains, draws)
    n_draws = diverging.shape[1]
    cum_divergences = np.cumsum(diverging.sum(axis=0))
    total_divergences = int(cum_divergences[-1]) if n_draws else 0

    fig, (rhat_ax, ess_ax, div_ax) = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)

    rhat_ax.plot(draw_counts, rhats, marker="o", markersize=3.0, linewidth=1.5, label="Worst-case r-hat")
    rhat_ax.axhline(rhat_threshold, color="black", linestyle="--", linewidth=1.2, label=f"Threshold ({rhat_threshold})")
    rhat_ax.set_ylabel("r-hat")
    rhat_ax.set_title("r-hat")
    rhat_ax.legend(loc="best")
    plt.setp(rhat_ax.get_xticklabels(), visible=False)

    ess_ax.plot(draw_counts, esses, marker="o", markersize=3.0, linewidth=1.5, color="tab:orange", label="Min bulk-ESS")
    ess_ax.axhline(ess_threshold, color="black", linestyle="--", linewidth=1.2, label=f"Threshold ({ess_threshold:.0f})")
    ess_ax.set_ylabel("Bulk-ESS")
    ess_ax.legend(loc="best")
    plt.setp(ess_ax.get_xticklabels(), visible=False)

    div_x = np.arange(1, n_draws + 1) + n_tune
    div_ax.plot(div_x, cum_divergences, linewidth=1.5, color="tab:red", label=f"Cumulative divergences (total={total_divergences})")
    div_ax.set_xlabel("Cumulative draws (including warmup)" if n_tune else "Cumulative sampling draws")
    div_ax.set_ylabel("Divergences")
    if total_divergences == 0:
        div_ax.set_ylim(-0.5, 1.0)
    div_ax.legend(loc="best")

    if n_tune:
        for ax in (rhat_ax, ess_ax, div_ax):
            # Shade warmup so it reads as context, not as part of the criterion.
            ax.axvspan(0, n_tune, color="0.85", alpha=0.45, zorder=0)
            ax.axvline(n_tune, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"End of warmup ({n_tune}, excluded from criteria)")
        rhat_ax.legend(loc="best")
        ess_ax.legend(loc="best")
        div_ax.legend(loc="best")

    if criteria_met_at is not None:
        for ax in (rhat_ax, ess_ax, div_ax):
            ax.axvline(
                criteria_met_at, color="firebrick", linestyle=":", linewidth=1.8,
                label=f"Criteria met ({criteria_met_sampling} sampling draws)",
            )
        rhat_ax.legend(loc="best")
        ess_ax.legend(loc="best")
        div_ax.legend(loc="best")

    fig.suptitle(
        f"{system_name} — Convergence Diagnostics" if system_name else "Convergence Diagnostics vs. Sampling Draws"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)

    if not show:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": (rhat_ax, ess_ax, div_ax),
        "draw_counts": draw_counts,
        "rhats": rhats,
        "esses": esses,
        "criteria_met_at": criteria_met_at,
        "n_tune": n_tune,
        "total_divergences": total_divergences,
        "plot_file": plot_file,
    }


def plot_energy_diagnostics(
    inf_data: az.InferenceData,
    bfmi_threshold: float = 0.3,
    save_file: str | None = None,
    show: bool = False,
    system_name: str | None = None,
) -> dict[str, Any]:
    """BFMI (per-chain, left) and the energy distributions it summarizes
    (right): the marginal energy distribution (every value the chain visited)
    vs. the transition distribution (the size of each post-momentum-resampling
    jump). If resampling explores the energy distribution well, the two
    distributions overlap; BFMI is the ratio of their variances, so a much
    narrower transition distribution is exactly what a low BFMI flags.
    Betancourt's conventional caution threshold is 0.3 -- a *floor*, not a
    ceiling; higher is better and there's no upper bound to worry about.
    """
    _apply_plot_style()

    try:
        bfmi_raw = az.bfmi(inf_data)
        try:
            # arviz >=1.0 returns a DataTree/Dataset with an "energy" data_var
            # (one value per chain); older arviz returned a plain ndarray.
            bfmi = np.asarray(bfmi_raw["energy"].values, dtype=float)
        except (KeyError, TypeError, IndexError):
            bfmi = np.asarray(bfmi_raw, dtype=float)
    except Exception:  # noqa: BLE001 - BFMI needs sample_stats.energy; absent on older saved runs
        bfmi = np.array([])

    fig, (bfmi_ax, energy_ax) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    if bfmi.size:
        chain_idx = np.arange(bfmi.size)
        colors = ["tab:red" if b < bfmi_threshold else "tab:blue" for b in bfmi]
        bfmi_ax.bar(chain_idx, bfmi, color=colors)
        bfmi_ax.axhline(bfmi_threshold, color="black", linestyle="--", linewidth=1.2, label=f"Caution threshold ({bfmi_threshold})")
        bfmi_ax.set_xticks(chain_idx)
        bfmi_ax.legend(loc="best")
    else:
        bfmi_ax.text(0.5, 0.5, "BFMI unavailable\n(no energy stat)", ha="center", va="center")
    bfmi_ax.set_xlabel("Chain")
    bfmi_ax.set_ylabel("BFMI")
    bfmi_ax.set_title("Energy Fraction of\nMissing Information")

    energy_vals = np.asarray(inf_data.sample_stats["energy"].values, dtype=float) if "energy" in inf_data.sample_stats else None
    if energy_vals is not None and energy_vals.size:
        centered = energy_vals - energy_vals.mean(axis=1, keepdims=True)
        marginal = centered.reshape(-1)
        transition = np.diff(centered, axis=1).reshape(-1)
        sns.kdeplot(marginal, ax=energy_ax, fill=True, alpha=0.4, label="marginal")
        sns.kdeplot(transition, ax=energy_ax, fill=True, alpha=0.4, label="transition")
        energy_ax.legend(loc="best")
    else:
        energy_ax.text(0.5, 0.5, "Energy distribution\nunavailable", ha="center", va="center")
    energy_ax.set_xlabel("Energy - mean(Energy)")
    energy_ax.set_ylabel("Density")
    energy_ax.set_title("Marginal vs. Transition Energy")

    fig.suptitle(f"{system_name} — Energy Diagnostics" if system_name else "Energy Diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)

    if not show:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": (bfmi_ax, energy_ax),
        "bfmi": bfmi,
        "plot_file": plot_file,
    }


def plot_loo_diagnostics(
    inf_data: az.InferenceData,
    save_file: str | None = None,
    show: bool = False,
    system_name: str | None = None,
) -> dict[str, Any]:
    """PSIS-LOO model-comparison diagnostics in one figure: per-observation
    Pareto k (left) plus an elpd_loo/p_loo text summary (right).

    WAIC is intentionally not included -- this arviz version has dropped it
    from the public API entirely (only loo and its variants remain), which
    tracks the field's own move away from WAIC in favor of PSIS-LOO (Vehtari,
    Gelman & Gabry) as the preferred approximate leave-one-out estimator.

    az.loo needs >=5 roughly-distinct tail draws per observation to fit the
    Pareto tail; very short/early-converged runs can't compute it yet (see
    posterior_burn_in_draws/post_convergence_checks in resumable_sampler.py
    for extending a run specifically to make this stable). That failure is
    caught here and shown as a placeholder rather than raised.
    """
    _apply_plot_style()
    fig, (khat_ax, text_ax) = plt.subplots(1, 2, figsize=(12.0, 4.5), gridspec_kw={"width_ratios": (2.0, 1.0)})
    text_ax.axis("off")

    loo_result = None
    error_message = None
    try:
        loo_result = az.loo(inf_data, pointwise=True)
    except Exception as exc:  # noqa: BLE001 - LOO is diagnostic, not required to compute
        error_message = f"{type(exc).__name__}: {exc}"

    if loo_result is not None:
        khat = np.asarray(loo_result.pareto_k.values, dtype=float).reshape(-1)
        obs_index = np.arange(khat.size)
        good_k = float(loo_result.good_k) if loo_result.good_k is not None else 0.7
        colors = np.where(khat > 1.0, "darkred", np.where(khat > good_k, "tab:orange", "tab:blue"))
        khat_ax.scatter(obs_index, khat, c=colors, s=25.0)
        for threshold, style in ((good_k, "--"), (1.0, ":")):
            khat_ax.axhline(threshold, color="black", linestyle=style, linewidth=1.0)
        khat_ax.set_xlabel("Observation index")
        khat_ax.set_ylabel("Pareto k")
        khat_ax.set_title("PSIS-LOO Pareto k Diagnostic")

        n_bad = int(np.sum(khat > good_k))
        summary_lines = [
            f"elpd_loo = {loo_result.elpd:.2f} ± {loo_result.se:.2f}",
            f"p_loo = {loo_result.p:.2f}",
            f"n observations = {loo_result.n_data_points}",
            f"good_k threshold = {good_k:.2f}",
            f"k > threshold: {n_bad}/{khat.size}"
            + (" (unreliable -- treat elpd_loo cautiously)" if n_bad else " (all reliable)"),
        ]
    else:
        khat_ax.text(0.5, 0.5, "LOO unavailable\n(see summary panel)", ha="center", va="center")
        khat_ax.set_xticks([])
        khat_ax.set_yticks([])
        summary_lines = ["LOO could not be computed:", error_message or "unknown error"]

    text_ax.text(0.0, 0.95, "\n".join(summary_lines), transform=text_ax.transAxes, va="top", fontsize=13)

    fig.suptitle(f"{system_name} — LOO Diagnostics" if system_name else "LOO Diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)

    if not show:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": (khat_ax, text_ax),
        "loo_result": loo_result,
        "error_message": error_message,
        "plot_file": plot_file,
    }


def compute_recovery_metrics(
    inf_data: az.InferenceData,
    param_name: str,
    true_value: float,
) -> dict[str, Any]:
    """Prior-to-posterior recovery diagnostics for one parameter against a
    known synthetic truth: shrinkage, z-score, and credible-interval coverage.

    Shrinkage is in variance terms, 1 - (posterior_sd/prior_sd)**2 (Betancourt/
    Talts-style SBC convention): 0 means the posterior is as wide as the
    prior (uninformative), 1 means a point mass.
    """
    posterior = np.asarray(inf_data.posterior[param_name].values, dtype=float).reshape(-1)
    posterior = posterior[np.isfinite(posterior)]
    prior_group = getattr(inf_data, "prior", None)
    if prior_group is not None and param_name in set(prior_group.data_vars):
        prior = np.asarray(prior_group[param_name].values, dtype=float).reshape(-1)
        prior = prior[np.isfinite(prior)]
    else:
        prior = np.array([])

    posterior_mean = float(np.mean(posterior)) if posterior.size else float("nan")
    posterior_sd = float(np.std(posterior)) if posterior.size else float("nan")
    prior_sd = float(np.std(prior)) if prior.size > 1 else float("nan")
    shrinkage = (1.0 - (posterior_sd / prior_sd) ** 2) if prior_sd > 0 else float("nan")
    z_score = (posterior_mean - true_value) / posterior_sd if posterior_sd > 0 else float("nan")

    coverage: dict[float, bool] = {}
    for mass in (0.50, 0.90, 0.95):
        if posterior.size < 2:
            coverage[mass] = False
            continue
        lower_q, upper_q = (1 - mass) / 2, 1 - (1 - mass) / 2
        lo, hi = np.quantile(posterior, [lower_q, upper_q])
        coverage[mass] = bool(lo <= true_value <= hi)

    return {
        "param_name": param_name,
        "true_value": float(true_value),
        "posterior_mean": posterior_mean,
        "posterior_sd": posterior_sd,
        "prior_sd": prior_sd,
        "shrinkage": shrinkage,
        "z_score": z_score,
        "coverage": coverage,
    }


def plot_posterior_vs_truth(
    inf_data: az.InferenceData,
    param_name: str,
    true_value: float,
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = True,
    system_name: str | None = None,
) -> dict[str, Any]:
    """Posterior density with the known synthetic truth marked, a faint prior
    overlay for context (both scaled to peak 1, same convention as the
    trace-diagnostics prior-vs-posterior panel -- an unscaled wide prior next
    to a narrow posterior renders as an invisible flat line), and
    shrinkage/z-score/coverage annotated directly on the figure.
    """
    _apply_plot_style()
    metrics = compute_recovery_metrics(inf_data, param_name, true_value)

    posterior = np.asarray(inf_data.posterior[param_name].values, dtype=float).reshape(-1)
    posterior = posterior[np.isfinite(posterior)]
    prior_group = getattr(inf_data, "prior", None)
    prior = None
    if prior_group is not None and param_name in set(prior_group.data_vars):
        prior = np.asarray(prior_group[param_name].values, dtype=float).reshape(-1)
        prior = prior[np.isfinite(prior)]

    log_x = (
        use_log_param_axis and true_value > 0.0
        and posterior.size > 0 and np.all(posterior > 0.0)
        and (prior is None or prior.size == 0 or np.all(prior > 0.0))
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if prior is not None and prior.size > 1 and np.nanstd(prior) > 0:
        px, pd_ = _kde_curve(prior, log_x)
        ax.fill_between(px, pd_ / np.max(pd_), color="tab:gray", alpha=0.4, label="Prior")
        ax.plot(px, pd_ / np.max(pd_), color="tab:gray", linewidth=1.2)

    if posterior.size > 1 and np.nanstd(posterior) > 0:
        qx, qd = _kde_curve(posterior, log_x)
        qd = qd / np.max(qd)
        ax.fill_between(qx, qd, color="tab:blue", alpha=0.55, label="Posterior")
        ax.plot(qx, qd, color="tab:blue", linewidth=1.5)
    elif posterior.size == 1:
        ax.axvline(float(posterior[0]), color="tab:blue", linewidth=2.0, label="Posterior")

    ax.axvline(true_value, color="firebrick", linestyle="--", linewidth=1.8, label=f"Truth ({true_value:g})")
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)

    annotation = (
        f"shrinkage = {metrics['shrinkage']:.3f}\n"
        f"z = {metrics['z_score']:+.2f}\n"
        f"coverage (50/90/95%): "
        f"{'Y' if metrics['coverage'][0.50] else 'N'}/"
        f"{'Y' if metrics['coverage'][0.90] else 'N'}/"
        f"{'Y' if metrics['coverage'][0.95] else 'N'}"
    )
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_title(f"{system_name} — {param_name} Recovery" if system_name else f"{param_name} Recovery")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Relative density (each scaled to peak 1)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)
    if not show:
        plt.close(fig)

    return {"figure": fig, "axes": ax, "metrics": metrics, "plot_file": plot_file}


def plot_posterior_correlation(
    inf_data: az.InferenceData,
    param_x: str,
    param_y: str,
    save_file: str | None = None,
    show: bool = False,
    use_log_axes: bool = True,
    system_name: str | None = None,
) -> dict[str, Any]:
    """Pairwise posterior scatter + 2D KDE contours for two named parameters --
    the ridge/correlation visualization for a deliberately-correlated pair
    (e.g. an over-parameterized split of a true single scaling group).
    Lighter cousin of plot_diffusion_feasible_parameter_family, without its
    feasibility-mask machinery. Correlation is computed in log space when
    both axes are positive scale factors, since that is the space in which a
    fixed-ratio relationship is linear.
    """
    _apply_plot_style()
    x = np.asarray(inf_data.posterior[param_x].values, dtype=float).reshape(-1)
    y = np.asarray(inf_data.posterior[param_y].values, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    log_x = use_log_axes and x.size > 0 and bool(np.all(x > 0.0))
    log_y = use_log_axes and y.size > 0 and bool(np.all(y > 0.0))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    correlation = float("nan")
    if x.size > 2 and y.size > 2:
        ax.scatter(x, y, s=4.0, alpha=0.25, color="tab:blue", linewidths=0)
        x_corr = np.log10(x) if log_x else x
        y_corr = np.log10(y) if log_y else y
        try:
            sns.kdeplot(x=x_corr, y=y_corr, ax=ax, levels=5, color="black", linewidths=0.8)
        except Exception:  # noqa: BLE001 - KDE can fail on degenerate/collinear draws; scatter alone still informative
            pass
        correlation = float(np.corrcoef(x_corr, y_corr)[0, 1])

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(
        (f"{system_name} — " if system_name else "") + f"{param_x} vs. {param_y} (r = {correlation:+.2f})"
    )
    fig.tight_layout()

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)
    if not show:
        plt.close(fig)

    return {"figure": fig, "axes": ax, "correlation": correlation, "plot_file": plot_file}


def plot_correlation_matrix(
    inf_data: az.InferenceData,
    free_params: list[str],
    save_file: str | None = None,
    show: bool = False,
    system_name: str | None = None,
) -> dict[str, Any]:
    """NxN posterior correlation heatmap across all fitted parameters (log
    space for any strictly-positive parameter) -- the general answer to
    "which parameters are correlated with each other," complementing
    plot_posterior_correlation's single deliberately-chosen pair.
    """
    _apply_plot_style()
    selected = _select_free_params(inf_data, free_params)
    columns, labels = [], []
    for name in selected:
        values = np.asarray(inf_data.posterior[name].values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size < 2 or np.nanstd(values) == 0.0:
            continue
        columns.append(np.log10(values) if np.all(values > 0.0) else values)
        labels.append(name)

    if len(columns) < 2:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.text(0.5, 0.5, "Fewer than 2 varying parameters", ha="center", va="center")
        ax.axis("off")
    else:
        matrix = np.corrcoef(np.stack(columns))
        fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 2.0, 1.0 * len(labels) + 1.5))
        sns.heatmap(matrix, ax=ax, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f",
                    cmap="RdBu_r", vmin=-1.0, vmax=1.0, square=True, cbar_kws={"label": "Correlation (log space)"})

    ax.set_title(f"{system_name} — Posterior Correlation Matrix" if system_name else "Posterior Correlation Matrix")
    fig.tight_layout()

    plot_file = None
    if save_file:
        plot_path = Path(save_file).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_file = str(plot_path)
    if not show:
        plt.close(fig)

    return {"figure": fig, "axes": ax, "labels": labels, "plot_file": plot_file}


def plot_parameter_marginals(
    inf_data: az.InferenceData,
    free_params: list[str],
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
    system_name: str | None = None,
) -> dict[str, Any]:
    _apply_plot_style()
    selected_free_params = _select_free_params(inf_data, free_params)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    all_positive = True
    for param_name in selected_free_params:
        values = np.asarray(inf_data.posterior[param_name].values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        all_positive = all_positive and bool(np.all(values > 0.0))
        if values.size == 1 or np.nanstd(values) == 0.0:
            ax.axvline(float(values[0]), linewidth=2.0, label=param_name)
        else:
            sns.kdeplot(values, ax=ax, fill=False, linewidth=2.0, label=param_name)

    if use_log_param_axis and all_positive:
        positive_values = []
        for param_name in selected_free_params:
            values = np.asarray(inf_data.posterior[param_name].values, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size > 0:
                positive_values.append(values)
        if positive_values:
            _configure_decade_log_x_axis(ax, np.concatenate(positive_values))

    ax.set_title(f"{system_name} — Posterior Parameter Marginals" if system_name else "Posterior Parameter Marginals")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Density")
    ax.set_xlim([1e-3,1e1])
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

    return {
        "figure": fig,
        "axes": ax,
        "selected_free_params": selected_free_params,
        "plot_file": plot_file,
    }


def plot_priors(
    inf_data: az.InferenceData,
    free_params: list[str],
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
) -> dict[str, Any]:
    _apply_plot_style()
    prior_group = getattr(inf_data, "prior", None)
    if prior_group is None:
        return {
            "figure": None,
            "axes": None,
            "prior_var_names": [],
            "plot_file": None,
        }

    prior_var_names = [name for name in free_params if name in set(prior_group.data_vars)]
    if not prior_var_names:
        return {
            "figure": None,
            "axes": None,
            "prior_var_names": [],
            "plot_file": None,
        }

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    all_positive = True
    for param_name in prior_var_names:
        values = np.asarray(prior_group[param_name].values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        all_positive = all_positive and bool(np.all(values > 0.0))
        if values.size == 1 or np.nanstd(values) == 0.0:
            ax.axvline(float(values[0]), linewidth=2.0, label=param_name)
        else:
            sns.kdeplot(values, ax=ax, fill=False, linewidth=2.0, label=param_name)

    if use_log_param_axis and all_positive:
        positive_values = []
        for param_name in prior_var_names:
            values = np.asarray(prior_group[param_name].values, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size > 0:
                positive_values.append(values)
        if positive_values:
            _configure_decade_log_x_axis(ax, np.concatenate(positive_values))

    ax.set_title("Prior Marginal Distributions")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Density")
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

    return {
        "figure": fig,
        "axes": ax,
        "prior_var_names": prior_var_names,
        "plot_file": plot_file,
    }


def plot_diffusion_feasible_parameter_marginals(
    inf_data: az.InferenceData,
    free_params: list[str],
    feasibility_mask: np.ndarray,
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
) -> dict[str, Any]:
    """Plot posterior marginals with diffusion-feasible area shaded under the curve."""
    _apply_plot_style()
    selected_free_params = _select_free_params(inf_data, free_params)
    mask = _validate_feasibility_mask(feasibility_mask, inf_data, selected_free_params)

    n_params = len(selected_free_params)
    fig, axes = plt.subplots(n_params, 1, figsize=(7.0, 3.2 * n_params), squeeze=False)
    axes_flat = axes[:, 0]
    feasible_intervals: dict[str, tuple[float, float] | None] = {}

    for ax, param_name in zip(axes_flat, selected_free_params):
        values = np.asarray(inf_data.posterior[param_name].values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(values)
        full_values = values[finite_mask]
        feasible_values = values[mask & finite_mask]

        density_line = None
        if full_values.size > 1 and np.nanstd(full_values) > 0:
            sns.kdeplot(full_values, ax=ax, fill=False, linewidth=1.8, label="Full posterior")
            density_line = ax.lines[-1] if ax.lines else None
        elif full_values.size == 1:
            ax.axvline(float(full_values[0]), linewidth=2.0, alpha=0.45, label="Full posterior")

        if use_log_param_axis and full_values.size > 0 and np.all(full_values > 0.0):
            _configure_decade_log_x_axis(ax, full_values)

        if feasible_values.size > 1:
            upper = float(np.nanmax(feasible_values))
            lower = float(ax.get_xlim()[0]) if use_log_param_axis else 0.0
            feasible_intervals[param_name] = (lower, upper)
            if density_line is not None:
                line_color = density_line.get_color()
                x_density = np.asarray(density_line.get_xdata(), dtype=float)
                y_density = np.asarray(density_line.get_ydata(), dtype=float)
                fill_mask = (x_density >= lower) & (x_density <= upper)
                ax.fill_between(
                    x_density,
                    0.0,
                    y_density,
                    where=fill_mask,
                    interpolate=True,
                    color=line_color,
                    alpha=0.28,
                    label="Diffusion-feasible area",
                )
            else:
                ax.axvline(upper, linewidth=2.5, alpha=0.8, label="Diffusion-feasible limit")
        elif feasible_values.size == 1:
            value = float(feasible_values[0])
            feasible_intervals[param_name] = (value, value)
            ax.axvline(
                value,
                color="C2",
                linewidth=2.5,
                alpha=0.8,
                label="Diffusion-feasible value",
            )
        else:
            feasible_intervals[param_name] = None
            ax.text(0.98, 0.85, "No feasible draws", ha="right", va="top", transform=ax.transAxes)

        ax.set_title(f"{param_name}: Posterior with Diffusion-Feasible Area")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        ax.legend(loc="best")

    fig.tight_layout()
    plot_file = _save_figure(fig, save_file)
    if not show:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": axes_flat,
        "selected_free_params": selected_free_params,
        "feasible_intervals": feasible_intervals,
        "feasible_fraction": float(np.mean(mask)),
        "plot_file": plot_file,
    }


def plot_diffusion_feasible_parameter_family(
    inf_data: az.InferenceData,
    free_params: list[str],
    feasibility_mask: np.ndarray,
    save_file: str | None = None,
    show: bool = False,
    max_params: int = 6,
) -> dict[str, Any]:
    """Pairwise free-parameter plots with globally feasible draws highlighted."""
    _apply_plot_style()
    selected_free_params = _select_free_params(inf_data, free_params)[:max_params]
    mask = _validate_feasibility_mask(feasibility_mask, inf_data, selected_free_params)

    if len(selected_free_params) < 2:
        return {
            "figure": None,
            "axes": None,
            "selected_free_params": selected_free_params,
            "feasible_fraction": float(np.mean(mask)),
            "plot_file": None,
        }

    pairs = list(combinations(selected_free_params, 2))
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax, (x_name, y_name) in zip(axes_flat, pairs):
        x_values = np.asarray(inf_data.posterior[x_name].values, dtype=float).reshape(-1)
        y_values = np.asarray(inf_data.posterior[y_name].values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        feasible = mask & finite_mask
        all_draws = finite_mask

        ax.scatter(x_values[all_draws], y_values[all_draws], s=12, alpha=0.18, label="Full posterior")
        if np.any(feasible):
            ax.scatter(x_values[feasible], y_values[feasible], s=16, alpha=0.75, label="Diffusion-feasible")
        else:
            ax.text(0.98, 0.95, "No feasible draws", ha="right", va="top", transform=ax.transAxes)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend(loc="best")

    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    fig.suptitle("Free-Parameter Family Supporting Diffusion-Limited Effective Rates")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plot_file = _save_figure(fig, save_file)
    if not show:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": axes,
        "selected_free_params": selected_free_params,
        "feasible_fraction": float(np.mean(mask)),
        "plot_file": plot_file,
    }


def _validate_feasibility_mask(
    feasibility_mask: np.ndarray,
    inf_data: az.InferenceData,
    selected_free_params: list[str],
) -> np.ndarray:
    mask = np.asarray(feasibility_mask, dtype=bool).reshape(-1)
    expected_size = np.asarray(inf_data.posterior[selected_free_params[0]].values).reshape(-1).size
    if mask.size != expected_size:
        raise ValueError(
            f"feasibility_mask has {mask.size} entries; expected {expected_size} flattened posterior draws."
        )
    return mask


def _save_figure(fig: Any, save_file: str | None) -> str | None:
    if not save_file:
        return None
    plot_path = Path(save_file).expanduser().resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    return str(plot_path)


def plot_predictive_time_vs_observable(
    posterior_samples: np.ndarray,
    x_values: np.ndarray,
    observed_values: np.ndarray,
    observed_sigma: np.ndarray,
    dataset_name: str,
    output_name: str,
    x_label: str,
    y_label: str,
    posterior_color: str = "C0",
    observed_color: str = "C1",
    ax: Any | None = None,
) -> tuple[Any, Any]:
    _apply_plot_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure
    pred_low, pred_high = np.nanpercentile(posterior_samples, [2.5, 97.5], axis=0)
    pred_mean = np.nanmean(posterior_samples, axis=0)

    ax.fill_between(
        x_values,
        pred_low,
        pred_high,
        alpha=0.25,
        color=posterior_color,
        label="Posterior 95% CI",
    )
    ax.plot(x_values, pred_mean, linewidth=2.0, color=posterior_color, label="Posterior Mean")
    ax.errorbar(
        x_values,
        observed_values,
        yerr=observed_sigma,
        fmt="o",
        markersize=4,
        linestyle="none",
        color=observed_color,
        ecolor=observed_color,
        capsize=3,
        label="Observed",
    )

    ax.set_title(f"Time Course: {output_name}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    if standalone:
        fig.tight_layout()

    return fig, ax


def plot_predictive_observable_vs_initial_concentration_table(
    posterior_samples: np.ndarray,
    observed_values: np.ndarray,
    observed_sigma: np.ndarray,
    dataset_name: str,
    output_name: str,
    y_label: str,
    table_rows: list[tuple[str, list[str]]],
    posterior_color: str = "C0",
    observed_color: str = "C1",
    ax: Any | None = None,
    table_height_fraction: float | None = None,
) -> tuple[Any, Any]:
    _apply_plot_style()
    n_points = observed_values.shape[0]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
    else:
        fig = ax.figure

    bar_positions = np.arange(n_points)
    bar_width = 0.38
    pred_low, pred_high = np.nanpercentile(posterior_samples, [2.5, 97.5], axis=0)
    pred_mean = np.nanmean(posterior_samples, axis=0)

    pred_bar_positions = bar_positions - (bar_width / 2.0)
    obs_bar_positions = bar_positions + (bar_width / 2.0)

    ax.bar(
        pred_bar_positions,
        pred_mean,
        alpha=0.45,
        width=bar_width,
        label="Posterior Mean",
        color=posterior_color,
    )
    ax.errorbar(
        pred_bar_positions,
        pred_mean,
        yerr=[pred_mean - pred_low, pred_high - pred_mean],
        fmt="none",
        ecolor=posterior_color,
        capsize=3,
    )
    ax.bar(
        obs_bar_positions,
        observed_values,
        alpha=0.55,
        width=bar_width,
        label="Observed",
        color=observed_color,
    )
    ax.errorbar(
        obs_bar_positions,
        observed_values,
        yerr=observed_sigma,
        fmt="none",
        ecolor=observed_color,
        capsize=3,
    )

    # Keep bar-group centers at integer positions with no extra x padding so
    # table columns can align exactly with each grouped bar position.
    ax.set_xlim(-0.5, n_points - 0.5)
    ax.margins(x=0.0)

    ax.set_title(f"Final Concentration vs Initial Conditions: {output_name}")
    ax.set_ylabel(y_label)
    # The table columns provide the bar reference, so hide numeric x-axis tick labels.
    ax.set_xticks([])
    ax.legend(loc="best")

    if table_rows:
        table_row_labels = [row_label for row_label, _ in table_rows]
        table_cell_text = [row_values for _, row_values in table_rows]

        # Size columns to their actual rendered text (plus a small buffer)
        # instead of splitting the width evenly -- a "1e+03" column and a "3"
        # column don't need the same room. Data columns still span exactly
        # axes-fraction [0, 1] (so they stay aligned with the bars above),
        # with the species column's width expressed relative to that same
        # data region and extended to the left of it.
        species_col_width_in, data_col_widths_in = _measure_table_column_widths_in(
            table_row_labels, table_cell_text, TABLE_FONT_SIZE
        )
        total_data_width_in = sum(data_col_widths_in) or 1.0
        data_col_fractions = [w / total_data_width_in for w in data_col_widths_in]
        species_col_fraction = species_col_width_in / total_data_width_in
        total_width = 1.0 + species_col_fraction
        col_widths = [species_col_fraction / total_width] + [f / total_width for f in data_col_fractions]

        # Row height derived from font metrics (fixed per-row height) rather
        # than a formula that grows steeply with row count -- keeps many-
        # species tables compact instead of ballooning vertically.
        if table_height_fraction is None:
            row_height_in = (TABLE_FONT_SIZE / 72.0) * 1.8 + 0.22
            table_height_fraction = (len(table_row_labels) * row_height_in) / 4.0
        table_bottom = -(table_height_fraction + 0.05)

        # Build a custom first column for species names because Matplotlib rowLabels
        # do not reliably honor explicit width updates.
        table_cell_text_with_species = [
            [table_row_labels[row_index], *table_cell_text[row_index]]
            for row_index in range(len(table_row_labels))
        ]

        table = ax.table(
            cellText=table_cell_text_with_species,
            bbox=[-species_col_fraction, table_bottom, total_width, table_height_fraction],
            colWidths=col_widths,
            cellLoc="center",
        )

        for (row_index, col_index), cell in table.get_celld().items():
            if col_index == 0:
                cell.get_text().set_ha("center")

        # Deliberately smaller than the rest of the figure's font (PLOT_FONT_SIZE) --
        # tables pack far more text into the same width than an axis label does.
        table.auto_set_font_size(False)
        table.set_fontsize(TABLE_FONT_SIZE)
        if standalone:
            bottom_margin = 0.12 + 0.06 * len(table_row_labels)
            fig.subplots_adjust(left=0.14, right=0.98, bottom=bottom_margin)
            fig.text(
                0.5,
                0.03,
                "Initial Conditions (uM)",
                ha="center",
                va="center",
                fontsize=TABLE_FONT_SIZE,
            )
        else:
            # Embedded in a shared grid figure -- the caller owns figure-level
            # margins, so label in axes coordinates just below the table
            # instead of touching fig.subplots_adjust/fig.text.
            ax.text(
                0.5,
                table_bottom - 0.04,
                "Initial Conditions (uM)",
                ha="center",
                va="top",
                fontsize=TABLE_FONT_SIZE,
                transform=ax.transAxes,
            )
    else:
        ax.set_xlabel("Initial Conditions")

    return fig, ax


def plot_predictive(
    inf_data: az.InferenceData,
    experiment: Any,
    save_dir: str | Path | None = None,
    file_stem: str = "trace_plot",
    show: bool = False,
    posterior_color: str = "C0",
    observed_color: str = "C1",
    system_name: str | None = None,
) -> dict[str, Any]:
    predictive_values, predictive_var_name = _extract_predictive_data(inf_data)
    empty_result = {
        "figure": None,
        "axes": None,
        "predictive_figures": [],
        "data_vs_results_plot_files": {},
        "plot_file": None,
        "predictive_var_name": predictive_var_name,
    }
    if predictive_values is None or experiment is None:
        return empty_result

    if predictive_values.ndim != 4:
        raise ValueError("Expected posterior predictive variable with shape (chains, draws, 1, n_obs).")

    save_dir_path = Path(save_dir).expanduser().resolve() if save_dir else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    observed_sigma_all = np.asarray(experiment.observed_sigma, dtype=float).reshape(-1)

    # First pass: gather one panel spec per (dataset, observable) pair so the
    # grid can be sized correctly before any axes are created.
    panel_specs: list[dict[str, Any]] = []
    obs_offset = 0
    for dataset in experiment.datasets:
        n_points = len(dataset.time_values)
        if n_points == 0:
            continue

        dataset_type = str(dataset.dataset_type).lower()
        x_values = np.asarray(dataset.time_values, dtype=float)
        dataset_noise_model = str(getattr(dataset, "noise_model", "")).lower()

        dataset_observable_results = []
        last_table_rows = None
        for output_name, column_name in dataset.observables_mapping:
            chunk = predictive_values[:, :, 0, obs_offset : obs_offset + n_points]
            chunk = chunk.reshape(-1, n_points)
            observed_values = dataset.frame[column_name].to_numpy(dtype=float)
            observed_sigma = observed_sigma_all[obs_offset : obs_offset + n_points]

            if dataset_noise_model == "groupwise":
                (
                    x_values_for_plot,
                    observed_values_for_plot,
                    observed_sigma_for_plot,
                    posterior_chunk_for_plot,
                ) = _aggregate_groupwise_series(
                    x_values=x_values,
                    observed_values=observed_values,
                    observed_sigma=observed_sigma,
                    posterior_samples=chunk,
                )
            else:
                x_values_for_plot = x_values
                observed_values_for_plot = observed_values
                observed_sigma_for_plot = observed_sigma
                posterior_chunk_for_plot = chunk

            last_table_rows = (
                _build_table_rows(dataset=dataset, experiment=experiment, n_points=int(observed_values_for_plot.size))
                if dataset_type == "endpoint"
                else None
            )
            dataset_observable_results.append(
                {
                    "x_values": x_values_for_plot,
                    "observed_values": observed_values_for_plot,
                    "observed_sigma": observed_sigma_for_plot,
                    "posterior_chunk": posterior_chunk_for_plot,
                }
            )
            panel_specs.append(
                {
                    "dataset_type": dataset_type,
                    "dataset_name": str(dataset.name),
                    "output_name": str(output_name),
                    "x_values": x_values_for_plot,
                    "observed_values": observed_values_for_plot,
                    "observed_sigma": observed_sigma_for_plot,
                    "posterior_chunk": posterior_chunk_for_plot,
                    "x_label": str(dataset.time_column if dataset.time_column else "time"),
                    "table_rows": last_table_rows,
                }
            )
            obs_offset += n_points

        # A dataset spanning multiple chain-length products (e.g. C4_FA,
        # C6_FA, ... for a Chain C12 system) also gets one combined "total
        # fatty acids" panel -- observed sigmas add in quadrature since each
        # observable's noise is independent.
        if len(dataset_observable_results) > 1:
            total_x_values = dataset_observable_results[0]["x_values"]
            total_observed_values = sum(r["observed_values"] for r in dataset_observable_results)
            total_observed_sigma = np.sqrt(sum(r["observed_sigma"] ** 2 for r in dataset_observable_results))
            total_posterior_chunk = sum(r["posterior_chunk"] for r in dataset_observable_results)
            panel_specs.append(
                {
                    "dataset_type": dataset_type,
                    "dataset_name": str(dataset.name),
                    "output_name": "Total FA (uM)",
                    "x_values": total_x_values,
                    "observed_values": total_observed_values,
                    "observed_sigma": total_observed_sigma,
                    "posterior_chunk": total_posterior_chunk,
                    "x_label": str(dataset.time_column if dataset.time_column else "time"),
                    "table_rows": last_table_rows,
                }
            )

    if not panel_specs:
        return empty_result

    # One combined figure, one row per chain length (observable): its
    # time-course panel next to its final-concentration/table panel.
    bar_chart_height_in = 4.0
    ts_panel_width_in = 6.0
    table_row_height_in = (PLOT_FONT_SIZE / 72.0) * 1.8 + 0.22

    output_order: list[str] = []
    groups: dict[str, dict[str, Any]] = {}
    for spec in panel_specs:
        key = spec["output_name"]
        if key not in groups:
            groups[key] = {}
            output_order.append(key)
        slot = "endpoint" if spec["dataset_type"] == "endpoint" else "timeseries"
        groups[key][slot] = spec

    system_tag = (
        system_name.split(" - ")[0].replace("Chain ", "").strip() if system_name else None
    ) or file_stem

    # Precompute each row's endpoint-table geometry; the endpoint column
    # width is shared across all rows (sized to the widest table) so the
    # grid stays rectangular.
    row_layout = []
    for output_name in output_order:
        ep_spec = groups[output_name].get("endpoint")
        if ep_spec is None:
            row_layout.append({"table_height_in": 0.0, "ep_width_in": 0.0})
            continue
        table_row_labels = [label for label, _ in ep_spec["table_rows"]]
        table_cell_text = [values for _, values in ep_spec["table_rows"]]
        species_width_in, data_widths_in = _measure_table_column_widths_in(
            table_row_labels, table_cell_text, PLOT_FONT_SIZE
        )
        row_layout.append(
            {
                "table_height_in": len(table_row_labels) * table_row_height_in,
                "ep_width_in": species_width_in + sum(data_widths_in),
            }
        )

    has_ts = any(groups[name].get("timeseries") is not None for name in output_order)
    has_ep = any(groups[name].get("endpoint") is not None for name in output_order)
    ep_col_width_in = max((layout["ep_width_in"] for layout in row_layout), default=0.0)
    ep_col_width_in = max(ep_col_width_in, 3.0)
    width_ratios = [w for w in (ts_panel_width_in if has_ts else None, ep_col_width_in if has_ep else None) if w]
    n_cols = len(width_ratios)
    row_heights_in = [bar_chart_height_in + layout["table_height_in"] for layout in row_layout]
    fig_width_in = sum(width_ratios)
    fig_height_in = sum(row_heights_in)

    _apply_plot_style()
    # Plain GridSpec + bbox_inches="tight" at save time -- constrained_layout
    # fights the endpoint tables' out-of-axes-bounds bbox and collapses axes
    # to zero trying to "fit" them.
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    gs = fig.add_gridspec(
        len(output_order), n_cols, width_ratios=width_ratios, height_ratios=row_heights_in, hspace=0.55, wspace=0.3
    )

    for row_index, output_name in enumerate(output_order):
        group = groups[output_name]
        ts_spec = group.get("timeseries")
        ep_spec = group.get("endpoint")
        layout = row_layout[row_index]

        col = 0
        if ts_spec is not None:
            ax_ts = fig.add_subplot(gs[row_index, col])
            col += 1
            plot_predictive_time_vs_observable(
                posterior_samples=ts_spec["posterior_chunk"],
                x_values=ts_spec["x_values"],
                observed_values=ts_spec["observed_values"],
                observed_sigma=ts_spec["observed_sigma"],
                dataset_name=ts_spec["dataset_name"],
                output_name=ts_spec["output_name"],
                x_label=ts_spec["x_label"],
                y_label=ts_spec["output_name"],
                posterior_color=posterior_color,
                observed_color=observed_color,
                ax=ax_ts,
            )
        elif has_ep and col == 0:
            col += 1

        if ep_spec is not None:
            ax_ep = fig.add_subplot(gs[row_index, col])
            table_height_fraction = layout["table_height_in"] / bar_chart_height_in
            plot_predictive_observable_vs_initial_concentration_table(
                posterior_samples=ep_spec["posterior_chunk"],
                observed_values=ep_spec["observed_values"],
                observed_sigma=ep_spec["observed_sigma"],
                dataset_name=ep_spec["dataset_name"],
                output_name=ep_spec["output_name"],
                y_label=ep_spec["output_name"],
                table_rows=ep_spec["table_rows"],
                posterior_color=posterior_color,
                observed_color=observed_color,
                ax=ax_ep,
                table_height_fraction=table_height_fraction,
            )
            # The bar-chart axes only needs bar_chart_height_in of this row's
            # full height -- shrink it to that fraction, top-anchored, so the
            # rest is genuinely blank space for the table to draw into.
            pos = ax_ep.get_position()
            shrink = bar_chart_height_in / row_heights_in[row_index]
            ax_ep.set_position([pos.x0, pos.y0 + pos.height * (1.0 - shrink), pos.width, pos.height * shrink])

    title = f"{system_name} — Posterior Predictive Checks" if system_name else "Posterior Predictive Checks"
    fig.suptitle(title)

    plot_file = None
    if save_dir_path is not None:
        plot_file = save_dir_path / f"predictive_plots_{system_tag}.png"
        fig.savefig(plot_file, dpi=200, bbox_inches="tight")

    if not show:
        plt.close(fig)

    data_vs_results_plot_files = {"combined": str(plot_file)} if plot_file is not None else {}

    return {
        "figure": fig,
        "axes": None,
        "predictive_figures": [fig],
        "data_vs_results_plot_files": data_vs_results_plot_files,
        "plot_file": str(plot_file) if plot_file is not None else None,
        "predictive_var_name": predictive_var_name,
    }


def plot_inference_diagnostics(
    inf_data: az.InferenceData,
    free_params: list[str],
    experiment: Any | None = None,
    save_path: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
    include_tuning_in_trace: bool = False,
    posterior_start_line: bool = True,
) -> dict[str, Any]:
    selected_free_params = _select_free_params(inf_data, free_params)
    summary = az.summary(inf_data, var_names=selected_free_params, round_to=4)

    paths = _resolve_paths(save_path)

    trace_artifacts = plot_posterior_trace_diagnostics(
        inf_data=inf_data,
        free_params=selected_free_params,
        save_file=str(paths["trace"]) if paths["trace"] is not None else None,
        show=show,
        use_log_param_axis=use_log_param_axis,
        include_tuning=include_tuning_in_trace,
        posterior_start_line=posterior_start_line,
    )
    posterior_marginal_artifacts = plot_parameter_marginals(
        inf_data=inf_data,
        free_params=selected_free_params,
        save_file=str(paths["posterior_marginals"]) if paths["posterior_marginals"] is not None else None,
        show=show,
        use_log_param_axis=use_log_param_axis,
    )
    prior_artifacts = plot_priors(
        inf_data=inf_data,
        free_params=selected_free_params,
        save_file=str(paths["prior_marginals"]) if paths["prior_marginals"] is not None else None,
        show=show,
        use_log_param_axis=use_log_param_axis,
    )
    predictive_artifacts = plot_predictive(
        inf_data=inf_data,
        experiment=experiment,
        save_dir=paths["save_dir"],
        file_stem=str(paths["stem"] or "trace_plot"),
        show=show,
    )

    return {
        "summary": summary,
        "figure": trace_artifacts["figure"],
        "axes": trace_artifacts["axes"],
        "trace_plot_file": trace_artifacts["plot_file"],
        "trace_diagnostics": trace_artifacts,
        "parameter_marginals": posterior_marginal_artifacts,
        "parameter_marginals_plot_file": posterior_marginal_artifacts["plot_file"],
        "prior_marginals": prior_artifacts,
        "prior_marginals_plot_file": prior_artifacts["plot_file"],
        "predictive": predictive_artifacts,
        "predictive_figures": predictive_artifacts["predictive_figures"],
        "data_vs_results_plot_files": predictive_artifacts["data_vs_results_plot_files"],
        # Compatibility aliases for existing consumers.
        "parameter_density_figure": posterior_marginal_artifacts["figure"],
        "parameter_density_axes": posterior_marginal_artifacts["axes"],
        "parameter_density_plot_file": posterior_marginal_artifacts["plot_file"],
        "prior_density_figure": prior_artifacts["figure"],
        "prior_density_axes": prior_artifacts["axes"],
        "prior_density_var_names": prior_artifacts["prior_var_names"],
        "prior_density_plot_file": prior_artifacts["plot_file"],
        "data_vs_results_legend_plot_files": {},
    }
