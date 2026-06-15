from __future__ import annotations

from pathlib import Path
from typing import Any
from itertools import combinations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


PLOT_FONT_SIZE = 16
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


def plot_posterior_trace_diagnostics(
    inf_data: az.InferenceData,
    free_params: list[str],
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
) -> dict[str, Any]:
    _apply_plot_style()
    selected_free_params = _select_free_params(inf_data, free_params)

    n_params = len(selected_free_params)
    # Use 6x4 per subplot panel (2 columns => 12 width, n rows => 4*n height).
    fig, axes = plt.subplots(n_params, 2, figsize=(12.0, 4.0 * n_params), squeeze=False)

    for row_index, param_name in enumerate(selected_free_params):
        posterior_values = np.asarray(inf_data.posterior[param_name].values, dtype=float)
        posterior_flat = posterior_values[np.isfinite(posterior_values)]

        density_ax = axes[row_index, 0]
        if posterior_flat.size > 1 and np.nanstd(posterior_flat) > 0:
            sns.kdeplot(posterior_flat, ax=density_ax, fill=True, label="Posterior density")
        elif posterior_flat.size == 1:
            density_ax.axvline(float(posterior_flat[0]), linewidth=2.0, label="Posterior value")
        else:
            density_ax.text(0.5, 0.5, "No finite samples", ha="center", va="center")

        if use_log_param_axis and posterior_flat.size > 0 and np.all(posterior_flat > 0.0):
            _configure_decade_log_x_axis(density_ax, posterior_flat)

        density_ax.set_title(f"{param_name} Posterior")
        density_ax.set_xlabel("Parameter Value")
        density_ax.set_ylabel("Density")
        density_ax.legend(loc="best")

        trace_ax = axes[row_index, 1]
        if posterior_values.ndim < 2:
            trace_series = posterior_values.reshape(1, -1)
        else:
            chains = posterior_values.shape[0]
            draws = posterior_values.shape[1]
            trace_series = posterior_values.reshape(chains, draws, -1)[:, :, 0]

        x_draws = np.arange(trace_series.shape[1], dtype=int)
        for chain_index in range(trace_series.shape[0]):
            trace_ax.plot(x_draws, trace_series[chain_index], linewidth=1.0, label=f"Chain {chain_index}")

        trace_ax.set_title(f"{param_name} Trace")
        trace_ax.set_xlabel("Draw")
        trace_ax.set_ylabel("Parameter Value")
        trace_ax.legend(loc="best")

    fig.suptitle("Posterior Trace Diagnostics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

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


def plot_parameter_marginals(
    inf_data: az.InferenceData,
    free_params: list[str],
    save_file: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
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

    ax.set_title("Posterior Parameter Marginals")
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
) -> tuple[Any, Any]:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
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

    ax.set_title(f"{dataset_name} ({output_name})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
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
) -> tuple[Any, Any]:
    _apply_plot_style()
    n_points = observed_values.shape[0]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

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

    ax.set_title(f"{dataset_name} ({output_name})")
    ax.set_ylabel(y_label)
    # The table columns provide the bar reference, so hide numeric x-axis tick labels.
    ax.set_xticks([])
    ax.legend(loc="best")

    if table_rows:
        table_row_labels = [row_label for row_label, _ in table_rows]
        table_cell_text = [row_values for _, row_values in table_rows]
        # Keep the table close to the axis while preserving a usable bar plotting region.
        table_height = 0.05+0.03*(len(table_row_labels)**1.88)
        table_bottom = -(table_height + 0.05)

        # Build a custom first column for species names because Matplotlib rowLabels
        # do not reliably honor explicit width updates.
        species_col_fraction = 0.1
        table_cell_text_with_species = [
            [table_row_labels[row_index], *table_cell_text[row_index]]
            for row_index in range(len(table_row_labels))
        ]

        # Keep numeric data columns spanning exactly [0, 1] in axis coordinates and
        # place the species column in a dedicated left extension.
        total_width = 1.0 + species_col_fraction
        col_widths = [species_col_fraction / total_width] + [
            (1.0 / max(n_points, 1)) / total_width for _ in range(n_points)
        ]

        table = ax.table(
            cellText=table_cell_text_with_species,
            bbox=[-species_col_fraction, table_bottom, total_width, table_height],
            colWidths=col_widths,
            cellLoc="center",
        )

        for (row_index, col_index), cell in table.get_celld().items():
            if col_index == 0:
                cell.get_text().set_ha("center")

        # Match global plot font sizing for readability consistency.
        table.auto_set_font_size(False)
        table.set_fontsize(PLOT_FONT_SIZE)
        bottom_margin = 0.12 + 0.06 * len(table_row_labels)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=bottom_margin)
        fig.text(
            0.5,
            0.03,
            "Initial Conditions (uM)",
            ha="center",
            va="center",
            fontsize=PLOT_FONT_SIZE,
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
) -> dict[str, Any]:
    predictive_values, predictive_var_name = _extract_predictive_data(inf_data)
    if predictive_values is None or experiment is None:
        return {
            "predictive_figures": [],
            "data_vs_results_plot_files": {},
            "predictive_var_name": predictive_var_name,
        }

    if predictive_values.ndim != 4:
        raise ValueError("Expected posterior predictive variable with shape (chains, draws, 1, n_obs).")

    save_dir_path = Path(save_dir).expanduser().resolve() if save_dir else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    predictive_figures = []
    data_vs_results_plot_files: dict[str, str] = {}
    observed_sigma_all = np.asarray(experiment.observed_sigma, dtype=float).reshape(-1)

    obs_offset = 0
    for dataset in experiment.datasets:
        n_points = len(dataset.time_values)
        if n_points == 0:
            continue

        dataset_type = str(dataset.dataset_type).lower()
        x_values = np.asarray(dataset.time_values, dtype=float)
        dataset_noise_model = str(getattr(dataset, "noise_model", "")).lower()

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


            if dataset_type == "timeseries":
                fig_pred, _ = plot_predictive_time_vs_observable(
                    posterior_samples=posterior_chunk_for_plot,
                    x_values=x_values_for_plot,
                    observed_values=observed_values_for_plot,
                    observed_sigma=observed_sigma_for_plot,
                    dataset_name=str(dataset.name),
                    output_name=str(output_name),
                    x_label=str(dataset.time_column if dataset.time_column else "time"),
                    y_label=str(output_name),
                    posterior_color=posterior_color,
                    observed_color=observed_color,
                )
            elif dataset_type == "endpoint":
                table_rows = _build_table_rows(
                    dataset=dataset,
                    experiment=experiment,
                    n_points=int(observed_values_for_plot.size),
                )
                fig_pred, _ = plot_predictive_observable_vs_initial_concentration_table(
                    posterior_samples=posterior_chunk_for_plot,
                    observed_values=observed_values_for_plot,
                    observed_sigma=observed_sigma_for_plot,
                    dataset_name=str(dataset.name),
                    output_name=str(output_name),
                    y_label=str(output_name),
                    table_rows=table_rows,
                    posterior_color=posterior_color,
                    observed_color=observed_color,
                )
            else:
                fig_pred, _ = plot_predictive_time_vs_observable(
                    posterior_samples=posterior_chunk_for_plot,
                    x_values=x_values_for_plot,
                    observed_values=observed_values_for_plot,
                    observed_sigma=observed_sigma_for_plot,
                    dataset_name=str(dataset.name),
                    output_name=str(output_name),
                    x_label=str(dataset.time_column if dataset.time_column else "time"),
                    y_label=str(output_name),
                    posterior_color=posterior_color,
                    observed_color=observed_color,
                )

            predictive_figures.append(fig_pred)

            if save_dir_path is not None:
                safe_dataset = _sanitize_name(str(dataset.name), "dataset")
                safe_output = _sanitize_name(str(output_name), "output")
                plot_file = save_dir_path / f"{file_stem}_predictive_{safe_dataset}_{safe_output}.pdf"
                fig_pred.savefig(plot_file, dpi=300, bbox_inches="tight")
                data_vs_results_plot_files[f"{dataset.name}::{output_name}"] = str(plot_file)

            if not show:
                plt.close(fig_pred)

            obs_offset += n_points

    return {
        "predictive_figures": predictive_figures,
        "data_vs_results_plot_files": data_vs_results_plot_files,
        "predictive_var_name": predictive_var_name,
    }


def plot_inference_diagnostics(
    inf_data: az.InferenceData,
    free_params: list[str],
    experiment: Any | None = None,
    save_path: str | None = None,
    show: bool = False,
    use_log_param_axis: bool = False,
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
