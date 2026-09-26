"""Draft figures from the Tier-1 post-processing JSON (Figs 6, 7, 8 and 9).

  python plot_tier1_drafts.py fig6 [identifiability.json]
  python plot_tier1_drafts.py fig9 [expected_information_grid.json]
  python plot_tier1_drafts.py fig7 [posterior_morris.json]
  python plot_tier1_drafts.py fig8 [posterior_ratio_response.json]

Each writes a PNG next to its JSON. These are drafts for choosing what the paper's figures
show, not final artwork. Fig 3's draft comes from sbc.py ranks.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POSTERIOR, POINT = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
# Diverging: red arm, gray midpoint, blue arm, equal steps per side (for signed values in [-1, 1]).
DIV = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#8f2424", "#e34948", "#f4b3b0", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def heatmap(ax, values, rows, cols, fmt, title, signed=False):
    if signed:
        im = ax.imshow(values, cmap=DIV, vmin=-1, vmax=1, aspect="auto")
    else:
        im = ax.imshow(values, cmap=SEQ, aspect="auto")
    lo, hi = np.nanmin(values), np.nanmax(values)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = values[i, j]
            dark = abs(v) > 0.6 if signed else (v - lo) / (hi - lo + 1e-300) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK_2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig6(path):
    d = json.loads(Path(path).read_text())
    names = d["params"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=150, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ax = axes[0]
    style(ax)
    y = np.arange(len(names))[::-1]
    shrink = [d["shrinkage"][p] for p in names]
    ax.barh(y, shrink, height=0.5, color=POSTERIOR)
    for yi, v in zip(y, shrink):
        ax.text(max(v, 0) + 0.02, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=INK_2, lw=1, ls="--")
    ax.set_ylim(-0.85, y[0] + 0.4)
    ax.text(0.48, -0.62, "weak below 0.5", ha="right", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, [f"{p} ({d['scale'][p]})" for p in names], fontsize=8, color=INK_2)
    ax.set_xlim(min(0, min(shrink)) - 0.02, 1.12)
    ax.set_xlabel("1 - posterior sd / prior sd", fontsize=8, color=INK_2)
    ax.set_title("A  Shrinkage per parameter", fontsize=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    heatmap(axes[1], np.array(d["correlation"]), names, names, "{:+.2f}", "B  Posterior correlation",
            signed=True)
    eig = d["eigen"]
    loadings = np.array([[e["direction"][p] for p in names] for e in eig])
    heatmap(axes[2], loadings, [f"{e['variance_left']:.2g} left" for e in eig], names, "{:+.2f}",
            "C  Covariance directions, tightest first", signed=True)
    axes[2].set_ylabel("fraction of prior variance", fontsize=8, color=INK_2)
    fig.suptitle(f"Identifiability from {d['n_draws']} posterior draws ({Path(d['posterior']).parent.name}"
                 + (f", {d['key']}" if d.get("key") else "") + ")", fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Log scale for LogNormal parameters. C: eigenvectors of the posterior covariance with "
             "each parameter scaled by its prior sd (prior = identity); rows sum to 1 in squares.",
             fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path(path).with_suffix(".png")
    fig.savefig(out, facecolor=SURFACE)
    return out


def fig9(path):
    d = json.loads(Path(path).read_text())
    cells = list(d["grid"].values())
    rows = list(dict.fromkeys(c["measurement"] for c in cells))
    cols = list(dict.fromkeys(c["timing"] for c in cells))
    get = lambda key: np.array([[next(c[key] for c in cells if c["measurement"] == r and c["timing"] == t)
                                 for t in cols] for r in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), dpi=150, facecolor=SURFACE)
    heatmap(axes[0], get("info_nats_per_point"), rows, cols, "{:.3f}", "A  Expected information per data point (nats)")
    heatmap(axes[1], get("mean_shrinkage_log"), rows, cols, "{:.3f}",
            f"B  Mean shrinkage, log scale ({', '.join(d['params'])})")
    t1 = d["tier1_design"]
    fig.text(0.01, 0.01, f"{d['system']}, five Tier-1 conditions. Tier-1 design for reference: "
             f"{t1['info_nats_per_point']:.3f} nats per point over {t1['n_points']} points, mean shrinkage "
             f"{t1['mean_shrinkage_log']:.3f}.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path(path).with_suffix(".png")
    fig.savefig(out, facecolor=SURFACE)
    return out


def fig7(path):
    d = json.loads(Path(path).read_text())
    objs = d["objectives"]
    fig, axes = plt.subplots(1, len(objs), figsize=(4.2 * len(objs), 3.6), dpi=150, facecolor=SURFACE)
    for ax, (name, v) in zip(np.atleast_1d(axes), objs.items()):
        style(ax)
        rows = v["enzymes"]                                  # sorted by point-estimate rank
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows):
            lo, hi = r["mu_star_posterior_5_95"]
            ax.plot([lo, hi], [yi, yi], color=POSTERIOR, lw=2, solid_capstyle="round")
        ax.scatter([r["mu_star_posterior_median"] for r in rows], y, s=36, color=POSTERIOR, zorder=3,
                   label="posterior median, 5-95%")
        ax.scatter([r["mu_star_point"] for r in rows], y, s=40, marker="D", color=POINT, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2, label="point estimate")
        ax.set_yticks(y, [r["enzyme"] for r in rows], fontsize=8, color=INK_2)
        ax.set_xlabel("Morris mu*", fontsize=8, color=INK_2)
        ax.set_title(f"{name.replace('_', ' ')}   Spearman {v['spearman_vs_point_median']:.2f}",
                     fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK_2)
    fig.suptitle(f"Enzyme sensitivity across {d['n_draws']} posterior draws vs the point estimate ({d['system']})",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = Path(path).with_suffix(".png")
    fig.savefig(out, facecolor=SURFACE)
    return out


def fig8(path):
    d = json.loads(Path(path).read_text())
    ratios = np.asarray(d["ratios"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    style(ax)
    for i, r in enumerate(d["draws"]):
        ax.plot(ratios, r["avg_chain_length"], color=POSTERIOR, lw=1, alpha=0.35,
                label="posterior draws" if i == 0 else None)
    ax.plot(ratios, d["point_estimate"]["avg_chain_length"], color=POINT, lw=2, label="point estimate")
    ax.set_xscale("log")
    ax.set_xlabel("(FabF, FabB) : TesA, relative to baseline", fontsize=8, color=INK_2)
    ax.set_ylabel("average chain length", fontsize=8, color=INK_2)
    s = d["summary"]
    ax.set_title(f"A  Chain length rises with the ratio in {s['fraction_positive_slope']:.0%} of draws",
                 fontsize=9, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_2)
    ax = axes[1]
    style(ax)
    rng = np.random.default_rng(0)
    for row, key in enumerate(("longest", "shortest")):
        vals = [r[key]["ratio"] for r in d["draws"]]
        ax.scatter(vals, row + rng.uniform(-0.12, 0.12, len(vals)), s=14, color=POSTERIOR, alpha=0.7,
                   label="posterior draws" if row == 0 else None)
        ax.scatter([d["point_estimate"][key]["ratio"]], [row], s=60, marker="D", color=POINT,
                   edgecolor=SURFACE, linewidth=1.2, zorder=4, label="point estimate" if row == 0 else None)
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["longest chains", "shortest chains"], fontsize=8, color=INK_2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("optimal (FabF + FabB) : TesA, relative to baseline", fontsize=8, color=INK_2)
    ax.set_title("B  Enzyme setting at each extreme", fontsize=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="center right")
    fig.suptitle(f"Ratiometric strategy across {d['n_draws']} posterior draws ({d['system']})",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = Path(path).with_suffix(".png")
    fig.savefig(out, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    default = {"fig6": "identifiability.json", "fig9": "expected_information_grid.json", "fig7": "posterior_morris.json",
               "fig8": "posterior_ratio_response.json"}[which]
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / default
    print("wrote", {"fig6": fig6, "fig9": fig9, "fig7": fig7, "fig8": fig8}[which](src))
