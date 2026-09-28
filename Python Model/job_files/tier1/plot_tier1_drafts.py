"""Figures from the Tier-1 post-processing JSON: Fig 6 (identifiability) and Fig 9 (expected
information), in the same style as tier1_result_figures.py.

  python plot_tier1_drafts.py fig6 [identifiability.json]
  python plot_tier1_drafts.py fig9 [expected_information_grid.json]

Each writes a PNG next to its JSON. Figs 7 and 8 are tier1_result_figures.py fig7 / fig8; Fig 3's
rank histogram comes from sbc.py ranks.
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
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))
from inference_plotting import PLOT_FONT_SIZE, THRESHOLD_STYLE, _apply_plot_style, place_suptitle  # noqa: E402

POSTERIOR = "tab:blue"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
# Diverging: red arm, gray midpoint, blue arm, equal steps per side (for signed values in [-1, 1]).
DIV = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#8f2424", "#e34948", "#f4b3b0", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
CELL_FONT = PLOT_FONT_SIZE - 2


def title_case(label):
    """'total fatty acid' -> 'Total Fatty Acid'; acronyms, numbers and units ('150 s') stay as they are."""
    return " ".join(w[0].upper() + w[1:] if w.isalpha() and len(w) > 1 else w for w in label.split(" "))


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
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=CELL_FONT,
                    color="white" if dark else "black")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def _save(fig, out):
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig6(path):
    d = json.loads(Path(path).read_text())
    names = d["params"]
    run = Path(d["posterior"]).parent.name + (f", {d['key']}" if d.get("key") else "")
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ax = axes[0]
    y = np.arange(len(names))[::-1]
    contraction = [d["contraction"][p] for p in names]
    ax.barh(y, contraction, height=0.5, color=POSTERIOR)
    for yi, v in zip(y, contraction):
        ax.text(max(v, 0) + 0.02, yi, f"{v:.3f}", va="center", fontsize=CELL_FONT)
    threshold = ax.axvline(0.5, **THRESHOLD_STYLE)
    ax.set_yticks(y, [f"{p} ({d['scale'][p]})" for p in names])
    ax.set_xlim(min(0, min(contraction)) - 0.02, 1.25)
    ax.set_xlabel("1 − Posterior Variance / Prior Variance")
    ax.set_title("Posterior Contraction")
    heatmap(axes[1], np.array(d["correlation"]), names, names, "{:+.2f}", "Posterior Correlation", signed=True)
    eig = d["eigen"]
    loadings = np.array([[e["direction"][p] for p in names] for e in eig])
    heatmap(axes[2], loadings, [f"{e['variance_left']:.2g}" for e in eig], names, "{:+.2f}",
            "Covariance Directions, Tightest First", signed=True)
    axes[2].set_ylabel("Fraction of Prior Variance Left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    legend = fig.legend([threshold], ["Weakly Identified Below 0.5"], loc="upper center", bbox_to_anchor=(0.5, 0.09))
    fig.text(0.5, 0.0, f"{d['n_draws']} posterior draws; log scale for LogNormal parameters. Directions: eigenvectors "
             "of the posterior covariance with each parameter scaled by its prior sd; each row's squares sum to 1.",
             ha="center", va="top", fontsize=PLOT_FONT_SIZE - 4, color="0.35")
    place_suptitle(fig, f"{run} — Identifiability")
    return _save(fig, Path(path).with_suffix(".png"))


def fig9(path):
    d = json.loads(Path(path).read_text())
    cells = list(d["grid"].values())
    rows = list(dict.fromkeys(c["measurement"] for c in cells))
    cols = list(dict.fromkeys(c["timing"] for c in cells))
    get = lambda key: np.array([[next(c[key] for c in cells if c["measurement"] == r and c["timing"] == t)
                                 for t in cols] for r in rows], dtype=float)
    labels_r, labels_c = [title_case(r) for r in rows], [title_case(c) for c in cols]
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.6))
    heatmap(axes[0], get("info_nats_per_point"), labels_r, labels_c, "{:.3f}", "Expected Information per Point (Nats)")
    heatmap(axes[1], get("mean_contraction"), labels_r, labels_c, "{:.4f}", "Mean Posterior Contraction (Log Scale)")
    axes[1].set_yticklabels([])
    t1 = d["tier1_design"]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.02, f"Five Tier-1 conditions. The Tier-1 design for reference: {t1['info_nats_per_point']:.3f} nats "
             f"per point over {t1['n_points']} points, mean contraction {t1['mean_contraction']:.4f}.",
             ha="center", va="top", fontsize=PLOT_FONT_SIZE - 4, color="0.35")
    place_suptitle(fig, f"{d['system']} ({' + '.join(d['params'])}) — Expected Information by Measurement")
    return _save(fig, Path(path).with_suffix(".png"))


if __name__ == "__main__":
    which = sys.argv[1]
    default = {"fig6": "identifiability.json", "fig9": "expected_information_grid.json"}[which]
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / default
    print("wrote", {"fig6": fig6, "fig9": fig9}[which](src))
