"""Mock of the paper's Figure 1 for the draft layout: (A) the FAS network, (B) the previous
point-estimate workflow against the Bayesian one, (C) the two validation tiers.

Panel A is an image passed in (for the draft, Ruppe et al. 2020 PNAS Fig 1A, cropped); the
final figure needs a redrawn network or the publisher's permission.

    python paper_fig1_mock.py --network pnas2020_fig1.jpg --crop 0,0,545,578 --out fig1_mock.png
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from inference_plotting import PLOT_FONT_SIZE, _apply_plot_style, place_suptitle  # noqa: E402

OLD, NEW, NEUTRAL = "#8c8c8c", "tab:blue", "0.25"


def box(ax, x, y, w, h, text, color, fill=0.12, size=None, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.8, edgecolor=color, facecolor=matplotlib.colors.to_rgba(color, fill)))
    ax.text(x, y, text, ha="center", va="center", fontsize=size or PLOT_FONT_SIZE - 3,
            fontweight="bold" if bold else "normal", color="0.1", wrap=True)


def arrow(ax, x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, linewidth=1.8,
                                 color=color, shrinkA=2, shrinkB=2))


def workflow(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # previous workflow (top row)
    ax.text(0.0, 0.97, "Previous (point estimate)", fontsize=PLOT_FONT_SIZE - 1, fontweight="bold", color=OLD, va="top")
    y = 0.80
    box(ax, 0.10, y, 0.17, 0.12, "Kinetic data", OLD)
    box(ax, 0.32, y, 0.19, 0.12, "Least-squares\nfit", OLD)
    box(ax, 0.54, y, 0.17, 0.12, "One parameter\nset", OLD)
    for x0, x1 in ((0.185, 0.225), (0.415, 0.455)):
        arrow(ax, x0, y, x1, y, OLD)
    for i, t in enumerate(["Morris sensitivity", "Enzyme-ratio design", "Knockout predictions"]):
        yy = 0.89 - i * 0.09
        box(ax, 0.84, yy, 0.27, 0.058, t, OLD, size=PLOT_FONT_SIZE - 4)
        arrow(ax, 0.625, y, 0.705, yy, OLD)
    # Bayesian workflow (bottom row)
    ax.text(0.0, 0.56, "This paper (Bayesian)", fontsize=PLOT_FONT_SIZE - 1, fontweight="bold", color=NEW, va="top")
    y = 0.30
    box(ax, 0.10, y + 0.09, 0.17, 0.10, "Priors", NEW)
    box(ax, 0.10, y - 0.09, 0.17, 0.10, "Kinetic data", NEW)
    box(ax, 0.32, y, 0.19, 0.12, "NUTS sampling\n(JAX ODEs)", NEW)
    box(ax, 0.54, y, 0.17, 0.12, "Posterior\ndistribution", NEW, fill=0.25, bold=True)
    arrow(ax, 0.185, y + 0.09, 0.225, y + 0.02, NEW)
    arrow(ax, 0.185, y - 0.09, 0.225, y - 0.02, NEW)
    arrow(ax, 0.415, y, 0.455, y, NEW)
    outs = ["Trust: uncertainty (Figs 2-4)", "Groupings tested (Fig 5)", "What data constrain (Fig 6)",
            "Sensitivity with uncertainty (Fig 7)", "Robust ratio design (Fig 8)", "What to measure next (Fig 9)"]
    for i, t in enumerate(outs):
        yy = 0.52 - i * 0.088
        box(ax, 0.84, yy, 0.30, 0.056, t, NEW, size=PLOT_FONT_SIZE - 4)
        arrow(ax, 0.625, y, 0.69, yy, NEW)


def tiers(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.25, 0.5, 0.44, 0.78,
        "Tier 1: synthetic data, known truth\n\nTruncated networks (C4 to C20+unsat)\nhours per fit\n\n"
        "Does the method recover the truth,\nwith honest uncertainty?", NEUTRAL, fill=0.06)
    box(ax, 0.75, 0.5, 0.44, 0.78,
        "Tier 2: real ME1 data\n\nFull model (194 constants,\n318 species)\n\n"
        "Do the Bayesian conclusions\ndiffer from the point estimate?", NEW, fill=0.08)
    arrow(ax, 0.475, 0.5, 0.525, 0.5, NEUTRAL)


def pictures(fig, gs, a):
    """Panel B drawn from a real fit: before data (prior, what it allows, the data) and after
    (posterior, what it predicts, what it is used for)."""
    import numpy as np
    import arviz as az
    from scipy.stats import gaussian_kde
    PRIOR, POST, DATA = "0.55", "tab:blue", "tab:orange"
    pr = np.load(a.prior_pred)
    fit = np.load(a.fit)["pred"]
    t = pr["times"] / 60.0
    obs, sig = pr["obs"][:10], pr["sigma"][:10]
    truth = np.load(a.fit)["truth"][:10]
    post = az.from_netcdf(a.posterior)
    pa1, pc3 = (np.asarray(post["posterior"][v]).ravel() for v in ("a1", "c3"))
    qa1, qc3 = (np.asarray(post["prior"][v]).ravel() for v in ("a1", "c3"))

    ax = fig.add_subplot(gs[0, 1])
    lx = np.linspace(np.log(0.01), np.log(100), 400)
    sd = np.log(100) / 1.959963984540054 / 2
    ax.fill_between(np.exp(lx), np.exp(-0.5 * (lx / sd) ** 2), color=PRIOR, alpha=0.35, linewidth=0)
    ax.plot(np.exp(lx), np.exp(-0.5 * (lx / sd) ** 2), color=PRIOR, linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_yticks([])
    ax.set_xlabel("Rate-Constant Scale Factor")
    ax.set_title("Prior: 0.1-10x Plausible", fontsize=PLOT_FONT_SIZE - 2)

    ax = fig.add_subplot(gs[0, 2])
    for row in pr["pred"]:
        ax.plot(t, row[:10], color=PRIOR, alpha=0.35, linewidth=1)
    ax.plot(t, truth, color="0.15", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("C16 Equiv. (µM)")
    ax.set_title("What the Prior Allows", fontsize=PLOT_FONT_SIZE - 2)

    ax = fig.add_subplot(gs[0, 3])
    ax.errorbar(t, obs, yerr=sig, fmt="o", color=DATA, capsize=3, markersize=6)
    ax.plot(t, truth, color="0.15", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Time (min)")
    ax.set_title("Data (Noisy Measurements)", fontsize=PLOT_FONT_SIZE - 2)

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(qa1, qc3, s=4, color=PRIOR, alpha=0.25, linewidths=0)
    ax.scatter(pa1, pc3, s=3, color=POST, alpha=0.4, linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.03, 30); ax.set_ylim(0.03, 30)
    ax.set_xlabel("a1 (Scale Factor)")
    ax.set_ylabel("c3 (Scale Factor)")
    ax.set_title("Posterior: Data Pin the Values", fontsize=PLOT_FONT_SIZE - 2)

    ax = fig.add_subplot(gs[1, 2])
    lo, med, hi = np.percentile(fit[:, :10], [2.5, 50, 97.5], axis=0)
    ax.fill_between(t, lo, hi, color=POST, alpha=0.3, linewidth=0)
    ax.plot(t, med, color=POST, linewidth=2)
    ax.errorbar(t, obs, yerr=sig, fmt="o", color=DATA, capsize=3, markersize=5)
    ax.plot(t, truth, color="0.15", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("C16 Equiv. (µM)")
    ax.set_title("What the Posterior Predicts", fontsize=PLOT_FONT_SIZE - 2)

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    uses = ["Uncertainty on every\nparameter and prediction", "Tests of grouping\nassumptions",
            "What the data do and\ndo not constrain", "Sensitivity and design\nwith uncertainty",
            "Which experiment\nto run next"]
    for i, u in enumerate(uses):
        box(ax, 0.5, 0.9 - i * 0.2, 0.95, 0.16, u, POST, size=PLOT_FONT_SIZE - 4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("What It Is Used For", fontsize=PLOT_FONT_SIZE - 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="image for panel A")
    ap.add_argument("--crop", default=None, help="left,top,right,bottom in pixels")
    ap.add_argument("--blank", default=None, help="left,top,right,bottom to paint white (e.g. the source's own panel letter)")
    ap.add_argument("--credit", default="Network: Ruppe et al. 2020, PNAS 117:23557, Fig 1A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", choices=["boxes", "pictures"], default="boxes")
    ap.add_argument("--prior_pred", help="pictures: npz of model predictions at prior draws (pred, times, obs, sigma)")
    ap.add_argument("--fit", help="pictures: npz of predictions at posterior draws (pred, truth)")
    ap.add_argument("--posterior", help="pictures: the run's posterior_samples_pm.nc (posterior + prior draws)")
    a = ap.parse_args()

    from PIL import Image
    img = Image.open(a.network)
    if a.blank:
        from PIL import ImageDraw
        img = img.convert("RGB")
        ImageDraw.Draw(img).rectangle(tuple(int(v) for v in a.blank.split(",")), fill="white")
    if a.crop:
        img = img.crop(tuple(int(v) for v in a.crop.split(",")))

    _apply_plot_style()
    if a.style == "pictures":
        fig = plt.figure(figsize=(22.0, 10.0))
        gs = fig.add_gridspec(2, 4, width_ratios=[1.45, 1, 1, 1], wspace=0.32, hspace=0.62, top=0.84)
        ax_a = fig.add_subplot(gs[:, 0])
        ax_a.imshow(img)
        ax_a.axis("off")
        ax_a.set_anchor("N")
        ax_a.text(0.0, -0.02, a.credit, transform=ax_a.transAxes, ha="left", va="top", fontsize=PLOT_FONT_SIZE - 5,
                  color="0.35")
        pictures(fig, gs, a)
        # place the row labels, panel heading and inference arrow from the panels' own positions
        axes = {(r, c): fig.axes[1 + 3 * r + (c - 1)] for r in (0, 1) for c in (1, 2, 3)}
        top, bot = axes[(0, 1)].get_position(), axes[(1, 1)].get_position()
        mid_col = axes[(0, 2)].get_position()
        left = top.x0 - 0.045
        for pos, lab in ((top, "Before Data"), (bot, "After Data")):
            fig.text(left, (pos.y0 + pos.y1) / 2, lab, rotation=90, ha="center", va="center",
                     fontsize=PLOT_FONT_SIZE, fontweight="bold", color="0.3")
        head_y = top.y1 + 0.075
        fig.text(ax_a.get_position().x0, head_y, "A  FAS Network", fontsize=PLOT_FONT_SIZE, fontweight="bold",
                 ha="left", va="bottom")
        fig.text(left - 0.012, head_y, "B  Bayesian Workflow (C8 Fit; Dashed = Known Truth)",
                 fontsize=PLOT_FONT_SIZE, fontweight="bold", ha="left", va="bottom")
        y_hi, y_lo = top.y0 - 0.075, bot.y1 + 0.045
        x = (mid_col.x0 + mid_col.x1) / 2 + 0.07
        fig.patches.append(FancyArrowPatch((x, y_hi), (x, y_lo), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=28, linewidth=3, color=NEW))
        fig.text(x + 0.012, (y_hi + y_lo) / 2, "Bayesian inference\n(NUTS sampling of the ODE model)", ha="left",
                 va="center", fontsize=PLOT_FONT_SIZE - 2, color=NEW)
        # set directly: place_suptitle sits just above the panels, under the two headings
        fig.suptitle("Figure 1 (Mock) — FAS Network and the Bayesian Workflow", y=head_y + 0.075,
                     fontsize=PLOT_FONT_SIZE + 1)
        fig.savefig(a.out, dpi=250, bbox_inches="tight")
        print(f"Saved: {a.out}")
        return
    fig = plt.figure(figsize=(18.0, 11.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[2.2, 1.0], wspace=0.05, hspace=0.12)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_a.imshow(img)
    ax_a.axis("off")
    ax_a.set_anchor("N")
    ax_a.set_title("A  FAS Network", loc="left", fontweight="bold")
    ax_a.text(0.0, -0.02, a.credit, transform=ax_a.transAxes, ha="left", va="top", fontsize=PLOT_FONT_SIZE - 5,
              color="0.35")
    ax_b = fig.add_subplot(gs[0, 1])
    workflow(ax_b)
    ax_b.set_title("B  Workflow: Point Estimate vs Posterior", loc="left", fontweight="bold")
    ax_c = fig.add_subplot(gs[1, 1])
    tiers(ax_c)
    ax_c.set_title("C  Validation in Two Tiers", loc="left", fontweight="bold")
    place_suptitle(fig, "Figure 1 (Mock) — FAS Network and the Bayesian Workflow")
    fig.savefig(a.out, dpi=250, bbox_inches="tight")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
