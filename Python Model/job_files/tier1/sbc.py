"""Simulation-based calibration for R4 (Fig 3): a1 + c3 on C8, truths drawn from the prior.

SBC (Talts et al. 2018): if the sampler is calibrated, the rank of the true value among L
posterior draws is uniform on 0..L when the truth is drawn from the prior the fit uses. A
histogram that is U-shaped means intervals too narrow, a dome too wide, a slope a biased
posterior. The same replicates give nominal-vs-observed coverage of the central intervals.

  generate   draw each replicate's truth from the fit's prior (LogNormal, median 1, 95% in
             [0.1, 10], the exact sigma inference_runner._fit_prior uses), generate its data
             (make_tier1_rate_data.py --set, data seed = replicate index) and build its config
             (build_tier1_configs.py). Each step runs in its own process. A replicate whose data
             fail to generate is recorded, never silently redrawn: dropping extreme truths
             would bias the ranks.
  ranks      for every replicate whose run has finished, rank the truth among L posterior
             draws thinned evenly from the saved posterior (stranded chains are already
             excluded there), then report per-parameter rank histograms with a chi-square
             uniformity test, 50/90/95% coverage, and a draft Fig 3.
  selftest   the rank machinery on a conjugate Normal model, whose exact posterior is known,
             so the ranks must come out uniform before any real fit is trusted with them.

The pilot is 10 replicates; the plan continues to 40 only if the pilot looks sane.

Usage:
  python sbc.py generate [--start 0] [--n 10]
  python sbc.py ranks [--L 99] [--bins 10]
  python sbc.py selftest
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
FIGURES = PROJECT / "Results" / "Tier1" / "figures"   # figures live here; the rank data stays in job_files
MANIFEST = HERE / "sbc_manifest.json"
SYSTEM, PARAMS = "C8", ("a1", "c3")
PRIOR_SIGMA = math.log(10.0) / stats.norm.ppf(0.975)   # 1.1748099, as _fit_prior solves it
BASE_SEED = 20260925
LEVELS = (0.50, 0.90, 0.95)


def data_name(i):
    return f"Chain_{SYSTEM}_sbc{i:03d}"


def run_name(i):
    return f"Tier1 {SYSTEM}_sbc{i:03d} - {''.join(PARAMS)}"


def draw_truth(i):
    """Replicate i's truth: an independent draw from the LogNormal prior (its own stream)."""
    rng = np.random.default_rng([BASE_SEED, i])
    return {p: float(math.exp(rng.normal(0.0, PRIOR_SIGMA))) for p in PARAMS}


def load_manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"replicates": {}}


def generate(start, n):
    manifest = load_manifest()
    manifest.update({"system": SYSTEM, "params": list(PARAMS), "prior_sigma_log": PRIOR_SIGMA,
                     "base_seed": BASE_SEED})
    for i in range(start, start + n):
        truth = draw_truth(i)
        entry = {"truth": truth, "data_name": data_name(i), "run": run_name(i), "data_seed": i}
        gen = [sys.executable, "-u", str(HERE / "make_tier1_rate_data.py"), SYSTEM, "--seed", str(i),
               "--out_name", data_name(i)] + [a for p, v in truth.items() for a in ("--set", f"{p}={v!r}")]
        p = subprocess.run(gen, cwd=HERE, capture_output=True, text=True)
        if p.returncode != 0:
            entry["status"] = "data_failed"
            entry["error"] = (p.stderr or p.stdout)[-1500:]
        else:
            cfg = [sys.executable, str(HERE / "build_tier1_configs.py"), "--system", SYSTEM,
                   "--params", ",".join(PARAMS), "--data_name", data_name(i)]
            q = subprocess.run(cfg, cwd=HERE, capture_output=True, text=True)
            entry["status"] = "config_failed" if q.returncode != 0 else "ready"
            if q.returncode != 0:
                entry["error"] = (q.stderr or q.stdout)[-1500:]
        manifest["replicates"][str(i)] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"replicate {i:03d}: " + ", ".join(f"{k}={v:.4g}" for k, v in truth.items()) + f"  -> {entry['status']}",
              flush=True)


def rank_of(truth, draws, L):
    """Rank of truth among L draws thinned evenly from draws (all chains pooled)."""
    draws = np.asarray(draws).ravel()
    idx = np.linspace(0, len(draws) - 1, L).round().astype(int)
    return int(np.sum(draws[idx] < truth))


def covered(truth, draws, level):
    lo, hi = np.quantile(np.asarray(draws).ravel(), [(1 - level) / 2, (1 + level) / 2])
    return bool(lo <= truth <= hi)


def uniformity(ranks, L, bins):
    """Chi-square test of rank counts in `bins` equal bins of 0..L against uniform."""
    counts, _ = np.histogram(ranks, bins=bins, range=(-0.5, L + 0.5))
    expected = len(ranks) / bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return counts.tolist(), chi2, float(stats.chi2.sf(chi2, bins - 1))


def summarize(rank_rows, cover_rows, L, bins):
    out = {}
    for p in PARAMS:
        r = [row[p] for row in rank_rows]
        counts, chi2, pval = uniformity(r, L, bins)
        cov = {f"{int(lv * 100)}%": float(np.mean([c[p][lv] for c in cover_rows])) for lv in LEVELS}
        out[p] = {"ranks": r, "bin_counts": counts, "chi2": round(chi2, 3), "p_uniform": round(pval, 4),
                  "coverage": cov}
    return out


PARAM_COLOR = {"a1": "tab:blue", "c3": "tab:green"}   # as tier1_result_figures.py


def plot(summary, n, L, bins, path, excluded=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(PROJECT / "Utilities"))
    from inference_plotting import THRESHOLD_STYLE, _apply_plot_style, place_suptitle
    _apply_plot_style()
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(7.5 * len(PARAMS), 5.8), sharey=True, squeeze=False)
    lo, hi = stats.binom.ppf([0.005, 0.995], n, 1 / bins)   # 99% band for one bin's count
    for ax, p in zip(axes[0], PARAMS):
        s = summary[p]
        ax.axhspan(lo, hi, color="0.94", zorder=0, lw=0)
        for edge in (lo, hi):
            ax.axhline(edge, zorder=1, **THRESHOLD_STYLE)
        ax.axhline(n / bins, color="0.35", lw=1.2, zorder=1)
        ax.bar(np.arange(bins), s["bin_counts"], width=0.86, color=PARAM_COLOR.get(p, "tab:blue"), zorder=2)
        ax.set_title(f"{p}: Chi-Square p = {s['p_uniform']:.2f}")
        ax.set_xticks([0, bins - 1], ["Low", "High"])
        ax.set_xlabel(f"Rank of Truth Among {L} Draws ({bins} Bins)")
    axes[0][0].set_ylabel(f"Replicates (of {n})")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend([Line2D([], [], **THRESHOLD_STYLE), Line2D([], [], color="0.35", lw=1.2)],
               ["99% Range for a Uniform Bin", "Expected Count"], loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=2)
    title = f"Tier1 C8 SBC ({n} Replicates) — Rank Histograms"
    if excluded:
        title += " (Excluded: " + ", ".join(f"sbc{int(i):03d}" for i in excluded) + ")"
    place_suptitle(fig, title)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


EXCLUDE_FILE = HERE / "sbc_exclude.json"   # {replicate index: reason}, left out of the ranks until resolved


def ranks(L, bins):
    import arviz as az
    manifest = load_manifest()
    excluded = json.loads(EXCLUDE_FILE.read_text()) if EXCLUDE_FILE.exists() else {}
    rank_rows, cover_rows, used, pending = [], [], [], []
    for i, e in sorted(manifest["replicates"].items(), key=lambda kv: int(kv[0])):
        post = PROJECT / "Results" / "Tier1" / e["run"] / "posterior_samples_pm.nc"
        if str(int(i)) in excluded:
            continue
        if e.get("status") != "ready" or not post.exists():
            pending.append(int(i))
            continue
        posterior = az.from_netcdf(post).posterior
        rank_rows.append({p: rank_of(e["truth"][p], posterior[p].values, L) for p in PARAMS})
        cover_rows.append({p: {lv: covered(e["truth"][p], posterior[p].values, lv) for lv in LEVELS} for p in PARAMS})
        used.append(int(i))
    if not used:
        print(f"no finished SBC runs yet ({len(pending)} pending)")
        return
    summary = summarize(rank_rows, cover_rows, L, bins)
    out = {"L": L, "bins": bins, "replicates_used": used, "pending_or_failed": pending,
           "excluded": excluded, "params": summary}
    (HERE / "sbc_ranks.json").write_text(json.dumps(out, indent=1) + "\n")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = plot(summary, len(used), L, bins, FIGURES / "sbc_ranks.png", sorted(excluded, key=int))
    for p in PARAMS:
        s = summary[p]
        print(f"{p}: chi-square p = {s['p_uniform']:.3f}; coverage " + ", ".join(f"{k} {v:.2f}" for k, v in s["coverage"].items()))
    print(f"{len(used)} replicates used, {len(pending)} pending/failed, {len(excluded)} excluded "
          f"({', '.join(f'sbc{int(i):03d}' for i in excluded)}); wrote sbc_ranks.json, {fig.name}")


def selftest(n=400, L=99, bins=10, seed=1):
    """Conjugate Normal: prior N(0, 1), 5 observations with sd 1. Exact posterior, so ranks are uniform."""
    rng = np.random.default_rng(seed)
    rank_rows, cover_rows = [], []
    for _ in range(n):
        truth = {p: rng.normal() for p in PARAMS}
        row, cov = {}, {}
        for p in PARAMS:
            y = rng.normal(truth[p], 1.0, size=5)
            post_var = 1.0 / (1.0 + 5.0)
            draws = rng.normal(post_var * y.sum(), math.sqrt(post_var), size=1000)
            row[p] = rank_of(truth[p], draws, L)
            cov[p] = {lv: covered(truth[p], draws, lv) for lv in LEVELS}
        rank_rows.append(row)
        cover_rows.append(cov)
    s = summarize(rank_rows, cover_rows, L, bins)
    ok = all(s[p]["p_uniform"] > 0.001 for p in PARAMS) and all(
        abs(s[p]["coverage"][f"{int(lv * 100)}%"] - lv) < 0.06 for p in PARAMS for lv in LEVELS)
    for p in PARAMS:
        print(f"selftest {p}: chi-square p = {s[p]['p_uniform']:.3f}; coverage " +
              ", ".join(f"{k} {v:.3f}" for k, v in s[p]["coverage"].items()))
    print("PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--n", type=int, default=10)
    r = sub.add_parser("ranks")
    r.add_argument("--L", type=int, default=99)
    r.add_argument("--bins", type=int, default=10)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.start, a.n)
    elif a.cmd == "ranks":
        ranks(a.L, a.bins)
    else:
        sys.exit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
