"""Posterior-integrated enzyme sensitivity for Fig 7 (outline 3.4).

The original target identification (Ruppe et al. 2020, Fig 3E) ran a Morris screen over the
nine enzyme concentrations at the single fitted point. This repeats that screen at each of n
posterior draws of the fitted scaling parameters, and at the point estimate (every scaling
group at its no-op value), to show whether parameter uncertainty changes which enzymes the
model says to engineer.

The Morris method is morris_screen.py's, imported rather than re-implemented: a radial design
with Latin-hypercube base and auxiliary points, uniform in log10 over [1/span, span] times
each enzyme's baseline concentration, elementary effect = change in objective / change in
log10 concentration, and the same three objectives (total production, average chain length
and, where the system has an unsaturated branch, unsaturated fraction), read from the fatty
acids at 720 s. Every parameter set uses the same design, so differences between draws come
from the parameters, not from design noise. Solves use the system's working tolerance, as
morris_screen.py does.

Reported per objective and enzyme: mu* at the point estimate; the posterior median and 5-95%
range of mu*; how often the enzyme holds its point-estimate rank; and, per draw, the Spearman
correlation between its mu* ranking and the point estimate's.

Usage:
  python posterior_morris.py --posterior "Results/Tier1/<run>/posterior_samples_pm.nc" --params a1,c3,a2
  python posterior_morris.py --posterior job_files/multiparam_tests/mp_posterior.json \\
      --key a1c3_no_floor/C14 --system C14 --params a1,c3 --draws 3 --r 4   # quick test
"""
import argparse
import json
import re
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy import stats

from forward_model import ENZYMES, PROJECT, ForwardModel, posterior_draws
from make_tier1_rate_data import tolerances

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("morris_screen", HERE.parent / "chain_system_sensitivity_analysis" / "morris_screen.py")
morris = module_from_spec(_spec)
_spec.loader.exec_module(morris)


def design(r, k, span, seed):
    """morris_screen.py's radial design, in log10 multipliers: (base, aux), each (r, k)."""
    rng = np.random.default_rng(seed)
    L = np.log10(span)
    return (morris.lhs(r, k, rng) * 2 - 1) * L, (morris.lhs(r, k, rng) * 2 - 1) * L


def y0_rows(fm, base, aux):
    """Initial conditions for every design point: per trajectory, its base, then each enzyme moved."""
    y_base = fm.y0()
    enz_idx = [fm.index[e] for e in ENZYMES]
    rows = []
    for b, x in zip(base, aux):
        for j in range(-1, len(ENZYMES)):
            m = b.copy()
            if j >= 0:
                m[j] = x[j]
            y = y_base.copy()
            y[enz_idx] = y_base[enz_idx] * 10.0 ** m
            rows.append(y)
    return np.stack(rows)


def elementary_effects(fm, theta, rows, base, aux, objectives, n_obj):
    """(r, k, n_obj) elementary effects; NaN where a solve failed."""
    ys, ok = fm.run(rows, theta)
    k = len(ENZYMES)
    f = np.array([objectives(y[-1]) if good else np.full(n_obj, np.nan) for y, good in zip(ys, ok)])
    f = f.reshape(len(base), k + 1, n_obj)
    delta = (aux - base)[:, :, None]
    return (f[:, 1:, :] - f[:, :1, :]) / delta


def mu_star(ee):
    return np.nanmean(np.abs(ee), axis=0)          # (k, n_obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posterior", required=True, help="posterior_samples_pm.nc, or an export_posterior_series.py JSON")
    ap.add_argument("--key", default=None, help="for a JSON export: '<set>/<system>'")
    ap.add_argument("--system", default="C14+unsat")
    ap.add_argument("--params", default="a1,c3,a2")
    ap.add_argument("--draws", type=int, default=50, help="posterior draws, thinned evenly")
    ap.add_argument("--r", type=int, default=20, help="radial base points per parameter set")
    ap.add_argument("--span", type=float, default=10.0, help="enzyme range is [1/span, span] x baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "posterior_morris.json"))
    a = ap.parse_args()
    params = a.params.split(",")
    src = Path(a.posterior) if Path(a.posterior).is_absolute() else PROJECT / a.posterior

    t0 = time.perf_counter()
    rtol, atol = tolerances(a.system)
    fm = ForwardModel(a.system, times=[720.0], rtol=rtol, atol=atol)
    targets = [s for s in fm.species if re.fullmatch(r"C\d+_FA(_unsat)?", s)]
    objectives, names = morris.make_objectives(fm._sys, targets, 1e-9)
    base, aux = design(a.r, len(ENZYMES), a.span, a.seed)
    rows = y0_rows(fm, base, aux)

    point = mu_star(elementary_effects(fm, fm.theta(), rows, base, aux, objectives, len(names)))
    draws = posterior_draws(src, params, a.draws, key=a.key)
    mus = []
    for i, d in enumerate(draws):
        mus.append(mu_star(elementary_effects(fm, fm.theta(d), rows, base, aux, objectives, len(names))))
        print(f"  draw {i + 1}/{len(draws)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    mus = np.stack(mus)                              # (draws, k, n_obj)

    out = {"system": a.system, "posterior": str(a.posterior), "key": a.key, "params": params,
           "n_draws": len(draws), "r": a.r, "span": a.span, "seed": a.seed, "tolerance": [rtol, atol],
           "enzymes": list(ENZYMES), "objectives": {}}
    for o, name in enumerate(names):
        pt = point[:, o]
        pt_rank = stats.rankdata(-pt, method="min")
        draw_ranks = np.array([stats.rankdata(-m[:, o], method="min") for m in mus])
        rho = [float(stats.spearmanr(pt, m[:, o]).statistic) for m in mus]
        rows_out = []
        for j, e in enumerate(ENZYMES):
            q = np.nanpercentile(mus[:, j, o], [5, 50, 95])
            rows_out.append({"enzyme": e, "mu_star_point": float(pt[j]), "rank_point": int(pt_rank[j]),
                             "mu_star_posterior_median": float(q[1]), "mu_star_posterior_5_95": [float(q[0]), float(q[2])],
                             "p_same_rank": float(np.mean(draw_ranks[:, j] == pt_rank[j]))})
        rows_out.sort(key=lambda row: row["rank_point"])
        out["objectives"][name] = {"enzymes": rows_out, "spearman_vs_point_median": float(np.nanmedian(rho)),
                                   "spearman_vs_point_min": float(np.nanmin(rho))}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    for name, v in out["objectives"].items():
        print(f"\n{name}: Spearman vs point estimate, median {v['spearman_vs_point_median']:.3f} "
              f"(min {v['spearman_vs_point_min']:.3f})")
        for row in v["enzymes"]:
            lo, hi = row["mu_star_posterior_5_95"]
            print(f"  {row['rank_point']}. {row['enzyme']:5s} point {row['mu_star_point']:10.4g}   "
                  f"posterior {row['mu_star_posterior_median']:10.4g} [{lo:.4g}, {hi:.4g}]   same rank {row['p_same_rank']:.2f}")
    print(f"\nwrote {a.out} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
