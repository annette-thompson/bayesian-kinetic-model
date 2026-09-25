"""Expected information for every cell of Fig 9's data-type grid, with no sampling.

Outline 3.6 asks which measurements most efficiently tighten the posterior, crossed two ways:

  timing       endpoint (720 s) only | initial (150 s) only | both
  measurement  total fatty acid (C16 equivalents) | individual fatty-acid species |
               individual species + ACP intermediates (ketoacyl-, hydroxyacyl-, enoyl- and acyl-ACP;
               whichever FA_acylACP_conc.py's MEASURED_FORMS exposes)

each cell measured under the five Tier-1 conditions (baseline; FabH 0.1 uM; FabB 0; TesA
0.5 uM; FabZ 0), with the Tier-1 noise model (sd = 10% of the value + 0.01 uM).

For each cell this computes the Laplace approximation of the posterior on the main-fit model
(a1 + c3 + a2 on C14+unsat, R2) at the truth. With J the sensitivity of every data point to
log(parameter) and W the inverse noise variances, the Fisher information is F = J^T W J. The
prior on each log-parameter is Normal with sd 1.1748 (95% of the LogNormal in [0.1, 10]), so
the posterior covariance is (F + I / 1.1748^2)^-1. Reported per cell:

  shrinkage_log   1 - posterior sd / prior sd per parameter, on the log scale (recovery_report's
                  shrinkage_log, so directly comparable with a sampled run)
  info_nats       expected information gain, 0.5 * log(det prior cov / det posterior cov)
  per_point       info_nats and mean shrinkage divided by the number of data points: the
                  figure's ranking, value per unit of collection cost

The actual Tier-1 design (time series, baseline profile, five initial rates) is scored the same
way as a reference row, since R1 is its sampled check and R7 samples three grid cells against it.

J comes from central finite differences in log(parameter) at tight tolerance (rtol 1e-8,
atol 1e-10), from one compiled solve. It is computed at two step sizes, and their agreement is
reported.

Usage: python expected_information_grid.py [--system C14+unsat] [--params a1,c3,a2]
                                           [--out expected_information_grid.json]
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from forward_model import CONDITIONS, FLOOR_CONC, NOISE_FRAC, ForwardModel

HERE = Path(__file__).resolve().parent
PRIOR_SD_LOG = math.log(10.0) / 1.959964      # LogNormal, 95% in [0.1, 10], median 1
RATE_TIME, END_TIME = 150.0, 720.0
SERIES_TIMES = [72.0 * k for k in range(1, 11)]  # the Tier-1 time series, 72 ... 720 s
FLOOR_RATE = 0.004                              # uM C16/min, the Tier-1 rate floor
TOTAL = "C16 Equivalents (uM)"
RATE = "Initial Rate (uM C16 Equivalents/min)"
TIMINGS = {"endpoint (720 s)": [END_TIME], "initial (150 s)": [RATE_TIME], "both": [RATE_TIME, END_TIME]}


def measurement_sets(fm):
    species = fm.names(r"C\d+_FA(_unsat)? \(uM\)")
    intermediates = fm.names(r"C\d+_[A-Za-z]+ACP(_unsat)? \(uM\)")   # whatever the module exposes
    return {"total fatty acid": [TOTAL], "FA species": species,
            "FA species + ACP intermediates": species + intermediates}


def simulate(fm, theta, names):
    y0 = np.stack([fm.y0(changes) for _, changes in CONDITIONS])
    ys, ok = fm.run(y0, theta)
    if not ok.all():
        raise RuntimeError(f"solve failed for conditions {[c for (c, _), o in zip(CONDITIONS, ok) if not o]}")
    return fm.observe(ys, names)                       # name -> (conditions, times)


def jacobians(fm, params, names, h):
    """Central differences of every observable w.r.t. log(parameter): name -> (cond, times, params)."""
    cols = {name: [] for name in names}
    for p in params:
        up = simulate(fm, fm.theta({p: math.exp(h)}), names)
        dn = simulate(fm, fm.theta({p: math.exp(-h)}), names)
        for name in names:
            cols[name].append((up[name] - dn[name]) / (2 * h))
    return {name: np.stack(c, axis=-1) for name, c in cols.items()}


def score(J, values, floors):
    """Laplace posterior on log scale from stacked sensitivities J (points, params)."""
    sd = NOISE_FRAC * np.abs(values) + floors
    F = J.T @ (J / sd[:, None] ** 2)
    k = J.shape[1]
    post = np.linalg.inv(F + np.eye(k) / PRIOR_SD_LOG ** 2)
    post_sd = np.sqrt(np.diag(post))
    info = 0.5 * (k * math.log(PRIOR_SD_LOG ** 2) - np.linalg.slogdet(post)[1])
    return post_sd, info


def cell_points(fm, base, J, names, times):
    """Stack a cell's data points: every condition x time x observable."""
    t_idx = [int(np.argmin(np.abs(np.asarray(fm.times) - t))) for t in times]
    rows_J, rows_v = [], []
    for name in names:
        rows_J.append(J[name][:, t_idx, :].reshape(-1, J[name].shape[-1]))
        rows_v.append(base[name][:, t_idx].reshape(-1))
    return np.concatenate(rows_J), np.concatenate(rows_v)


def summarize(params, post_sd, info, n):
    shrink = 1.0 - post_sd / PRIOR_SD_LOG
    return {"n_points": int(n),
            "shrinkage_log": {p: round(float(s), 5) for p, s in zip(params, shrink)},
            "mean_shrinkage_log": round(float(shrink.mean()), 5),
            "info_nats": round(float(info), 4),
            "info_nats_per_point": round(float(info / n), 5),
            "mean_shrinkage_per_point": round(float(shrink.mean() / n), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="C14+unsat")
    ap.add_argument("--params", default="a1,c3,a2")
    ap.add_argument("--h", type=float, default=1e-3, help="finite-difference step in log(parameter)")
    ap.add_argument("--out", default=str(HERE / "expected_information_grid.json"))
    a = ap.parse_args()
    params = a.params.split(",")

    t0 = time.perf_counter()
    times = sorted(set(SERIES_TIMES) | {RATE_TIME, END_TIME})
    fm = ForwardModel(a.system, times=times)
    sets = measurement_sets(fm)
    names = sorted({n for s in sets.values() for n in s} | {RATE})
    base = simulate(fm, fm.theta(), names)
    J = jacobians(fm, params, names, a.h)
    J_check = jacobians(fm, params, names, 3 * a.h)
    # Largest h-vs-3h difference, relative to each observable's largest sensitivity.
    fd_agreement = max(float(np.max(np.abs(J[n] - J_check[n])) / np.max(np.abs(J[n])))
                       for n in names if np.max(np.abs(J[n])) > 0)

    grid = {}
    for mname, mnames in sets.items():
        for tname, ttimes in TIMINGS.items():
            Jc, vc = cell_points(fm, base, J, mnames, ttimes)
            post_sd, info = score(Jc, vc, FLOOR_CONC)
            grid[f"{mname} | {tname}"] = {"measurement": mname, "timing": tname,
                                          **summarize(params, post_sd, info, len(vc))}

    # The Tier-1 design itself: baseline time series + baseline profile + five initial rates.
    bl = 0                                                    # CONDITIONS[0] is the baseline
    t = np.asarray(fm.times)
    si = [int(np.argmin(np.abs(t - s))) for s in SERIES_TIMES]
    ei, ri = int(np.argmin(np.abs(t - END_TIME))), int(np.argmin(np.abs(t - RATE_TIME)))
    parts_J = [J[TOTAL][bl, si, :]] + [J[n][bl, ei, :][None, :] for n in sets["FA species"]] + [J[RATE][:, ri, :]]
    parts_v = [base[TOTAL][bl, si]] + [base[n][bl, ei][None] for n in sets["FA species"]] + [base[RATE][:, ri]]
    parts_f = [np.full(len(si), FLOOR_CONC), np.full(len(sets["FA species"]), FLOOR_CONC), np.full(len(CONDITIONS), FLOOR_RATE)]
    Jt, vt, ft = np.concatenate(parts_J), np.concatenate(parts_v), np.concatenate(parts_f)
    post_sd, info = score(Jt, vt, ft)
    tier1 = summarize(params, post_sd, info, len(vt))

    best = max(grid, key=lambda k: grid[k]["info_nats_per_point"])
    out = {"system": a.system, "params": params, "prior_sd_log": PRIOR_SD_LOG,
           "conditions": [c for c, _ in CONDITIONS], "noise": {"frac": NOISE_FRAC, "floor_conc": FLOOR_CONC,
                                                                "floor_rate": FLOOR_RATE},
           "measurement_sets": sets, "fd_step_log": a.h,
           "fd_max_relative_difference_h_vs_3h": fd_agreement,
           "grid": grid, "tier1_design": tier1, "best_cell_per_point": best,
           "seconds": round(time.perf_counter() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"{a.system}, params {params}; finite-difference check (h vs 3h): max relative difference {fd_agreement:.1e}")
    print(f"{'cell':52s} {'points':>6s} {'info (nats)':>11s} {'per point':>9s}  shrinkage_log " + " ".join(f"{p:>6s}" for p in params))
    for k, v in list(grid.items()) + [("Tier-1 design (series + profile + rates)", tier1)]:
        print(f"{k:52s} {v['n_points']:6d} {v['info_nats']:11.3f} {v['info_nats_per_point']:9.4f}  "
              f"{'':13s} " + " ".join(f"{v['shrinkage_log'][p]:6.3f}" for p in params))
    print(f"best cell per data point: {best}")
    print(f"wrote {a.out} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
