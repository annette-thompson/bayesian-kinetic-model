"""Turn each benchmark run into one row: cost, information gained, and predictors.

Cost is deliberately decomposed rather than reported as a single wall-clock number,
because three independent things multiply together to produce it and they have
different causes:

  sec_per_solve    numerical stiffness at the parameter values the sampler visited
  solves_per_draw  posterior geometry -- how far NUTS has to travel, ~2^tree_depth
                   leapfrog steps per draw, each needing a gradient
  draws_to_converge  mixing and identifiability

Wall-clock is recorded but should not be the headline: these run on a shared,
heterogeneous, preemptable pool, and today a single system swung from 10 to 142
s/draw purely because a neighbour started compiling. `total_ode_solves` is the
hardware- and preemption-independent cost measure.

Cost alone is also not interpretable, because it is NOT monotonic in how useful a
parameter is. A parameter the data cannot constrain has posterior == prior, mixes
trivially, and converges FAST while teaching you nothing -- x2 is in the grid
precisely as that control. So every cost column is paired with an information column
(shrinkage, posterior width, whether the true value was recovered). A metric set that
cannot tell "converged fast because it was easy" from "converged fast because nothing
was learned" is the wrong metric set.

Predictors come from the Morris screen, the footprint count, and the measured ranges,
so each row is directly usable in a regression of log(cost) on parameter character.

Usage:
    python bench_metrics.py --tag bench [--csv out.csv]
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/projects/anth4580/Bayesian")
CFG_ROOT = ROOT / "Results" / "Chain Scaling Tests"
SCAL = ROOT / "job_files" / "chain_system_sensitivity_analysis"


def morris_lookup(system, param):
    """mu_star, sigma/mu_star and rank per objective from the Morris screen."""
    f = SCAL / f"morris_{system.replace('+', '_')}.json"
    if not f.exists():
        return {}
    j = json.loads(f.read_text())
    out = {}
    for obj, rows in j["results"].items():
        for i, r in enumerate(rows):
            if r["group"] == param:
                mu = r["mu_star"]
                out[f"morris_{obj}_mu"] = mu
                out[f"morris_{obj}_rank"] = i + 1
                out[f"morris_{obj}_sigma_ratio"] = (r["sigma"] / mu) if mu else float("nan")
    return out


def progress_series(run_dir):
    """(draws, wall_seconds, seconds_per_draw) from the checkpoint progress log."""
    p = run_dir / "checkpoint" / "progress_log.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue          # torn final line while the sampler is mid-write
    if len(rows) < 2:
        return None
    n = [r["warmup_done"] + r["sampling_done"] for r in rows]
    t = [r["t"] for r in rows]
    span = n[-1] - n[0]
    return dict(draws=n[-1], wall_s=t[-1] - t[0],
                sec_per_draw=(t[-1] - t[0]) / span if span else float("nan"),
                device=rows[-1].get("device", "?"))


def sampler_stats(run_dir):
    """Mean tree depth and total leapfrog steps from the checkpointed sampling stats.

    n_steps is the leapfrog count per draw; each leapfrog step costs one gradient
    evaluation, which is one forward plus one adjoint ODE solve over all conditions.
    Summing it gives a hardware-independent measure of total work.
    """
    z = run_dir / "checkpoint" / "draws.zarr" / "sampling_stats"
    if not z.exists():
        return {}
    try:
        import zarr
        g = zarr.open(str(z), mode="r")
        out = {}
        if "n_steps" in g:
            ns = np.asarray(g["n_steps"][:])
            ns = ns[ns > 0]
            if ns.size:
                out["total_leapfrog_steps"] = int(ns.sum())
                out["mean_leapfrog_per_draw"] = float(ns.mean())
        if "tree_depth" in g:
            td = np.asarray(g["tree_depth"][:])
            td = td[td > 0]
            if td.size:
                out["mean_tree_depth"] = float(td.mean())
                out["max_tree_depth"] = int(td.max())
        if "step_size" in g:
            ss = np.asarray(g["step_size"][:])
            ss = ss[ss > 0]
            if ss.size:
                out["final_step_size"] = float(ss.ravel()[-1])
        return out
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"  (zarr read failed for {run_dir.name}: {exc})", file=sys.stderr)
        return {}


def convergence_from_log(system, param, tag):
    """draws-to-converge and the binding criterion, parsed from the job log."""
    logs = sorted((ROOT / "job_files").glob(f"{tag}_{system}_{param}.*.out"))
    if not logs:
        return {}
    txt = logs[-1].read_text()
    out = {}
    m = re.search(r"Converged at (\d+) sampling draws", txt)
    if m:
        out["draws_to_converge"] = int(m.group(1))
    checks = re.findall(r"check at (\d+) sampling draws: r_hat=([\d.]+) ess_bulk=([\d.]+)", txt)
    if checks:
        n, rh, ess = checks[-1]
        out["final_r_hat"] = float(rh)
        out["final_ess_bulk"] = float(ess)
        # Which criterion actually held things up: if ESS cleared 400 well before
        # r_hat cleared 1.01, the ESS floor is doing no work.
        first_ess_ok = next((int(c[0]) for c in checks if float(c[2]) >= 400), None)
        first_rhat_ok = next((int(c[0]) for c in checks if float(c[1]) < 1.01), None)
        out["first_ess_ok_at"] = first_ess_ok
        out["first_rhat_ok_at"] = first_rhat_ok
        out["binding_criterion"] = (
            "r_hat" if (first_rhat_ok or 1e9) > (first_ess_ok or 1e9) else "ess")
    if "stopped_reason: converged" in txt:
        out["stopped_reason"] = "converged"
    elif re.search(r"stopped_reason: (\w+)", txt):
        out["stopped_reason"] = re.search(r"stopped_reason: (\w+)", txt).group(1)
    return out


def information(run_dir, param):
    """Shrinkage and recovery -- did the run actually learn anything?"""
    post_f = run_dir / "posterior_samples_pm.nc"
    prior_f = run_dir / "prior_samples_pm.nc"
    if not post_f.exists() or not prior_f.exists():
        return {}
    try:
        import arviz as az
        post = az.from_netcdf(str(post_f))
        prior = az.from_netcdf(str(prior_f))
        a = np.asarray(post.posterior[param]).ravel()
        p = np.asarray(prior.prior[param]).ravel()
        sa, sp = np.std(np.log(a)), np.std(np.log(p))
        q = np.percentile(a, [2.5, 50, 97.5])
        return dict(post_median=float(q[1]), post_lo=float(q[0]), post_hi=float(q[2]),
                    post_sd_log=float(sa), prior_sd_log=float(sp),
                    shrinkage=float(1 - (sa * sa) / (sp * sp)),
                    narrowing_factor=float(sp / sa) if sa else float("nan"),
                    recovered_truth=bool(q[0] <= 1.0 <= q[2]))
    except Exception as exc:  # noqa: BLE001
        print(f"  (netcdf read failed for {run_dir.name}: {exc})", file=sys.stderr)
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--csv")
    a = ap.parse_args()

    rows = []
    for d in sorted(CFG_ROOT.glob(f"Chain * - {a.tag} *")):
        m = re.match(rf"Chain (.+) - {re.escape(a.tag)} (\w+)$", d.name)
        if not m:
            continue
        system, param = m.group(1), m.group(2)
        row = dict(system=system, parameter=param)

        cfg_f = d / "solver_params.json"
        if cfg_f.exists():
            meta = json.loads(cfg_f.read_text()).get("benchmark_meta", {})
            row.update({k: meta[k] for k in
                        ("prior_lower", "prior_upper", "prior_width_decades")
                        if k in meta})
        row.update(morris_lookup(system, param))
        prog = progress_series(d)
        if prog:
            row.update(prog)
        row.update(sampler_stats(d))
        row.update(convergence_from_log(system, param, a.tag))
        row.update(information(d, param))
        if "total_leapfrog_steps" in row:
            # one leapfrog step = one gradient = forward + adjoint solve per condition
            row["total_ode_solves_est"] = row["total_leapfrog_steps"] * 2
        rows.append(row)

    if not rows:
        print(f"no runs found matching tag '{a.tag}'")
        return

    cols = ["system", "parameter", "prior_width_decades", "draws_to_converge",
            "sec_per_draw", "mean_leapfrog_per_draw", "mean_tree_depth",
            "total_ode_solves_est", "shrinkage", "narrowing_factor",
            "post_median", "recovered_truth", "binding_criterion",
            "morris_avg_chain_length_rank", "morris_avg_chain_length_mu",
            "morris_avg_chain_length_sigma_ratio", "stopped_reason", "device"]
    W = 15
    print("".join(str(c)[:W - 1].ljust(W) for c in cols[:9]))
    print("-" * (W * 9))
    for r in rows:
        print("".join(
            (f"{r[c]:.4g}" if isinstance(r.get(c), float) else str(r.get(c, "-")))[:W - 1].ljust(W)
            for c in cols[:9]))

    if a.csv:
        allcols = sorted({k for r in rows for k in r})
        ordered = [c for c in cols if c in allcols] + [c for c in allcols if c not in cols]
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ordered)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}  ({len(rows)} rows x {len(ordered)} cols)")


if __name__ == "__main__":
    main()
