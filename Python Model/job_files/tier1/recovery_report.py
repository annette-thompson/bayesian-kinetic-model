"""Did each Tier-1 run recover the known truth, and how honest is its uncertainty?

Tier 1's whole justification is that the answer is known in advance, so every run has to be
scored against it. Nothing else in the repo does this for more than one parameter.

The truth is read from each config's `tier1_truth` block, written by build_tier1_configs.py
from the data's own ground_truth.json. For standard Tier-1 data that is the no-op
parameterisation -- the previous model's own solution:

    LogNormal groups (multiplicative)          truth = 1
    Normal, d-type groups (additive inside an  truth = 0
      exponential, 1/exp(n*d1 + d2))

but runs on deliberately moved data (the grouping test's off-grouping dataset, c3l = 3) get
their actual truth. Configs without the block fall back to the no-op values. A parameter the
data's parameterisation does not contain (the grouped model's `c3` fit to split data) has no
truth, and is scored for fit only.

Per parameter it reports, following Schad, Betancourt & Vasishth (2021, eqs. 4-5), on the scale
the prior is Normal on: log(x) for the LogNormal groups, x itself for the d-type groups.
  z            (posterior mean - truth) / posterior sd. How far off the recovered value is,
               in units of its own stated uncertainty. |z| > 2 with high contraction is the
               dangerous case -- confidently wrong -- and is flagged.
  contraction  posterior contraction, 1 - posterior variance / prior variance: how much of
               the prior's uncertainty the data removed (1 = pinned down, 0 = learned
               nothing). The outline's 3.3 cutoff for "weakly identified" is < 0.5. On log(x)
               it does not depend on where the prior median sits or where the truth lies,
               which the natural-scale number does (a LogNormal's natural-scale variance is
               dominated by its upper tail).
  z_natural, contraction_natural   the same on the natural scale, for reference.
  covered      whether truth falls inside the 50 / 90 / 95% central credible intervals.
  quantile     the posterior quantile at which truth sits (0.5 = dead centre).
The z and contraction columns of the table are the log-scale ones where the prior is LogNormal.

One caveat this script will not paper over: the quantile column is NOT an SBC rank
statistic. A proper SBC uniformity test (Talts et al. 2018) needs the true value drawn from
the prior on each replicate; these runs all sit at the same fixed no-op truth, so the
quantiles are not expected to be uniform and testing them for uniformity would be wrong.
They become SBC ranks only in the replicate study (outline 3.1 Figure 3), where the truth
varies. Reported here because it is free and shows systematic bias at a glance.

Usage:
  python recovery_report.py                      # every finished Tier-1 run
  python recovery_report.py --only "Tier1 C8 - a1c3"
  python recovery_report.py --json out.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

_CLUSTER = Path("/projects/anth4580/Bayesian")
ROOT = _CLUSTER if _CLUSTER.is_dir() else Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "Results" / "Tier1"
Z975 = 1.959963984540054          # norm.ppf(0.975), for the 95% mass the priors are built on
CONTRACTION_WEAK = 0.5            # outline 3.3's "weakly identified" cutoff


def truth_and_prior_sd(spec, truths=None):
    """(truth, prior sd) for one free-parameter spec.

    Truth comes from the config's `tier1_truth` block (the scaling values the data were
    generated at) when present; otherwise it is the no-op value, 1 for multiplicative groups
    and 0 for d-type groups. It is None when the parameter does not exist in the data's own
    parameterisation (the grouped model fit to off-grouping data), and the run is then scored
    for fit only.

    Both prior families are specified as a central interval carrying `mass` (0.95), so the sd
    follows from the half-width. The LogNormal interval is in natural space; its median need
    not be 1 (the robustness runs shift it), so both log-mean and log-sd come from the bounds.
    """
    name = spec["param_name"]
    pr = spec["prior_dist_params"]
    dist = pr.get("distribution")
    z = Z975 if float(pr.get("mass", 0.95)) == 0.95 else _z(float(pr["mass"]))
    if truths is not None:
        truth = truths.get(name)
        if truth is None and name[:-1] in truths:              # c3s / c3l from a whole c3
            truth = truths[name[:-1]]
        children = {v for t, v in truths.items() if t[:-1] == name}
        if truth is None and len(children) == 1:               # whole c3 from equal halves
            truth = next(iter(children))
    else:
        truth = 0.0 if dist == "Normal" else 1.0
    if dist == "LogNormal":
        lo, hi = float(pr["lower"]), float(pr["upper"])
        mu, sigma = 0.5 * math.log(lo * hi), math.log(hi / lo) / (2 * z)
        sd = math.sqrt(math.exp(sigma ** 2) - 1.0) * math.exp(mu + sigma ** 2 / 2)
        return truth, sd
    if dist == "Normal":
        return truth, (float(pr["upper"]) - float(pr["lower"])) / (2 * z)
    raise ValueError(f"unhandled prior distribution {dist!r} for {name}")


def _z(mass):
    from scipy.stats import norm
    return float(norm.ppf(0.5 + mass / 2))


def _zc(x, truth, prior_sd):
    """(z, contraction) of draws x against a truth and prior sd on the same scale."""
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    z = (mean - truth) / sd if (truth is not None and sd > 0) else None
    contraction = 1.0 - (sd / prior_sd) ** 2 if prior_sd > 0 else float("nan")
    return z, contraction


def score(draws, truth, prior_sd, log_prior_sd=None):
    """Recovery scores for one parameter. With log_prior_sd (a LogNormal prior), z and
    contraction are on log(x); otherwise on x itself."""
    flat = np.asarray(draws, float).ravel()
    mean, sd = float(flat.mean()), float(flat.std(ddof=1))
    out = {"mean": mean, "sd": sd, "median": float(np.median(flat)), "truth": truth, "prior_sd": prior_sd}
    z_nat, c_nat = _zc(flat, truth, prior_sd)
    out["contraction_natural"] = c_nat
    if log_prior_sd is not None:
        logx = np.log(flat)
        z, c = _zc(logx, math.log(truth) if truth is not None and truth > 0 else None, log_prior_sd)
        out.update(scale="log", log_mean=float(logx.mean()), log_sd=float(logx.std(ddof=1)),
                   log_prior_sd=log_prior_sd)
    else:
        z, c = z_nat, c_nat
        out["scale"] = "natural"
    out["contraction"] = c
    out["ci95"] = [float(v) for v in np.percentile(flat, [2.5, 97.5])]
    if truth is None:
        out["confidently_wrong"] = False
        return out
    out["z"], out["z_natural"] = z, z_nat
    out["quantile_of_truth"] = float((flat < truth).mean())
    for lvl, lo, hi in ((50, 25.0, 75.0), (90, 5.0, 95.0), (95, 2.5, 97.5)):
        a, b = np.percentile(flat, [lo, hi])
        out[f"covered_{lvl}"] = bool(a <= truth <= b)
    # The failure mode that would undermine the paper: a tight posterior in the wrong place.
    out["confidently_wrong"] = bool(abs(out["z"]) > 2 and out["contraction"] > CONTRACTION_WEAK)
    return out


def analyse(run_dir):
    import arviz as az
    cfg = json.loads((run_dir / "solver_params.json").read_text())
    nc = next((p for p in (run_dir / f for f in
               (cfg.get("output_paths", {}).get("posterior_samples_file", "posterior_samples_pm.nc"),
                "posterior_samples_pm.nc")) if p.exists()), None)
    if nc is None:
        return {"run": run_dir.name, "skipped": "no posterior netcdf (not finalized yet)"}

    inf = az.from_netcdf(nc)
    have = set(inf.posterior.data_vars)
    rec = {"run": run_dir.name, "chains": int(inf.posterior.sizes["chain"]),
           "draws": int(inf.posterior.sizes["draw"]), "params": {}}
    for spec in cfg.get("free_kinetic_params", []):
        name = spec["param_name"]
        if name not in have:
            rec["params"][name] = {"error": f"not in posterior (have {sorted(have)})"}
            continue
        truth, prior_sd = truth_and_prior_sd(spec, cfg.get("tier1_truth"))
        pr = spec["prior_dist_params"]
        log_prior_sd = None
        if pr.get("distribution") == "LogNormal":
            # log(x) has a Normal prior whose sd does not depend on the median (the prior-shift
            # runs move only the median), so contraction on log(x) compares across runs.
            log_prior_sd = math.log(float(pr["upper"]) / float(pr["lower"])) / (2 * Z975)
        s = score(inf.posterior[name].values, truth, prior_sd, log_prior_sd)
        try:
            s["rhat"] = float(az.rhat(inf, var_names=[name], method="rank")[name].values)
            s["ess_bulk"] = float(az.ess(inf, var_names=[name], method="bulk")[name].values)
        except Exception:
            pass
        rec["params"][name] = s
    rec["all_covered_95"] = all(p.get("covered_95") for p in rec["params"].values()
                                if p.get("truth") is not None and "z" in p)
    rec["any_confidently_wrong"] = any(p.get("confidently_wrong") for p in rec["params"].values())
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="substring of the run directory name")
    ap.add_argument("--json", default=None, help="also write the full result here")
    a = ap.parse_args()

    if not RESULTS.is_dir():
        raise SystemExit(f"no Tier-1 results directory at {RESULTS}")
    runs = sorted(d for d in RESULTS.iterdir() if (d / "solver_params.json").exists())
    if a.only:
        runs = [r for r in runs if a.only in r.name]
    if not runs:
        raise SystemExit("no matching Tier-1 runs")

    out = []
    W = max(len(r.name) for r in runs) + 2
    hdr = f"{'run':<{W}}{'param':<8}{'truth':>7}{'mean':>10}{'sd':>9}{'z':>7}{'contr':>8}{'95%':>6}{'rhat':>8}"
    print(hdr); print("-" * len(hdr))
    for r in runs:
        try:
            rec = analyse(r)
        except Exception as e:
            rec = {"run": r.name, "error": f"{type(e).__name__}: {e}"}
        out.append(rec)
        if "skipped" in rec:
            print(f"{r.name:<{W}}(skipped: {rec['skipped']})")
            continue
        if "error" in rec:
            print(f"{r.name:<{W}}ERROR {rec['error'][:60]}")
            continue
        for name, p in rec["params"].items():
            if "error" in p:
                print(f"{rec['run']:<{W}}{name:<8}{p['error'][:50]}")
                continue
            if p["truth"] is None:
                print(f"{rec['run']:<{W}}{name:<8}{'n/a':>7}{p['mean']:>10.4f}"
                      f"{p['sd']:>9.4f}{'':>7}{p['contraction']:>8.3f}{'':>6}"
                      f"{p.get('rhat', float('nan')):>8.4f}  (no truth in this parameterisation)")
                continue
            flag = "  <-- CONFIDENTLY WRONG" if p["confidently_wrong"] else ""
            print(f"{rec['run']:<{W}}{name:<8}{p['truth']:>7.2f}{p['mean']:>10.4f}"
                  f"{p['sd']:>9.4f}{p['z']:>7.2f}{p['contraction']:>8.3f}"
                  f"{'yes' if p['covered_95'] else 'NO':>6}{p.get('rhat', float('nan')):>8.4f}{flag}")

    scored = [r for r in out if "params" in r]
    if scored:
        bad = [r["run"] for r in scored if r["any_confidently_wrong"]]
        miss = [r["run"] for r in scored if not r["all_covered_95"]]
        print(f"\n{len(scored)} run(s) scored.")
        print(f"  truth inside the 95% interval for every parameter: "
              f"{len(scored) - len(miss)}/{len(scored)}" + (f"  (missed: {miss})" if miss else ""))
        if bad:
            print(f"  CONFIDENTLY WRONG (|z| > 2 with contraction > {CONTRACTION_WEAK}): {bad}")
        weak = [(r["run"], n) for r in scored for n, p in r["params"].items()
                if "contraction" in p and p["contraction"] < CONTRACTION_WEAK]
        if weak:
            print(f"  weakly identified (contraction < {CONTRACTION_WEAK}): {weak}")
        print("  z and contraction on log(x) for LogNormal parameters; contraction = 1 - posterior "
              "variance / prior variance")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
