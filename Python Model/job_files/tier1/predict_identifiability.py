"""Predict Fig 6/6b's identifiability answer before sampling, from the Fisher information at
the truth: the same linear-Gaussian calculation as expected_information_grid.py (appendix D),
but for any free parameters and on the sampler's own coordinates.

For a run's config it perturbs each free parameter at its truth (in log space for LogNormal
groups, additively for the d-type Normal groups), stacks the Tier-1 design's sensitivities
(baseline time series, baseline profile at 720 s, initial rates under the five conditions),
and forms the posterior covariance Sigma = (J^T W J + prior precision)^-1. Reported in
prior-standardised coordinates, as identifiability_report.py reports a sampled posterior:
  contraction   1 - posterior variance / prior variance, per parameter
  correlation   the posterior correlation
  eigen         the variance left along each eigen-direction (tightest first) and the direction
Written to <run>/identifiability_predicted.json, to set beside the sampled identifiability.json.

  python predict_identifiability.py --run "Tier1 C14+unsat - d1d2"
  python predict_identifiability.py --run "Tier1 C14+unsat - d1d2" --system C16
      # the same free parameters, priors and truth on another system, which needs no config
      # of its own; written to Results/Tier1/figures/identifiability_predicted_<system>_<params>.json

One system per process: the laptop's jaxlib can abort compiling a second model in one process.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import expected_information_grid as eig
from forward_model import CONDITIONS, FLOOR_CONC, NOISE_FRAC, ForwardModel
from recovery_report import RESULTS, Z975

H = 1e-3   # step in prior sd units (additive groups) or in log (LogNormal groups)


def prior_of(spec):
    """(log scale?, prior sd on the sampling coordinate)."""
    pr = spec["prior_dist_params"]
    sd = math.log(float(pr["upper"]) / float(pr["lower"])) / (2 * Z975) if pr["distribution"] == "LogNormal" \
        else (float(pr["upper"]) - float(pr["lower"])) / (2 * Z975)
    return pr["distribution"] == "LogNormal", sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--system", default=None, help="predict on this system instead, with the run's parameters")
    a = ap.parse_args()
    run_dir = RESULTS / a.run
    cfg = json.loads((run_dir / "solver_params.json").read_text())
    system = a.system or Path(cfg["output_paths"]["reactions_source"][0]).parent.name
    params = [s["param_name"] for s in cfg["free_kinetic_params"]]
    priors = {s["param_name"]: prior_of(s) for s in cfg["free_kinetic_params"]}
    truth = {p: float(cfg["tier1_truth"][p]) for p in params}

    times = sorted(set(eig.SERIES_TIMES) | {eig.RATE_TIME, eig.END_TIME})
    fm = ForwardModel(system, times=times)
    species = fm.names(r"C\d+_FA(_unsat)? \(uM\)")
    names = sorted(set(species) | {eig.TOTAL, eig.RATE})
    t = np.asarray(fm.times)
    si = [int(np.argmin(np.abs(t - s))) for s in eig.SERIES_TIMES]
    ei, ri = int(np.argmin(np.abs(t - eig.END_TIME))), int(np.argmin(np.abs(t - eig.RATE_TIME)))

    # The truth block lists the run's own system's groups; keep the ones this system has.
    cfg["tier1_truth"] = {g: v for g, v in cfg["tier1_truth"].items() if g in fm.nominal}
    base = eig.simulate(fm, fm.theta(dict(cfg["tier1_truth"])), names)
    J = {n: [] for n in names}
    for p in params:
        log_scale, sd = priors[p]
        if log_scale:   # derivative with respect to log(p)
            up, dn = truth[p] * math.exp(H), truth[p] * math.exp(-H)
            step = 2 * H
        else:           # derivative with respect to p, stepping a small fraction of its prior sd
            up, dn = truth[p] + H * sd, truth[p] - H * sd
            step = 2 * H * sd
        yu = eig.simulate(fm, fm.theta({**cfg["tier1_truth"], p: up}), names)
        yd = eig.simulate(fm, fm.theta({**cfg["tier1_truth"], p: dn}), names)
        for n in names:
            J[n].append((yu[n] - yd[n]) / step)
    J = {n: np.stack(v, axis=-1) for n, v in J.items()}
    Jt = np.concatenate([J[eig.TOTAL][0, si, :], np.stack([J[n][0, ei, :] for n in species]), J[eig.RATE][:, ri, :]])
    vt = np.concatenate([base[eig.TOTAL][0, si], np.array([base[n][0, ei] for n in species]), base[eig.RATE][:, ri]])
    ft = np.concatenate([np.full(len(si), FLOOR_CONC), np.full(len(species), FLOOR_CONC),
                         np.full(len(CONDITIONS), eig.FLOOR_RATE)])
    sigma = NOISE_FRAC * np.abs(vt) + ft
    F = Jt.T @ (Jt / sigma[:, None] ** 2)
    sd_prior = np.array([priors[p][1] for p in params])
    post = np.linalg.inv(F + np.diag(1.0 / sd_prior ** 2))
    S = post / np.outer(sd_prior, sd_prior)                 # prior-standardised: prior = identity
    lam, vec = np.linalg.eigh(S)
    vec *= np.sign(vec[np.abs(vec).argmax(axis=0), range(len(params))])
    corr = post / np.sqrt(np.outer(np.diag(post), np.diag(post)))
    out = {"run": a.run, "system": system, "params": params, "n_points": int(len(vt)),
           "scale": {p: "log" if priors[p][0] else "natural" for p in params},
           "prior_sd": {p: float(s) for p, s in zip(params, sd_prior)},
           "contraction": {p: float(1 - S[i, i]) for i, p in enumerate(params)},
           "correlation": [[float(v) for v in row] for row in corr],
           "eigen": [{"variance_left": float(l), "contraction_along": float(1 - l),
                      "direction": {p: float(v) for p, v in zip(params, vec[:, i])}} for i, l in enumerate(lam)],
           "fisher_condition_number": float(np.linalg.cond(F)) if np.all(np.isfinite(F)) else None}
    path = run_dir / "identifiability_predicted.json" if not a.system else \
        RESULTS / "figures" / f"identifiability_predicted_{system}_{''.join(params)}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.run} ({system}, {len(vt)} points), predicted at the truth:")
    print("  contraction: " + ", ".join(f"{p} {out['contraction'][p]:.4f} ({out['scale'][p]})" for p in params))
    print("  correlation: " + " ".join(f"{v:+.3f}" for v in corr[0, 1:]))
    for e in out["eigen"]:
        print(f"  variance left {e['variance_left']:.3g} (contraction {e['contraction_along']:.4f}) along "
              + ", ".join(f"{p} {v:+.3f}" for p, v in e["direction"].items()))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
