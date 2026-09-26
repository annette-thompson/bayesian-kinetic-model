"""Identifiability from a finished run's posterior, for Figs 6 and 6b (outline 3.3).

  Fig 6   prior-to-posterior shrinkage per parameter (1 - posterior sd / prior sd), on the
          scale the sampler walks (log for LogNormal groups, natural for the d-type Normals).
          Below 0.5 is flagged as weakly identified.
  Fig 6b  the posterior correlation matrix, and an eigendecomposition of the posterior
          covariance with every coordinate standardised by its prior sd. The prior's
          covariance is then the identity, so each eigenvalue is the fraction of prior
          variance left along its direction: a small one is a combination the data pin down
          even when the individual parameters are not, one near 1 is a direction the data do
          not touch. Reported as eigenvalues and directions, not only a picture.

Known answer (R3): on C8 every d-scaled TesA step shares coefficient 12, so only 12*d1 + d2 is
identifiable. With d1's prior sd a twelfth of d2's, that combination is the (1, 1) direction in
prior-standardised units, which should be tight, while (1, -1) should stay near the prior. The
`--selftest` checks the method on a synthetic ridge with that shape.

Usage:
  python identifiability_report.py --run "Tier1 C14+unsat - a1c3a2"
  python identifiability_report.py --posterior <nc or export JSON> [--key set/system] --config <solver_params.json>
  python identifiability_report.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from forward_model import PROJECT, posterior_draws

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "Utilities"))
WEAK = 0.5


def prior_scale(cfg):
    """{param: (centre, sd, log_scale)} on the sampling coordinate, from the sampler's own priors."""
    from inference_runner import _fit_prior
    out = {}
    for spec in cfg["free_kinetic_params"]:
        d = _fit_prior(spec["param_name"], spec["prior_dist_params"])
        mu, sigma = (float(v) for v in d.params)
        out[spec["param_name"]] = (mu, sigma, d.__class__.__name__ == "LogNormal")
    return out


def analyse(draws, prior):
    """draws: {param: 1-D natural-scale array}; prior: prior_scale output."""
    names = list(prior)
    X = np.column_stack([np.log(draws[p]) if prior[p][2] else np.asarray(draws[p]) for p in names])
    sd_prior = np.array([prior[p][1] for p in names])
    sd_post = X.std(axis=0, ddof=1)
    shrink = 1.0 - sd_post / sd_prior
    corr = np.corrcoef(X, rowvar=False)
    Z = (X - X.mean(axis=0)) / sd_prior
    lam, vec = np.linalg.eigh(np.cov(Z, rowvar=False))       # ascending: tightest first
    vec *= np.sign(vec[np.abs(vec).argmax(axis=0), range(len(names))])   # largest loading positive
    return {
        "params": names,
        "scale": {p: ("log" if prior[p][2] else "natural") for p in names},
        "shrinkage": {p: float(s) for p, s in zip(names, shrink)},
        "weakly_identified": [p for p, s in zip(names, shrink) if s < WEAK],
        "posterior_sd": {p: float(s) for p, s in zip(names, sd_post)},
        "prior_sd": {p: float(s) for p, s in zip(names, sd_prior)},
        "correlation": [[float(v) for v in row] for row in corr],
        "eigen": [{"variance_left": float(l), "shrinkage_along": float(1 - np.sqrt(max(l, 0.0))),
                   "direction": {p: float(v) for p, v in zip(names, vec[:, i])}}
                  for i, l in enumerate(lam)],
    }


def report(res):
    print("shrinkage (sampling scale): " + ", ".join(
        f"{p} {res['shrinkage'][p]:.3f} ({res['scale'][p]})" for p in res["params"]))
    print(f"weakly identified (< {WEAK}): {res['weakly_identified'] or 'none'}")
    print("correlation:")
    for p, row in zip(res["params"], res["correlation"]):
        print(f"  {p:5s} " + " ".join(f"{v:+.3f}" for v in row))
    print("eigen (prior-standardised; variance left, direction):")
    for e in res["eigen"]:
        print(f"  {e['variance_left']:.3g}  shrinkage {e['shrinkage_along']:.3f}  " +
              ", ".join(f"{p} {v:+.3f}" for p, v in e["direction"].items()))


def selftest(seed=0):
    """A d1/d2-shaped ridge: prior sds 1.1748/12 and 1.1748, only 12*d1 + d2 informed."""
    rng = np.random.default_rng(seed)
    s2 = 1.1748
    prior = {"d1": (0.0, s2 / 12, False), "d2": (0.0, s2, False)}
    # Posterior in prior-standardised units: (1,1) direction shrunk to 5% of prior variance.
    u = np.array([1.0, 1.0]) / np.sqrt(2)
    w = np.array([1.0, -1.0]) / np.sqrt(2)
    z = np.outer(rng.normal(0, np.sqrt(0.05), 20000), u) + np.outer(rng.normal(0, 1.0, 20000), w)
    draws = {"d1": z[:, 0] * s2 / 12, "d2": z[:, 1] * s2}
    res = analyse(draws, prior)
    tight, flat = res["eigen"][0], res["eigen"][1]
    dir_ok = abs(abs(tight["direction"]["d1"]) - abs(tight["direction"]["d2"])) < 0.05 and \
        np.sign(tight["direction"]["d1"]) == np.sign(tight["direction"]["d2"])
    ok = abs(tight["variance_left"] - 0.05) < 0.01 and abs(flat["variance_left"] - 1.0) < 0.05 and dir_ok \
        and set(res["weakly_identified"]) == {"d1", "d2"}
    report(res)
    print("PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="Results/Tier1/<run>: reads its posterior_samples_pm.nc and solver_params.json")
    ap.add_argument("--posterior")
    ap.add_argument("--key", default=None)
    ap.add_argument("--config")
    ap.add_argument("--draws", type=int, default=4000, help="posterior draws used, thinned evenly")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.run:
        run_dir = PROJECT / "Results" / "Tier1" / a.run
        post, cfg_path = run_dir / "posterior_samples_pm.nc", run_dir / "solver_params.json"
    else:
        post = Path(a.posterior) if Path(a.posterior).is_absolute() else PROJECT / a.posterior
        cfg_path = Path(a.config) if Path(a.config).is_absolute() else PROJECT / a.config
    cfg = json.loads(cfg_path.read_text())
    prior = prior_scale(cfg)
    sets = posterior_draws(post, list(prior), a.draws, key=a.key)
    draws = {p: np.array([s[p] for s in sets]) for p in prior}
    res = analyse(draws, prior)
    res.update({"posterior": str(post), "key": a.key, "config": str(cfg_path), "n_draws": len(sets)})
    report(res)
    out = Path(a.out) if a.out else (post.parent / "identifiability.json" if a.run else HERE / "identifiability.json")
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
