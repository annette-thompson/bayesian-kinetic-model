"""Build a solver_params.json for a multi-parameter no-floor pilot run, cloning
the system's existing single-parameter (a1-only) no-floor config as a template.

Reuses everything about that config EXCEPT: free_kinetic_params (extended to
every requested parameter, LogNormal, median-pinned to 1.0, bounds as given --
same distribution family/convention a1's own prior already uses), tune (600,
per the pilot's explicit instruction to try a shorter warmup with plans to
validate that isn't too small later), ess_threshold (null -- convergence
simplified to r-hat alone), and rank_ecdf_prob/rank_ecdf_simulations (dropped
entirely, for the same reason). output_paths.results_save_dir points at a new,
clearly-named directory so this never collides with the single-parameter run
it was cloned from.

Priors are specified once, as the RATE-MULTIPLIER window every parameter is
screened over (--rate-window, default 0.1,10 -- the same 100-fold span a1's own
prior and the Morris screen already use), and converted per parameter here:

  * Ordinary multiplicative groups (a1, a2, c2, c3, ...) multiply a rate
    constant directly, so the group value IS the multiplier: LogNormal pinned
    to median 1.0 over the window as given.
  * The d-groups are additive inside exp() -- TesA is 1/exp(12*d1 + d2), with a
    nominal of 0.0, not 1.0 -- so a multiplier is not a valid value for them at
    all. To induce rate multiplier m they need 12*d1 + d2 = -ln(m), i.e.
    d1 = -ln(m)/12 or d2 = -ln(m) with the other at nominal. They therefore get
    a NORMAL prior centred on 0.0 over the converted window, which is the same
    conversion morris_screen.py and scaling_sensitivity.py already apply.

Getting this wrong is silent, not loud: a LogNormal median-1.0 prior on d1 puts
every draw in a region the nominal value (0.0) isn't even in.

Usage:
    python build_multiparam_config.py --system C6 --params a1,c2 --paramset a1c2
    python build_multiparam_config.py --system C6 --params a1,c3,d1 --paramset a1c3d1
    # explicit per-parameter bounds still override the converted window:
    python build_multiparam_config.py --system C6 --params a1,c2 \
        --bounds 0.1,10,0.09,1.35 --paramset a1c2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path("/projects/anth4580/Bayesian")
BASE = ROOT / "Results" / "Chain Scaling Tests"

# Coefficient each additive-in-exp group carries in TesA's 1/exp(12*d1 + d2).
# Anything absent here is an ordinary multiplicative group.
D_COEFF = {"d1": 12.0, "d2": 1.0}


def prior_for(param, rate_lo, rate_hi, explicit=None):
    """prior_dist_params for one parameter over a rate-multiplier window.

    `explicit`, when given, is used as the bounds verbatim (already in the
    parameter's own units) and only the distribution family is inferred.
    """
    coeff = D_COEFF.get(param)
    if coeff is None:
        lo, hi = explicit if explicit else (rate_lo, rate_hi)
        return {"distribution": "LogNormal", "lower": lo, "upper": hi,
                "mass": 0.95, "fixed_stat": ["median", 1.0]}
    # value = -ln(m)/coeff, so the multiplier window inverts: the largest
    # multiplier maps to the most negative value.
    lo, hi = explicit if explicit else (-math.log(rate_hi) / coeff, -math.log(rate_lo) / coeff)
    prior = {"distribution": "Normal", "lower": lo, "upper": hi, "mass": 0.95}
    # Sample through u = coeff * value: u is the log-rate multiplier itself, so
    # its prior sd matches the LogNormal groups' log-space sd exactly. Sampled
    # raw, d1's sd is 12x smaller than a1's -- which collapsed warmup step size
    # and let initial-point jitter start chains at TesA up to 25,800x.
    if coeff != 1.0:
        prior["sample_scale"] = coeff
    return prior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--params", required=True, help="comma-separated, e.g. a1,c2")
    ap.add_argument("--bounds", default=None,
                    help="optional explicit lower1,upper1,lower2,upper2,... in each parameter's OWN "
                         "units, matching --params order; overrides --rate-window")
    ap.add_argument("--rate-window", default="0.1,10",
                    help="rate-multiplier span every parameter is screened over (default 0.1,10)")
    ap.add_argument("--paramset", required=True, help="short label for the new dir, e.g. a1c2")
    ap.add_argument("--tune", type=int, default=600)
    a = ap.parse_args()

    params = a.params.split(",")
    rate_lo, rate_hi = (float(x) for x in a.rate_window.split(","))
    explicit_pairs = None
    if a.bounds:
        vals = [float(x) for x in a.bounds.split(",")]
        if len(vals) != 2 * len(params):
            raise SystemExit(f"--bounds needs {2*len(params)} numbers for {len(params)} params, got {len(vals)}")
        explicit_pairs = [(vals[2 * i], vals[2 * i + 1]) for i in range(len(params))]
    priors = [prior_for(p, rate_lo, rate_hi, explicit_pairs[i] if explicit_pairs else None)
              for i, p in enumerate(params)]

    template_dir = BASE / f"Chain {a.system} - a1_0.1-10_no_floor"
    template_path = template_dir / "solver_params.json"
    if not template_path.exists():
        raise SystemExit(f"template not found: {template_path}")
    cfg = json.loads(template_path.read_text())

    cfg["free_kinetic_params"] = [
        {"rxn_name": None, "param_name": p, "prior_dist_params": prior}
        for p, prior in zip(params, priors)
    ]

    post = cfg["posterior_sampling"]
    post["tune"] = a.tune
    post["ess_threshold"] = None
    post.pop("rank_ecdf_prob", None)
    post.pop("rank_ecdf_simulations", None)

    new_dir_name = f"Chain {a.system} - {a.paramset}_no_floor"
    cfg["output_paths"]["results_save_dir"] = f"Results/Chain Scaling Tests/{new_dir_name}"

    new_dir = BASE / new_dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    out_path = new_dir / "solver_params.json"
    out_path.write_text(json.dumps(cfg, indent=2))

    print(f"Wrote {out_path}")
    for p, prior in zip(params, priors):
        note = " (additive-in-exp, converted from the rate window)" if p in D_COEFF else ""
        if prior.get("sample_scale"):
            note += f", sampled as {prior['sample_scale']:g}*{p}"
        print(f"  {p}: {prior['distribution']}[{prior['lower']:.5g}, {prior['upper']:.5g}]{note}")
    print(f"  tune={a.tune}  ess_threshold=null  rank_ecdf=removed")
    print(f"  (cloned from {template_path}, that file is untouched)")


if __name__ == "__main__":
    main()
