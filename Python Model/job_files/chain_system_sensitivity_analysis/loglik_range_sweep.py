"""Prior bounds from the PROFILE LOG-LIKELIHOOD, with two thresholds compared.

Supersedes the sensitivity-threshold approach in group_range_sweep.py for the purpose
of choosing priors. That version asked whether a factor-of-2 move changed one
condition's worst observable by more than the 10% noise. It is demonstrably too
conservative to set a prior: on C4_NoFB it declared a1 uninformative below 1.0, yet
a1's actual posterior there is [0.422, 1.296] with 66% of its mass below 1.0. A prior
built from it would have truncated two thirds of the real posterior.

The reason is a mismatch of scope. That criterion looks at a single observable on a
single condition; the likelihood aggregates ~20-40 measurements across conditions and
both datasets, so its effective resolution is better by roughly sqrt(n). Profiling the
actual log-likelihood removes the mismatch by measuring the thing inference optimizes.

To guarantee the profile matches inference exactly rather than approximately, this
builds the real PyMC model via inference_runner._build_model_bundle and evaluates its
compiled logp. No reimplementation of the noise model or the observation mapping.

TWO THRESHOLDS, REPORTED SIDE BY SIDE

  conventional   delta_logp = 2.0. Standard practice: under asymptotic normality a
                 drop of ~1.92 from the maximum brackets a 95% interval for one
                 parameter. Defensible by convention, but an assumption about
                 posterior shape that an ODE likelihood need not satisfy.

  calibrated     the delta that actually reproduces the measured 95% posterior
                 intervals for a1 on the systems already sampled (C4_NoFB, C6, C8).
                 Empirical, and it tests the conventional value rather than assuming
                 it: if the fitted delta lands near 2, the convention is validated
                 here; if not, that is worth knowing before applying it to parameters
                 with no posterior to check against.

Both are reported for every (system, parameter), so the choice is visible in the
output rather than baked in.

Usage:
    python loglik_range_sweep.py --calibrate                  # fit delta from known posteriors
    python loglik_range_sweep.py --groups a1,a2,b3,c1,d1,b1,x2 C4_NoFB C6 C8
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")

import numpy as np

import inference_runner as ir

ROOT = Path("/projects/anth4580/Bayesian")
CFG_ROOT = ROOT / "Results" / "Chain Scaling Tests"

# Posterior 95% intervals already measured for a1, used to calibrate the threshold.
KNOWN_A1_POSTERIOR = {
    "C4_NoFB": (0.4220, 1.2957),
    "C6":      (0.9667, 1.0307),
    "C8":      (0.9792, 1.0207),
}

# log10 grid, denser near 1.0 where the profile turns over.
GRID = np.unique(np.concatenate([
    np.linspace(-3.0, -1.0, 9),
    np.linspace(-1.0, -0.1, 10),
    np.linspace(-0.1, 0.1, 21),
    np.linspace(0.1, 1.0, 10),
    np.linspace(1.0, 3.0, 9),
]))


def group_value_for(group, mult):
    """Value inducing rate multiplier `mult`; d-groups are additive inside exp()."""
    if group == "d1":
        return -np.log(mult) / 12.0
    if group == "d2":
        return -np.log(mult)
    return mult


def build_logp(system, param):
    """Compile the real model's LIKELIHOOD as a function of this one parameter.

    Two things this must get right, both of which bit on the first attempt:

    * Only the observed-data term is evaluated, not prior + likelihood. A profile
      likelihood contaminated by the prior would measure the prior's curvature as
      well as the data's, which defeats the purpose of calibrating a prior from it.
    * The carrier prior is Normal rather than LogNormal, so the value variable is
      untransformed and the point dict is keyed by the plain parameter name. A
      LogNormal would put a `_log__` transform in the way, and could not represent
      the d-groups at all, since those are additive inside exp() and take negative
      values (d1 = -ln(m)/12). Because only the likelihood is evaluated, the carrier
      prior has no effect on the result beyond fixing the transform and support.
    """
    cfg_path = CFG_ROOT / f"Chain {system} - a1 tightest" / "solver_params.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["free_kinetic_params"] = [dict(
        rxn_name=None, param_name=param,
        # The builder always routes through pz.maxent, so lower/upper are required
        # even for a Normal. Wide and symmetric: this is only a carrier that fixes the
        # transform and support, and it never enters the profile because only the
        # observed-data term is evaluated.
        prior_dist_params=dict(distribution="Normal", lower=-1000.0, upper=1000.0,
                               mass=0.95))]
    # path_base is stored relative to the config's OWN directory ("../../.." from
    # Results/Chain Scaling Tests/Chain X/). Writing the temp copy elsewhere would
    # resolve every reaction path from the wrong root, so pin it absolutely first.
    cfg["path_base"] = str((cfg_path.parent / cfg.get("path_base", ".")).resolve())
    tmp = HERE / f"_tmp_{system.replace('+','_')}_{param}.json"
    tmp.write_text(json.dumps(cfg))
    imported = ir.import_solver_params(tmp)
    bundle = ir._build_model_bundle(imported)
    model = bundle.pm_model
    fn = model.compile_logp(vars=model.observed_RVs, sum=True)
    ip = model.initial_point()
    keys = [k for k in ip if k == param or k.startswith(param + "_")]
    if not keys:
        raise RuntimeError(f"cannot find value variable for {param!r} in {sorted(ip)}")
    key = keys[0]
    if key != param:
        raise RuntimeError(f"{param!r} is transformed as {key!r}; expected untransformed")

    def logp(mult):
        pt = dict(ip)
        pt[key] = np.asarray(group_value_for(param, mult), dtype=float)
        return float(fn(pt))

    return logp, tmp


def profile(system, param):
    logp, tmp = build_logp(system, param)
    out = []
    for lg in GRID:
        m = 10.0 ** lg
        try:
            v = logp(m)
        except Exception as exc:
            if not out:      # report the first failure; a silent NaN grid is useless
                print(f"    logp failed at m={m:g}: {type(exc).__name__}: {exc}",
                      flush=True)
            v = float("nan")
        out.append((m, v))
    tmp.unlink(missing_ok=True)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def bounds_at(mults, lp, delta):
    """Where the profile falls `delta` below its maximum, either side of the peak."""
    ok = np.isfinite(lp)
    if ok.sum() < 3:
        return None, None
    m, v = mults[ok], lp[ok]
    i = int(np.argmax(v))
    target = v[i] - delta

    def cross(idx_range, side):
        prev_m, prev_v = m[i], v[i]
        for j in idx_range:
            if v[j] <= target:
                # linear interpolation in log10(m) against logp
                if prev_v == v[j]:
                    return m[j]
                f = (prev_v - target) / (prev_v - v[j])
                lg = np.log10(prev_m) + f * (np.log10(m[j]) - np.log10(prev_m))
                return 10.0 ** lg
            prev_m, prev_v = m[j], v[j]
        return m[0] if side == "lo" else m[-1]

    return cross(range(i - 1, -1, -1), "lo"), cross(range(i + 1, len(m)), "hi")


def delta_matching(mults, lp, lo, hi):
    """The delta whose crossings best reproduce a measured interval [lo, hi]."""
    best, bestd = None, None
    for d in np.linspace(0.05, 20.0, 400):
        blo, bhi = bounds_at(mults, lp, d)
        if blo is None:
            continue
        err = abs(np.log10(blo) - np.log10(lo)) + abs(np.log10(bhi) - np.log10(hi))
        if best is None or err < best:
            best, bestd = err, d
    return bestd, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="a1,a2,b3,c1,d1,b1,x2")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("systems", nargs="*", default=["C4_NoFB", "C6", "C8"])
    a = ap.parse_args()
    systems = a.systems or ["C4_NoFB", "C6", "C8"]

    if a.calibrate:
        print("=== CALIBRATION: what delta_logp reproduces the measured a1 posteriors? ===")
        print(f"{'system':<10}{'measured 95%':>22}{'fitted delta':>14}{'residual':>11}"
              f"{'delta=2 gives':>26}")
        print("-" * 84)
        fitted = []
        for s in systems:
            if s not in KNOWN_A1_POSTERIOR:
                continue
            t0 = time.time()
            m, lp = profile(s, "a1")
            lo, hi = KNOWN_A1_POSTERIOR[s]
            d, err = delta_matching(m, lp, lo, hi)
            c_lo, c_hi = bounds_at(m, lp, 2.0)
            if d is None:
                print(f"{s:<10}{f'[{lo:.4f}, {hi:.4f}]':>22}   NO FIT: profile has "
                      f"{np.isfinite(lp).sum()} finite points of {len(lp)}", flush=True)
                continue
            fitted.append(d)
            print(f"{s:<10}{f'[{lo:.4f}, {hi:.4f}]':>22}{d:>14.2f}{err:>11.4f}"
                  f"{f'[{c_lo:.4f}, {c_hi:.4f}]':>26}   ({time.time()-t0:.0f}s)", flush=True)
        if fitted:
            print("-" * 84)
            print(f"  mean fitted delta = {np.mean(fitted):.2f} "
                  f"(sd {np.std(fitted):.2f}) vs conventional 2.00")
            (HERE / "loglik_calibration.json").write_text(json.dumps(
                dict(fitted=fitted, systems=[s for s in systems if s in KNOWN_A1_POSTERIOR],
                     mean_delta=float(np.mean(fitted)), conventional=2.0), indent=2))
        return

    cal_f = HERE / "loglik_calibration.json"
    cal = json.loads(cal_f.read_text())["mean_delta"] if cal_f.exists() else 2.0
    print(f"=== PROFILE LOG-LIKELIHOOD BOUNDS ===")
    print(f"    conventional delta=2.00   calibrated delta={cal:.2f}\n")
    results = {}
    for system in systems:
        cfg = json.loads((CFG_ROOT / f"Chain {system} - a1 tightest"
                          / "solver_params.json").read_text())
        print(f"=== {system} ===", flush=True)
        for param in a.groups.split(","):
            if param not in cfg.get("scaling_groups", {}):
                print(f"  {param:<4} absent from this system")
                continue
            t0 = time.time()
            m, lp = profile(system, param)
            c = bounds_at(m, lp, 2.0)
            k = bounds_at(m, lp, cal)
            results.setdefault(param, {})[system] = dict(
                conventional=[c[0], c[1]], calibrated=[k[0], k[1]],
                peak_mult=float(m[int(np.nanargmax(lp))]),
                logp_max=float(np.nanmax(lp)),
                logp_range=float(np.nanmax(lp) - np.nanmin(lp)))
            print(f"  {param:<4} conventional [{c[0]:.4g}, {c[1]:.4g}]"
                  f"   calibrated [{k[0]:.4g}, {k[1]:.4g}]"
                  f"   peak at {m[int(np.nanargmax(lp))]:.3g}   ({time.time()-t0:.0f}s)",
                  flush=True)
        print(flush=True)

    out = HERE / f"loglik_ranges_{'-'.join(systems).replace('+','')}.json"
    out.write_text(json.dumps(dict(calibrated_delta=cal, conventional_delta=2.0,
                                   per_param=results), indent=2))
    print(f"wrote {out}\nDONE")


if __name__ == "__main__":
    main()
