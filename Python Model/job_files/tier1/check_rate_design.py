"""Do the proposed initial-rate conditions pass the Tier-1 selection metrics?

Design under test: a C16-equivalents time series and a chain-length profile at the
reference condition (720 s), plus initial rates (150 s) at a handful of one-enzyme
perturbations. That differs from the current Tier-1 data in two ways -- the perturbations
raise enzymes as well as lowering them, and the observable is a rate rather than a set of
per-species endpoints -- so the selection metrics have to be re-applied rather than assumed.

Metrics, as in make_tier1_data.py:
  loose steps    <= 1.5x the baseline's, at the system's own tolerances
  strict steps   <= 1.5x the baseline's, re-solved at 1e-9 / 1e-11
  output floor   C16 equivalents at the read time >= 10% of the baseline's
  diversity      >= 0.2 decades from every other kept condition. The original applied this
                 to the per-species endpoint vector; the observable here is a single rate,
                 so it is applied to log10 of the C16 equivalents at 150 s.

Usage: python check_rate_design.py C12 [--out check_rate_design_C12.json]
"""
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))

import diffrax as dfrx
import jax.numpy as jnp
import numpy as np
import generate_chain_data as gcd

RATE_TIME, END_TIME = 150.0, 720.0
STRICT_RTOL, STRICT_ATOL = 1e-9, 1e-11
MAX_STEPS = 200_000
STEP_CAP, MIN_FRAC, MIN_LOG_DIFF = 1.5, 0.10, 0.2

# Every candidate worth considering, including the alternatives to the proposed set.
CANDIDATES = [
    ("reference",   {}),
    ("FabH x10",    {"FabH": 10.0}),   ("FabH x0.1",  {"FabH": 0.1}),
    ("FabF x10",    {"FabF": 10.0}),   ("FabF x0.1",  {"FabF": 0.1}),
    ("FabB x10",    {"FabB": 10.0}),   ("FabB x0.1",  {"FabB": 0.1}),
    ("TesA x0.1",   {"TesA": 1.0}),    ("TesA x3",    {"TesA": 30.0}),
]


def tolerances(system):
    """The system's own tolerances from its Tier-1 config, else the ladder default."""
    p = PROJECT / "Results" / "Tier1" / f"Tier1 {system} - a1c3" / "solver_params.json"
    if p.exists():
        m = dict(re.findall(r'"(rtol|atol)":\s*([0-9.eE+-]+)', p.read_text()))
        if "rtol" in m and "atol" in m:
            return float(m["rtol"]), float(m["atol"])
    return 1e-3, 1e-7


def solve(sys_, y0, t_end, rtol, atol):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(), t0=0.0, t1=t_end, dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(ts=jnp.asarray([t_end])),
        stepsize_controller=dfrx.PIDController(rtol=rtol, atol=atol,
                                               pcoeff=sys_.pcoeff, icoeff=sys_.icoeff,
                                               dcoeff=sys_.dcoeff),
        max_steps=MAX_STEPS, throw=False)
    steps = int(np.asarray(sol.stats["num_steps"]))
    ok = steps < MAX_STEPS and bool(sol.result == dfrx.RESULTS.successful)
    return (np.asarray(sol.ys)[0] if ok else None), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rx = PROJECT / "Reactions" / "EC_FAS_ME1" / a.system
    groups = gcd.discover_scaling_groups(rx)
    rtol, atol = tolerances(a.system)
    sys_ = gcd.ChainSystem(rx, rtol, atol,
                           scaling_group_overrides=gcd.nominal_scaling_group_overrides(sorted(groups)))
    fa = [s for s in sys_.species if re.fullmatch(r"C\d+_FA(_unsat)?", s)]
    idx = [sys_.index_of[s] for s in fa]
    wts = np.array([int(re.fullmatch(r"C(\d+)_FA(_unsat)?", s).group(1)) / 16.0 for s in fa])

    def y0_for(over):
        y = sys_.y0()
        for k, v in over.items():
            if k in sys_.index_of:
                y[sys_.index_of[k]] = v
        return y

    rows = []
    for label, over in CANDIDATES:
        y0 = y0_for(over)
        ys_l, st_l = solve(sys_, y0, RATE_TIME, rtol, atol)
        ys_s, st_s = solve(sys_, y0, RATE_TIME, STRICT_RTOL, STRICT_ATOL)
        c16 = float(ys_l[idx] @ wts) if ys_l is not None else np.nan
        rows.append({"label": label, "steps_loose": st_l, "steps_strict": st_s,
                     "c16_150s": c16, "converged": ys_l is not None and ys_s is not None})
    # the reference is also read at 720 s for the time series and the profile
    ys720_l, st720_l = solve(sys_, y0_for({}), END_TIME, rtol, atol)
    ys720_s, st720_s = solve(sys_, y0_for({}), END_TIME, STRICT_RTOL, STRICT_ATOL)

    out = {"system": a.system, "rtol": rtol, "atol": atol, "candidates": rows,
           "reference_720s": {"steps_loose": st720_l, "steps_strict": st720_s,
                              "c16": float(ys720_l[idx] @ wts) if ys720_l is not None else None,
                              "converged": ys720_l is not None and ys720_s is not None}}
    path = a.out or (HERE / f"check_rate_design_{a.system}.json")
    Path(path).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.system}: wrote {path}")


if __name__ == "__main__":
    main()
