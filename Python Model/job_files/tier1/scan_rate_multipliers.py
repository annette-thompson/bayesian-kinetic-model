"""Best single multiplier per enzyme for an initial-rate Tier-1 condition.

For FabH, FabF, FabB and TesA, sweep a one-significant-figure multiplier grid and score
each against the Tier-1 selection metrics, read on the initial-rate observable (C16
equivalents at 150 s):

  loose steps  <= 1.5x the reference's, at the system's own tolerances
  strict steps <= 1.5x the reference's, re-solved at 1e-9 / 1e-11
  output floor >= 10% of the reference's C16 equivalents
  separation   >= 0.2 decades from the reference, on EVERY system -- a condition that
               separates on C8 but not C14 cannot be used for a shared design

0 is included: for the redundant condensing enzymes a knockout may be the only setting
that moves the rate at all.

Usage: python scan_rate_multipliers.py C12
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

RATE_TIME = 150.0
STRICT_RTOL, STRICT_ATOL = 1e-9, 1e-11
# A condition needing more steps than this at 150 s has already failed the 1.5x rule many
# times over -- the reference costs ~80 loose and ~1300 strict -- so the only information a
# higher ceiling buys is a bigger number for something already rejected. The settings that
# blow up (FabB x50/x100, FabH x100) were costing more than the other twelve multipliers
# combined and pushed the first scan past its wall clock.
MAX_STEPS = 50_000
ENZYMES = ["FabH", "FabF", "FabB", "TesA"]
MULTIPLIERS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5,
               2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def tolerances(system):
    for rel in (f"Results/Tier1/Tier1 {system} - a1c3/solver_params.json",
                f"Results/Chain Scaling Tests/Chain {system} - a1_0.1-10_no_floor/solver_params.json"):
        p = PROJECT / rel
        if p.exists():
            m = dict(re.findall(r'"(rtol|atol)":\s*([0-9.eE+-]+)', p.read_text()))
            if "rtol" in m and "atol" in m:
                return float(m["rtol"]), float(m["atol"])
    return 1e-3, 1e-7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--multipliers", default=None,
                    help="comma-separated override of the multiplier grid")
    ap.add_argument("--enzymes", default=None, help="comma-separated subset of enzymes")
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    a = ap.parse_args()
    mult_grid = ([float(x) for x in a.multipliers.split(",")] if a.multipliers else MULTIPLIERS)
    enzymes = (a.enzymes.split(",") if a.enzymes else ENZYMES)
    rx = PROJECT / "Reactions" / "EC_FAS_ME1" / a.system
    rtol, atol = tolerances(a.system)
    groups = gcd.discover_scaling_groups(rx)
    sys_ = gcd.ChainSystem(rx, rtol, atol,
                           scaling_group_overrides=gcd.nominal_scaling_group_overrides(sorted(groups)))
    fa = [s for s in sys_.species if re.fullmatch(r"C\d+_FA(_unsat)?", s)]
    idx = [sys_.index_of[s] for s in fa]
    wts = np.array([int(re.fullmatch(r"C(\d+)_FA(_unsat)?", s).group(1)) / 16.0 for s in fa])
    base_conc = dict(gcd.INITIAL_CONDITIONS)

    def solve(over, rt, at):
        y = sys_.y0()
        for k, v in over.items():
            if k in sys_.index_of:
                y[sys_.index_of[k]] = v
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(), t0=0.0, t1=RATE_TIME, dt0=1e-6,
            y0=jnp.asarray(y, dtype=jnp.float64), args=sys_.theta,
            saveat=dfrx.SaveAt(ts=jnp.asarray([RATE_TIME])),
            stepsize_controller=dfrx.PIDController(rtol=rt, atol=at, pcoeff=sys_.pcoeff,
                                                   icoeff=sys_.icoeff, dcoeff=sys_.dcoeff),
            max_steps=MAX_STEPS, throw=False)
        steps = int(np.asarray(sol.stats["num_steps"]))
        ok = steps < MAX_STEPS and bool(sol.result == dfrx.RESULTS.successful)
        return (float(np.asarray(sol.ys)[0][idx] @ wts) if ok else np.nan), steps, ok

    ref_c16, ref_l, _ = solve({}, rtol, atol)
    _, ref_s, _ = solve({}, STRICT_RTOL, STRICT_ATOL)
    rows = []
    for enz in enzymes:
        for mult in mult_grid:
            over = {enz: base_conc[enz] * mult}
            c16, sl, ok1 = solve(over, rtol, atol)
            _, ss, ok2 = solve(over, STRICT_RTOL, STRICT_ATOL)
            rows.append({"enzyme": enz, "multiplier": mult, "c16_150s": c16,
                         "frac_of_ref": c16 / ref_c16 if ref_c16 else np.nan,
                         "sep_decades": abs(np.log10(c16 / ref_c16)) if (c16 > 0 and ref_c16 > 0) else np.inf,
                         "steps_loose_ratio": sl / ref_l, "steps_strict_ratio": ss / ref_s,
                         "converged": bool(ok1 and ok2)})
    out = {"system": a.system, "rtol": rtol, "atol": atol,
           "reference": {"c16_150s": ref_c16, "steps_loose": ref_l, "steps_strict": ref_s},
           "rows": rows}
    p = HERE / f"scan_rate_multipliers_{a.system}{a.suffix}.json"
    p.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"{a.system}: wrote {p}")


if __name__ == "__main__":
    main()
