"""Tier-1 data for the rate-based design: one time series, one profile, five initial rates.

Design (settled 2026-09-18 from the multiplier scans in scan_rate_multipliers.py):

  time series   baseline, 0-720 s, C16 equivalents
  profile       baseline, 720 s, per chain length
  initial rates 150 s, five conditions:
                    baseline
                    FabH  0.1 uM   (0.1x)   separation 0.31 dec, adds a1, c1, e
                    FabB  0   uM   (knock)  separation 0.19 dec, adds c2
                    TesA  0.5 uM   (0.05x)  separation 0.20 dec, adds a3, c3, d1, d2
                    FabZ  0   uM   (knock)  separation 0.32 dec, adds c4

Separations are the minimum across C12, C14 and C14+unsat, which are prioritised over C8;
every condition also clears the 5% output floor and the 1.5x step cap. Together they cover
12 of the 14 live scaling parameters. `b2` (FabD only) and `f` (FabA only) are not covered:
neither enzyme had a usable perturbation anywhere in 0.1-30 uM.

FabF is deliberately absent. Its live parameter set is a strict superset of FabB's only by
`e`, which FabH already covers, so it added nothing; FabB is kept instead because FabA and
FabB are the only enzymes carrying cis reactions, making FabB the sole handle on the
unsaturated branch. The x1-x4 parameters that would have been unique to FabF and FabB are
unidentifiable in this ladder anyway -- they scale the FabB*/FabF* route, which has no
release step outside the +FBinit reaction sets.

Noise follows the project convention: sigma = 10% of the clean value plus an absolute floor,
and both the noisy value and its sigma are written, so the likelihood uses exactly the sigma
the noise was drawn with.

How much the PROFILE actually carries depends on the system, because a truncated ladder
piles everything onto its terminal chain length -- share of the 720 s total in the single
largest species, and the count of species above 1% of it:

    C8  99.0%  1 species     C14        58.0%  4      C20+unsat  25.7%  9
    C12 89.1%  3 species     C14+unsat  40.8%  6

On C8 the profile is very nearly a restatement of the time series' last point, so it adds
almost no independent information, and `d1` (the TesA free-energy SLOPE) has barely any
chain-length spread to act on. The profile costs nothing extra -- it is the same 720 s
solve -- but it should not be expected to identify the chain-length-dependent parameters
below C12, which is a further reason to weight C12+ over C8.

Outputs, per system, in <out_root>/<out_name>/ (default Data/Tier1_rates/Chain_<system>/):
  time_vs_conc.csv, init_vs_final_conc.csv, init_vs_rate.csv   noisy values + _sigma columns
  clean/                                                       the same files, noise-free
  ground_truth.json                                            every scaling value, seed,
                                                               noise setting and condition used

Ground truth is solved at rtol 1e-8 / atol 1e-10 with the production PID coefficients
(0.4/0.3/0), so the data carry no solver error of their own.

Off-grouping data (the grouping test's negative control) is the same design generated from a
variant reaction set with one scaling value moved off its no-op:

  python -u make_tier1_rate_data.py C14+unsat --reactions C14+unsat+c3split \\
         --set c3l=3 --out_name Chain_C14+unsat+c3split_c3l3

Noise-level variants for the robustness figure reuse the seed, so each is the 10% dataset's
draws rescaled:

  python -u make_tier1_rate_data.py C8 --noise_frac 0.20 --out_name Chain_C8_noise20

Usage: python -u make_tier1_rate_data.py C12 [--out_root Data/Tier1_rates] [--seed 0]
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))

import diffrax as dfrx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import generate_chain_data as gcd

RATE_TIME, END_TIME = 150.0, 720.0
N_TIMEPOINTS = 10
MAX_STEPS = 200_000
NOISE_FRAC = 0.10
FLOOR_CONC = 0.01        # uM, for concentrations
FLOOR_RATE = 0.004       # uM C16/min: 0.01 uM spread over the 2.5 min window
SEC_PER_MIN = 60.0
RATE_COL = "Initial Rate (uM C16 Equivalents/min)"   # must match FA_conc.INITIAL_RATE_NAME

# label -> enzyme overrides in uM. Absolute concentrations, not multipliers: TesA's baseline
# is 10 uM, so "0.5 uM" is a 0.05x change and writing 0.5 as a multiplier would be 10x off.
CONDITIONS = [
    ("baseline",     {}),
    ("FabH 0.1 uM",  {"FabH": 0.1}),
    ("FabB 0 uM",    {"FabB": 0.0}),
    ("TesA 0.5 uM",  {"TesA": 0.5}),
    ("FabZ 0 uM",    {"FabZ": 0.0}),
]


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system")
    ap.add_argument("--out_root", default="Data/Tier1_rates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reactions", default=None,
                    help="reaction folder under Reactions/EC_FAS_ME1 (default: the system itself)")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP=VALUE",
                    help="move a scaling group off its no-op value; repeatable")
    ap.add_argument("--out_name", default=None, help="output folder name (default Chain_<system>)")
    ap.add_argument("--noise_frac", type=float, default=NOISE_FRAC,
                    help="relative part of the noise sd (default 0.10); floors are unchanged")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rx_name = a.reactions or a.system
    rx = PROJECT / "Reactions" / "EC_FAS_ME1" / rx_name
    # Ground truth is solved tightly with the production PID coefficients, so the data carry
    # no solver error of their own; the fit's tolerance is recorded for the consistency check.
    fit_rtol, fit_atol = tolerances(a.system)
    rtol, atol = a.rtol, a.atol
    groups = gcd.discover_scaling_groups(rx)
    scaling = gcd.nominal_scaling_group_overrides(sorted(groups))
    for item in a.set:
        k, v = item.split("=")
        if k not in scaling:
            raise SystemExit(f"--set {k}: not a scaling group of {rx_name} ({sorted(scaling)})")
        scaling[k] = float(v)
    sys_ = gcd.ChainSystem(rx, rtol, atol, pcoeff=0.4, icoeff=0.3, dcoeff=0.0,
                           scaling_group_overrides=scaling)
    fa = [s for s in sys_.species if re.fullmatch(r"C\d+_FA(_unsat)?", s)]
    idx = [sys_.index_of[s] for s in fa]
    wts = np.array([int(re.fullmatch(r"C(\d+)_FA(_unsat)?", s).group(1)) / 16.0 for s in fa])
    inputs = [k for k in gcd.INITIAL_CONDITIONS if k in sys_.index_of]

    def solve(over, ts):
        y = sys_.y0()
        for k, v in over.items():
            if k in sys_.index_of:
                y[sys_.index_of[k]] = v
        sol = dfrx.diffeqsolve(
            dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(), t0=0.0, t1=float(max(ts)), dt0=1e-6,
            y0=jnp.asarray(y, dtype=jnp.float64), args=sys_.theta,
            saveat=dfrx.SaveAt(ts=jnp.asarray(ts, dtype=jnp.float64)),
            stepsize_controller=dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=sys_.pcoeff,
                                                   icoeff=sys_.icoeff, dcoeff=sys_.dcoeff),
            max_steps=MAX_STEPS, throw=False)
        steps = int(np.asarray(sol.stats["num_steps"]))
        if steps >= MAX_STEPS or bool(sol.result != dfrx.RESULTS.successful):
            raise RuntimeError(f"solve failed for {over} ({steps} steps)")
        return np.asarray(sol.ys), steps

    def noisy(clean, floor):
        # Not clipped at zero: clipping biases near-zero points against the Normal likelihood.
        sigma = a.noise_frac * np.abs(clean) + floor
        return clean + rng.normal(0.0, sigma), sigma

    out = PROJECT / a.out_root / (a.out_name or f"Chain_{a.system}")
    (out / "clean").mkdir(parents=True, exist_ok=True)

    def write(frame_noisy, frame_clean, name):
        frame_noisy.to_csv(out / name, index=False)
        frame_clean.to_csv(out / "clean" / name, index=False)

    # --- time series, baseline -------------------------------------------------------
    times = np.linspace(END_TIME / N_TIMEPOINTS, END_TIME, N_TIMEPOINTS)
    ys, st_ts = solve({}, times)
    c16 = ys[:, idx] @ wts
    val, sig = noisy(c16, FLOOR_CONC)
    ts_df = pd.DataFrame({"Time (s)": times, "C16 Equivalents (uM)": val,
                          "C16 Equivalents (uM)_sigma": sig})
    write(ts_df, pd.DataFrame({"Time (s)": times, "C16 Equivalents (uM)": c16}), "time_vs_conc.csv")

    # --- profile, baseline at 720 s --------------------------------------------------
    row = {f"{k} (uM)": float(sys_.y0()[sys_.index_of[k]]) for k in inputs}
    row_clean = dict(row)
    end = ys[-1, idx]
    pval, psig = noisy(end, FLOOR_CONC)
    for sp, v, sg, c in zip(fa, pval, psig, end):
        row[f"{sp} (uM)"] = v
        row[f"{sp} (uM)_sigma"] = sg
        row_clean[f"{sp} (uM)"] = float(c)
    write(pd.DataFrame([row]), pd.DataFrame([row_clean]), "init_vs_final_conc.csv")

    # --- initial rates at 150 s ------------------------------------------------------
    rate_rows, rate_rows_clean, steps_log = [], [], []
    for label, over in CONDITIONS:
        ys_r, st = solve(over, [RATE_TIME])
        rate = float(ys_r[0, idx] @ wts) / RATE_TIME * SEC_PER_MIN
        rval, rsig = noisy(np.array([rate]), FLOOR_RATE)
        y0 = sys_.y0()
        for k, v in over.items():
            if k in sys_.index_of:
                y0[sys_.index_of[k]] = v
        r = {f"{k} (uM)": float(y0[sys_.index_of[k]]) for k in inputs}
        rate_rows_clean.append({**r, RATE_COL: rate})
        r[RATE_COL] = float(rval[0])
        r[f"{RATE_COL}_sigma"] = float(rsig[0])
        rate_rows.append(r)
        steps_log.append((label, st, rate))
    write(pd.DataFrame(rate_rows), pd.DataFrame(rate_rows_clean), "init_vs_rate.csv")

    (out / "ground_truth.json").write_text(json.dumps({
        "system": a.system,
        "reactions": f"Reactions/EC_FAS_ME1/{rx_name}",
        "scaling_groups": scaling,
        "seed": a.seed,
        "noise": {"mode": "pointwise", "frac": a.noise_frac, "floor_conc_uM": FLOOR_CONC,
                  "floor_rate_uM_C16_per_min": FLOOR_RATE, "clip_negative": False},
        "generation_solver": {"rtol": rtol, "atol": atol, "pcoeff": 0.4, "icoeff": 0.3, "dcoeff": 0.0},
        "fit_solver": {"rtol": fit_rtol, "atol": fit_atol},
        "time_series_s": [float(t) for t in times],
        "profile_time_s": END_TIME,
        "rate_time_s": RATE_TIME,
        "rate_conditions_uM": {label: over for label, over in CONDITIONS},
        "profile_species": fa,
    }, indent=2))

    print(f"{a.system} ({rx_name}): generated at rtol={rtol:g} atol={atol:g}  timeseries steps={st_ts}")
    print(f"  {'condition':<14}{'steps':>7}{'rate uM/min':>13}{'vs baseline':>13}")
    base = steps_log[0][2]
    for label, st, rate in steps_log:
        print(f"  {label:<14}{st:>7}{rate:>13.4f}{rate/base:>13.3f}")
    print(f"  wrote {out}/")


if __name__ == "__main__":
    main()
