"""Simulate any system over the EXPERIMENTAL condition matrix and emit matching files.

The real datasets (Data/Experimental) measure three things, none of which match what the
Tier-1 files currently carry:

  time series     C16 Equivalents (uM) vs time, baseline condition
  profile         mole fraction per FA species (uM species / uM total FA) at 720 s
  initial rate    C16 Equivalents accumulated by 150 s, divided by 150 s

and they perturb by KNOCKOUT (an enzyme set to 0), not by the 0.1x titration the Tier-1
conditions use. Validating the machinery on one set of observables and then applying it to
a different set is not the "identical, unmodified recipe" the outline promises, so this
regenerates Tier-1-style truth in exactly the experimental form.

Run on the full model (C20+unsat) it doubles as a direct model-vs-measurement check: the
same conditions, the same three quantities, nominal parameters, so any disagreement is the
model's, not the pipeline's.

Also reports solver steps per condition, since the knockout conditions are not obviously as
cheap as the titrations -- an enzyme at exactly 0 removes reactions rather than slowing
them, which can make a system easier or stiffer, and that was never measured.

Usage:
  python make_experimental_style_data.py C20+unsat --compare
  python make_experimental_style_data.py C12 --out_root Data/Tier1_exp
  python make_experimental_style_data.py C12 --steps_only
"""
import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))

import diffrax as dfrx
import jax.numpy as jnp
import numpy as np
import pandas as pd

import generate_chain_data as gcd

END_TIME = 720.0
RATE_TIME = 150.0          # the assay's own early time; the rate is C16Equiv(150)/150
MAX_STEPS = 20_000
EXPERIMENTAL = PROJECT / "Data" / "Experimental"
RATE_COL = "Initial Rate (uM C16 Equivalents/min)"


def load_fa_conc():
    spec = spec_from_file_location("FA_conc", PROJECT / "Calculation Files" / "Full_FAS" / "FA_conc.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve_at(sys_, y0, ts, max_steps=MAX_STEPS):
    """Concentrations at exactly `ts`, plus the step count. Saving at requested times
    rather than at solver steps avoids interpolating the trajectory by hand."""
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=0.0, t1=float(max(ts)), dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(ts=jnp.asarray(ts, dtype=jnp.float64)),
        stepsize_controller=dfrx.PIDController(
            rtol=sys_.rtol, atol=sys_.atol,
            pcoeff=sys_.pcoeff, icoeff=sys_.icoeff, dcoeff=sys_.dcoeff),
        max_steps=max_steps, throw=False,
    )
    steps = int(np.asarray(sol.stats["num_steps"]))
    ok = steps < max_steps and bool(sol.result == dfrx.RESULTS.successful)
    return (np.asarray(sol.ys) if ok else None), steps


def condition_rows():
    """The experimental condition matrix, as (label, {species: concentration})."""
    df = pd.read_csv(EXPERIMENTAL / "init_vs_rate.csv")
    inputs = [c for c in df.columns if c.endswith(" (uM)")]
    base = df.iloc[0]
    rows = []
    for i, r in df.iterrows():
        changed = [f"{c[:-5]}={r[c]:g}" for c in inputs if r[c] != base[c]]
        rows.append((("baseline" if not changed else " ".join(changed)),
                     {c[:-5]: float(r[c]) for c in inputs}))
    return rows, inputs


def y0_for(sys_, overrides):
    y = sys_.y0()
    for name, conc in overrides.items():
        if name in sys_.index_of:
            y[sys_.index_of[name]] = conc
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system")
    ap.add_argument("--out_root", default="Data/Tier1_exp")
    ap.add_argument("--compare", action="store_true",
                    help="print model vs the real measurements (only meaningful on the full model)")
    ap.add_argument("--steps_only", action="store_true", help="report solver steps; write nothing")
    a = ap.parse_args()

    rx_dir = PROJECT / "Reactions" / "EC_FAS_ME1" / a.system
    if not rx_dir.is_dir():
        raise SystemExit(f"no reactions directory at {rx_dir}")
    groups = gcd.discover_scaling_groups(rx_dir)
    sys_ = gcd.ChainSystem(rx_dir, 1e-8, 1e-10,
                           scaling_group_overrides=gcd.nominal_scaling_group_overrides(sorted(groups)))
    fa = load_fa_conc()
    fa_species = [s for s in sys_.species if fa.SPECIES_PATTERN and
                  __import__("re").fullmatch(fa.SPECIES_PATTERN, s)]
    weights = np.array([fa.c16_equiv_weight(s) for s in fa_species])
    fa_idx = [sys_.index_of[s] for s in fa_species]

    rows, input_cols = condition_rows()
    ts_points = pd.read_csv(EXPERIMENTAL / "time_vs_conc.csv")["Time (s)"].to_numpy()
    save_ts = np.unique(np.concatenate([[RATE_TIME, END_TIME], ts_points]))

    print(f"system {a.system}: {len(sys_.species)} species, {len(fa_species)} FA species, "
          f"{len(rows)} experimental conditions")
    print(f"\n{'condition':<34}{'steps':>8}{'C16eq@720':>12}{'rate@150':>11}")
    per_cond, step_counts = [], []
    for label, overrides in rows:
        ys, steps = solve_at(sys_, y0_for(sys_, overrides), save_ts)
        step_counts.append(steps)
        if ys is None:
            print(f"{label:<34}{steps:>8}   DID NOT CONVERGE")
            per_cond.append(None)
            continue
        c16 = ys[:, fa_idx] @ weights
        at = {float(t): i for i, t in enumerate(save_ts)}
        rate = c16[at[RATE_TIME]] / RATE_TIME * 60.0   # uM C16/min, as FA_conc reports it
        per_cond.append({"label": label, "overrides": overrides, "ys": ys, "c16": c16,
                         "rate": float(rate), "at": at})
        print(f"{label:<34}{steps:>8}{c16[at[END_TIME]]:>12.4f}{rate:>11.6f}")

    good = [p for p in per_cond if p]
    print(f"\nsolver steps over {len(rows)} experimental (knockout) conditions: "
          f"min {min(step_counts)}, median {int(np.median(step_counts))}, max {max(step_counts)}")

    # Same measurement, the conditions Tier 1 currently uses, so the two designs are
    # comparable on cost rather than assumed equivalent.
    t1 = PROJECT / "Data" / "Tier1" / f"Chain_{a.system}" / "init_vs_final_conc.csv"
    if t1.exists():
        ep = pd.read_csv(t1)
        cols = [c for c in ep.columns if c.endswith(" (uM)") and "_FA" not in c
                and not c.startswith("C16 Equiv") and not c.endswith("_sigma")]
        t1_steps = []
        for _, r in ep.iterrows():
            _, s = solve_at(sys_, y0_for(sys_, {c[:-5]: float(r[c]) for c in cols}), save_ts)
            t1_steps.append(s)
        print(f"solver steps over {len(t1_steps)} Tier-1 (0.1x titration) conditions: "
              f"min {min(t1_steps)}, median {int(np.median(t1_steps))}, max {max(t1_steps)}")
        print(f"  -> experimental design costs {np.median(step_counts) / np.median(t1_steps):.2f}x "
              f"the median steps, {sum(step_counts) / sum(t1_steps):.2f}x the total")
    else:
        print(f"(no Tier-1 conditions at {t1} to compare against)")

    if a.compare:
        print("\n=== model vs measurement (nominal parameters, no fitting) ===")
        real_rate = pd.read_csv(EXPERIMENTAL / "init_vs_rate.csv")
        rc = [c for c in real_rate.columns if c.startswith("Initial Rate")][0]
        print(f"{'condition':<34}{'model':>11}{'measured':>11}{'ratio':>8}")
        for p, (_, rr) in zip(per_cond, real_rate.iterrows()):
            if not p:
                continue
            m, o = p["rate"], float(rr[rc])
            print(f"{p['label']:<34}{m:>11.6f}{o:>11.6f}{(m / o if o else float('nan')):>8.2f}")

        real_ts = pd.read_csv(EXPERIMENTAL / "time_vs_conc.csv")
        base = per_cond[0]
        if base:
            print(f"\n{'time (s)':<12}{'model C16eq':>13}{'measured':>11}{'ratio':>8}")
            for _, rr in real_ts.iterrows():
                t = float(rr["Time (s)"])
                m = float(base["c16"][base["at"][t]]); o = float(rr["C16 Equivalents (uM)"])
                print(f"{t:<12.2f}{m:>13.4f}{o:>11.4f}{(m / o if o else float('nan')):>8.2f}")

        mf_file = EXPERIMENTAL / "init_vs_final_conc_MF.csv"
        if mf_file.exists() and base:
            real_mf = pd.read_csv(mf_file).iloc[0]
            block = base["ys"][base["at"][END_TIME], fa_idx]
            total = block.sum()
            print(f"\n{'species':<18}{'model MF':>11}{'measured':>11}{'diff':>9}")
            for s, v in zip(fa_species, block):
                col = f"{s} (MF)"
                if col in real_mf.index:
                    m, o = float(v / total) if total > 0 else 0.0, float(real_mf[col])
                    print(f"{s:<18}{m:>11.4f}{o:>11.4f}{m - o:>9.4f}")

    if a.steps_only or not good:
        return

    out = PROJECT / a.out_root / f"Chain_{a.system}"
    out.mkdir(parents=True, exist_ok=True)
    base = per_cond[0]
    pd.DataFrame({"Time (s)": ts_points,
                  fa.C16_EQUIV_NAME: [base["c16"][base["at"][float(t)]] for t in ts_points]}
                 ).to_csv(out / "time_vs_conc.csv", index=False)

    rate_rows = [{**{f"{k} (uM)": v for k, v in p["overrides"].items()}, RATE_COL: p["rate"]}
                 for p in good]
    pd.DataFrame(rate_rows).to_csv(out / "init_vs_rate.csv", index=False)

    block = base["ys"][base["at"][END_TIME], fa_idx]
    total = block.sum()
    mf_row = {**{f"{k} (uM)": v for k, v in base["overrides"].items()},
              **{f"{s}{fa.MOLE_FRACTION_SUFFIX}": float(v / total) if total > 0 else 0.0
                 for s, v in zip(fa_species, block)}}
    pd.DataFrame([mf_row]).to_csv(out / "init_vs_final_mole_fraction.csv", index=False)
    print(f"\nwrote {out}/ (time_vs_conc, init_vs_rate, init_vs_final_mole_fraction) -- "
          f"NOISE-FREE; sigma columns still to come")


if __name__ == "__main__":
    main()
