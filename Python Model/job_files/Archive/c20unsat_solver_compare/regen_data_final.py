"""Regenerate the production training data for all 14 chain-length systems using
the FINAL chosen solver setting, reading each system's reactions_source and ODE
settings straight out of its own solver_params.json so the generated data cannot
drift from what the inference run will actually solve.

Two things are regenerated:

1. time_vs_conc.csv for ALL 14 systems. Every existing copy predates the d1/d2
   scaling-group fix (they were written Aug 27 - Sep 2; the fix landed Sep 3), so
   they were produced with d1=d2=1 instead of 0 -- a TesA rate error of ~440,000x
   at C12 up to ~4e12 x at C20. n_points=11 with min_observable=1e-8 drops the
   near-zero t~0 row, yielding the intended 10 points excluding (0,0).

2. init_vs_final_conc.csv for C4_NoFB ONLY. "NoFB" means No FabF / No FabB: that
   rung deliberately excludes both enzymes (and with them scaling groups x1-x4)
   because C4 *with* FabF/FabB fails via ACP sequestration -- those elongation
   enzymes bind ACP into complexes they can never resolve at a 4-carbon cap. Its
   config already lists only 7 YAMLs, but the earlier data generation passed the
   whole 9-YAML C4 directory, so its endpoint data came from the wrong (9-enzyme,
   59-species) model while inference uses the 7-enzyme, 45-species one. The other
   13 systems' lists match their directories exactly, so their endpoint data is
   already correct and is left untouched.

Endpoint rows are baseline + the 9 conditions with the fewest total steps.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import numpy as np
import pandas as pd
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd
from reaction_model_builder import build_ode_system_from_reactions

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

SYSTEMS = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
           "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]
ENDPOINT_REGEN = {"C4_NoFB"}
N_KEEP = 9
HARD_CAP = 20_000


def load_system(name):
    cfg_path = ROOT / "Results" / "Chain Scaling Tests" / f"Chain {name} - a1" / "solver_params.json"
    cfg = json.loads(cfg_path.read_text())
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    _, _, _, _, scaling_groups = build_ode_system_from_reactions(srcs)
    sys_ = gcd.ChainSystem(
        srcs, rtol=ctrl["rtol"], atol=ctrl["atol"],
        pcoeff=ctrl["pcoeff"], icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"],
        scaling_group_overrides=gcd.nominal_scaling_group_overrides(scaling_groups))
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in name else gcd.SAT_PATTERN)
    return cfg, sys_, targets, scaling_groups, ctrl


def solve_final(sys_, y0, ctrl):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(
            rtol=ctrl["rtol"], atol=ctrl["atol"],
            pcoeff=ctrl["pcoeff"], icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
        max_steps=HARD_CAP, throw=False,
    )
    ok = bool(sol.result == dfrx.RESULTS.successful) and int(sol.stats["num_steps"]) < HARD_CAP
    final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
    return final, ok


def main():
    for name in SYSTEMS:
        cfg, sys_, targets, scaling_groups, ctrl = load_system(name)
        out_dir = ROOT / "Data" / f"Chain_{name}"
        print(f"\n=== {name} === {len(sys_.species)} species, {len(targets)} target(s), "
              f"scaling groups={sorted(scaling_groups)}", flush=True)
        print(f"    solver: rtol={ctrl['rtol']:g} atol={ctrl['atol']:g} "
              f"PID=({ctrl['pcoeff']},{ctrl['icoeff']},{ctrl['dcoeff']})", flush=True)

        ts = gcd.export_timeseries(sys_, targets, out_dir, max_steps=HARD_CAP,
                                   n_points=11, min_observable=1e-8)
        print(f"    time_vs_conc.csv: {len(ts)} rows", flush=True)

        if name in ENDPOINT_REGEN:
            res = gcd.export_sweep_ranked(
                sys_, targets, out_dir, max_steps=HARD_CAP,
                min_log_diff=0.2, max_steps_relative_to_baseline=1.5,
                strict_rtol=1e-10, strict_atol=1e-12, strict_probe_max_steps=200_000,
                n_keep=N_KEEP)
            y0_base = sys_.y0()
            final, ok = solve_final(sys_, y0_base, ctrl)
            if not ok:
                raise RuntimeError(f"{name}: baseline endpoint solve did not converge")
            base_combo = [float(y0_base[sys_.index_of[n]]) for n in gcd.SWEEP_SPECIES
                          if n in sys_.index_of]
            base_out = [float(final[sys_.index_of[t]]) for t in targets]
            sweep_names = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]

            rows = [base_combo + base_out]
            for r in res["kept"][:N_KEEP]:
                rows.append(r["combo"] + r["out"])
            cols = [f"{n} (uM)" for n in sweep_names] + [f"{t} (uM)" for t in targets]
            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(out_dir / "init_vs_final_conc.csv", index=False)
            print(f"    init_vs_final_conc.csv: {len(df)} rows (baseline + {len(rows)-1}), "
                  f"cols={len(cols)}", flush=True)
            print(f"    found {res['n_found']} usable, kept {len(res['kept'])}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
