"""Solve ONE (candidate, sweep-condition) pair at both the candidate's own
PID/tolerance settings and the fixed tight-tolerance reference, writing a
single-row JSON result. Meant to be called once per SLURM job so 45
conditions (5 candidates x 9 sweep rows each) can run as independent,
per-condition jobs on Blanca instead of one lockstep-vmapped local batch.

    python solve_one_condition.py --candidate A --index 3
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))  # job_files/c20unsat_solver_compare -> Bayesian/Utilities

import numpy as np
import jax.numpy as jnp
import diffrax as dfrx
import pandas as pd
import generate_chain_data as gcd

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()
RX_DIR = ROOT / "Reactions" / "EC_FAS_ME1" / "C20+unsat"

CANDIDATES = {
    "A": ("(0.4,0.3,0) rtol=1e-6 atol=1e-6", 0.4, 0.3, 0.0, 1e-6, 1e-6),
    "B": ("(0.1,0.3,0) rtol=1e-6 atol=1e-6", 0.1, 0.3, 0.0, 1e-6, 1e-6),
    "C": ("(0.3,0.3,0) rtol=1e-5 atol=1e-7", 0.3, 0.3, 0.0, 1e-5, 1e-7),
    "D": ("(0.4,0.3,0) rtol=1e-5 atol=1e-7", 0.4, 0.3, 0.0, 1e-5, 1e-7),
    "E": ("(0.1,0.3,0) rtol=1e-5 atol=1e-7", 0.1, 0.3, 0.0, 1e-5, 1e-7),
}
REFERENCE_PID = (0.3, 0.3, 0.0)
REFERENCE_RTOL, REFERENCE_ATOL = 1e-10, 1e-12
ERROR_FLOOR = 1e-6
HARD_CAP = 20_000
REFERENCE_MAX_STEPS = 200_000


def solve_one(sys_, y0, pcoeff, icoeff, dcoeff, rtol, atol, max_steps):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff),
        max_steps=max_steps, throw=False,
    )
    final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
    steps = int(sol.stats["num_steps"])
    rejected = int(sol.stats["num_rejected_steps"])
    ok = bool(sol.result == dfrx.RESULTS.successful) and steps < max_steps
    return final, steps, rejected, ok


def relative_error(cand_final, ref_final):
    mask = np.abs(ref_final) > ERROR_FLOOR
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(cand_final[mask] - ref_final[mask]) / np.abs(ref_final[mask])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--index", required=True, type=int, help="sweep row, 1-9")
    a = ap.parse_args()

    label, pcoeff, icoeff, dcoeff, rtol, atol = CANDIDATES[a.candidate]
    sys_ = gcd.ChainSystem(RX_DIR, rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)

    csv_path = HERE / "candidate_data" / f"{a.candidate}_init_vs_final_conc.csv"
    sw_df = pd.read_csv(csv_path)
    row = sw_df.iloc[a.index - 1]
    y0 = sys_.y0().copy()
    for sp_name in gcd.SWEEP_SPECIES:
        y0[sys_.index_of[sp_name]] = row[f"{sp_name} (uM)"]

    print(f"[{a.candidate}/sweep{a.index}] solving at candidate settings ({label}) ...", flush=True)
    cand_final, cand_steps, cand_rejected, cand_ok = solve_one(sys_, y0, pcoeff, icoeff, dcoeff, rtol, atol, HARD_CAP)
    print(f"  candidate: steps={cand_steps} rejected={cand_rejected} ok={cand_ok}", flush=True)

    print(f"[{a.candidate}/sweep{a.index}] solving at reference (1e-10/1e-12) ...", flush=True)
    ref_final, ref_steps, ref_rejected, ref_ok = solve_one(
        sys_, y0, *REFERENCE_PID, REFERENCE_RTOL, REFERENCE_ATOL, REFERENCE_MAX_STEPS)
    print(f"  reference: steps={ref_steps} rejected={ref_rejected} ok={ref_ok}", flush=True)

    err_pct = relative_error(cand_final, ref_final) * 100 if cand_ok and ref_ok else float("nan")
    result = dict(candidate=a.candidate, label=label, condition=f"sweep{a.index}",
                 steps=cand_steps, rejected=cand_rejected, ok=cand_ok,
                 ref_steps=ref_steps, ref_rejected=ref_rejected, ref_ok=ref_ok,
                 err_pct=err_pct)
    print(f"RESULT: {result}", flush=True)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{a.candidate}_sweep{a.index}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
