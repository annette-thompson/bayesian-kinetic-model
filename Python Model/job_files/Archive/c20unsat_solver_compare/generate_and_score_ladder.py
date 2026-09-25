"""Regenerate ONE (system, candidate) pair's chain-ladder data with the
strict-tolerance filter, then solve all 10 resulting conditions at both the
candidate's own PID/tolerance and a fixed tight-tolerance reference, writing
one JSON. Used to compare candidates D and E (the two that succeeded cleanly
on C20+unsat) across the other 13 ladder rungs.

    python generate_and_score_ladder.py --system C12+unsat --candidate D
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import numpy as np
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

# system name -> reactions dir name (mirrors each "Chain <name> - a1" config's
# actual reactions_source; C4_NoFB is the one exception, using the plain C4 dir).
SYSTEM_TO_RXDIR = {
    "C4_NoFB": "C4", "C6": "C6", "C8": "C8", "C10": "C10", "C12": "C12",
    "C12+unsat": "C12+unsat", "C14": "C14", "C14+unsat": "C14+unsat",
    "C16": "C16", "C16+unsat": "C16+unsat", "C18": "C18", "C18+unsat": "C18+unsat",
    "C20": "C20",
}

CANDIDATES = {
    "D": ("(0.4,0.3,0) rtol=1e-5 atol=1e-7", 0.4, 0.3, 0.0, 1e-5, 1e-7),
    "E": ("(0.1,0.3,0) rtol=1e-5 atol=1e-7", 0.1, 0.3, 0.0, 1e-5, 1e-7),
}
REFERENCE_PID = (0.3, 0.3, 0.0)
REFERENCE_RTOL, REFERENCE_ATOL = 1e-10, 1e-12
ERROR_FLOOR = 1e-6
HARD_CAP = 20_000


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
    ap.add_argument("--system", required=True, choices=list(SYSTEM_TO_RXDIR))
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    a = ap.parse_args()
    label, pcoeff, icoeff, dcoeff, rtol, atol = CANDIDATES[a.candidate]

    rx_dir = ROOT / "Reactions" / "EC_FAS_ME1" / SYSTEM_TO_RXDIR[a.system]
    out_dir = HERE / "ladder_compare" / a.system / a.candidate
    sys_ = gcd.ChainSystem(rx_dir, rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    pattern = gcd.UNSAT_PATTERN if "+unsat" in a.system else gcd.SAT_PATTERN
    targets = sys_.targets(pattern)
    if not targets:
        raise RuntimeError(f"no targets matched for system={a.system} pattern={pattern}")

    print(f"[{a.system}/{a.candidate}] generating with strict-tolerance filter at reference "
          f"{REFERENCE_PID}/{REFERENCE_RTOL}/{REFERENCE_ATOL} ...", flush=True)
    ts_df = gcd.export_timeseries(sys_, targets, out_dir, max_steps=HARD_CAP, n_points=12, min_observable=1e-8)
    sw_df, n_bad, n_sim, n_stiff = gcd.export_sweep(
        sys_, targets, out_dir, max_steps=HARD_CAP, min_log_diff=0.2,
        max_steps_relative_to_baseline=1.5,
        strict_tolerance=(*REFERENCE_PID, REFERENCE_RTOL, REFERENCE_ATOL),
        strict_max_steps=10_000,
        strict_max_steps_relative_to_baseline=1.5,
        strict_max_steps_fallback_multiplier=2.0,
        prefer_fastest_at_strict=True,
    )
    print(f"[{a.system}/{a.candidate}] kept {len(sw_df)} rows ({n_bad} non-converged, {n_sim} too similar, "
          f"{n_stiff} stiff at strict tolerance)", flush=True)

    y0_list = [("baseline", sys_.y0())]
    for i, row in sw_df.iterrows():
        y0 = sys_.y0().copy()
        for sp_name in gcd.SWEEP_SPECIES:
            if sp_name in sys_.index_of:
                y0[sys_.index_of[sp_name]] = row[f"{sp_name} (uM)"]
        y0_list.append((f"sweep{i + 1}", y0))

    results = []
    for label_, y0 in y0_list:
        cand_final, cand_steps, cand_rejected, cand_ok = solve_one(sys_, y0, pcoeff, icoeff, dcoeff, rtol, atol, HARD_CAP)
        ref_final, ref_steps, ref_rejected, ref_ok = solve_one(sys_, y0, *REFERENCE_PID, REFERENCE_RTOL, REFERENCE_ATOL, 200_000)
        err_pct = relative_error(cand_final, ref_final) * 100 if cand_ok and ref_ok else float("nan")
        print(f"[{a.system}/{a.candidate}/{label_}] cand: steps={cand_steps} rejected={cand_rejected} ok={cand_ok}  "
              f"ref: steps={ref_steps} rejected={ref_rejected} ok={ref_ok}  err={err_pct:.4f}%", flush=True)
        results.append(dict(system=a.system, candidate=a.candidate, label=label, condition=label_,
                            steps=cand_steps, rejected=cand_rejected, ok=cand_ok,
                            ref_steps=ref_steps, ref_rejected=ref_rejected, ref_ok=ref_ok,
                            err_pct=err_pct))

    out_json = HERE / "ladder_compare_results" / f"{a.system}_{a.candidate}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{a.system}/{a.candidate}] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
