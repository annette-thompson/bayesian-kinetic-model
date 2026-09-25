"""Regenerate C20+unsat data for ONE explicit (pcoeff, icoeff, dcoeff, rtol, atol)
combination using the current self-referential export_sweep_ranked search
(n_keep=None, keep every survivor). Meant to run on a GPU (A100 or similar with
real fp64 throughput -- confirmed via a standalone test that jax_enable_x64
correctly yields a CudaDevice with float64 dtype when --gres=gpu:a100:1 is
requested). Takes PID/tolerance directly as CLI floats instead of a lettered
CANDIDATES lookup, since this run covers a full 4x9=36 grid.

    python generate_and_score_gpu.py --pcoeff 0.4 --icoeff 0.3 --dcoeff 0.0 --rtol 1e-5 --atol 1e-7
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import generate_chain_data as gcd
from reaction_model_builder import build_ode_system_from_reactions

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()
RX_DIR = ROOT / "Reactions" / "EC_FAS_ME1" / "C20+unsat"
HARD_CAP = 20_000


def label_for(pcoeff, icoeff, dcoeff, rtol, atol):
    return f"p{pcoeff:g}_i{icoeff:g}_d{dcoeff:g}_rtol{rtol:.0e}_atol{atol:.0e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcoeff", type=float, required=True)
    ap.add_argument("--icoeff", type=float, required=True)
    ap.add_argument("--dcoeff", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, required=True)
    ap.add_argument("--atol", type=float, required=True)
    ap.add_argument("--n_keep", type=int, default=None,
                    help="Keep only the top N (by steps) surviving conditions; omit to keep all.")
    a = ap.parse_args()
    label = label_for(a.pcoeff, a.icoeff, a.dcoeff, a.rtol, a.atol)

    out_dir = HERE / "candidate_data_gpu" / label
    _, _, _, _, scaling_groups = build_ode_system_from_reactions(RX_DIR)
    sys_ = gcd.ChainSystem(RX_DIR, rtol=a.rtol, atol=a.atol, pcoeff=a.pcoeff, icoeff=a.icoeff, dcoeff=a.dcoeff,
                          scaling_group_overrides=gcd.nominal_scaling_group_overrides(scaling_groups))
    targets = sys_.targets(gcd.UNSAT_PATTERN)

    print(f"[{label}] generating with self-referential export_sweep_ranked (n_keep={a.n_keep}) ...", flush=True)
    gcd.export_timeseries(sys_, targets, out_dir, max_steps=HARD_CAP, n_points=12, min_observable=1e-8)

    result = gcd.export_sweep_ranked(
        sys_, targets, out_dir, max_steps=HARD_CAP,
        min_log_diff=0.2, max_steps_relative_to_baseline=1.5,
        strict_rtol=1e-10, strict_atol=1e-12, strict_probe_max_steps=200_000,
        n_keep=a.n_keep,
    )

    out_json = HERE / "results_gpu" / f"{label}.json"
    out_json.parent.mkdir(exist_ok=True, parents=True)
    with open(out_json, "w") as f:
        json.dump(dict(label=label, pcoeff=a.pcoeff, icoeff=a.icoeff, dcoeff=a.dcoeff,
                       rtol=a.rtol, atol=a.atol, baseline=result["baseline"],
                       kept=result["kept"], n_found=result["n_found"], n_bad=result["n_bad"],
                       n_similar=result["n_similar"], n_stiff=result["n_stiff"],
                       summary=result["summary"]), f, indent=2)
    print(f"[{label}] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
