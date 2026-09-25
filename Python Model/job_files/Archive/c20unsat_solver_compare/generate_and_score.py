"""Regenerate ONE candidate's C20+unsat data using the new two-phase search
(generate_chain_data.export_sweep_ranked): exhaustively find every sweep
condition that converges at the candidate's own tolerance and is diverse from
what's already kept, THEN strict-tolerance-filter (self-referential: same PID,
tight rtol/atol) that whole pool, rank by steps, and keep the top --n_keep (or
all of them if omitted). No forced count -- reports however many actually
survive, so a candidate that only finds a few doesn't fail the whole job.

    python generate_and_score.py --candidate A                # keep every survivor
    python generate_and_score.py --candidate A --n_keep 9      # keep the 9 fastest
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

CANDIDATES = {
    # Every (PID, tolerance) combo from the 60-point baseline grid search with
    # rejected% < 10, relettered A-Q sorted by baseline steps ascending (ties
    # broken by max_rel_err ascending).
    "A": ("(0.4,0.3,0) rtol=1e-6 atol=1e-6", 0.4, 0.3, 0.0, 1e-6, 1e-6),    # 144 steps, err=0.0293%
    "B": ("(0.1,0.3,0) rtol=1e-6 atol=1e-6", 0.1, 0.3, 0.0, 1e-6, 1e-6),    # 145 steps, err=0.0381%
    "C": ("(0.4,0.3,0) rtol=1e-5 atol=1e-7", 0.4, 0.3, 0.0, 1e-5, 1e-7),    # 148 steps, err=0.0241%
    "D": ("(0.3,0.3,0) rtol=1e-5 atol=1e-7", 0.3, 0.3, 0.0, 1e-5, 1e-7),    # 148 steps, err=0.0469%
    "E": ("(0.1,0.3,0) rtol=1e-5 atol=1e-7", 0.1, 0.3, 0.0, 1e-5, 1e-7),    # 149 steps, err=0.0477%
    "F": ("(0.3,0.3,0) rtol=1e-4 atol=1e-8", 0.3, 0.3, 0.0, 1e-4, 1e-8),    # 155 steps, err=0.0530%, rejected=5.16%
    "G": ("(0.2,0.4,0) rtol=1e-6 atol=1e-7", 0.2, 0.4, 0.0, 1e-6, 1e-7),    # 162 steps, err=0.0088%
    "H": ("(0.4,0.3,0) rtol=1e-5 atol=1e-8", 0.4, 0.3, 0.0, 1e-5, 1e-8),    # 167 steps, err=0.0151%
    "I": ("(0.3,0.3,0) rtol=1e-5 atol=1e-8", 0.3, 0.3, 0.0, 1e-5, 1e-8),    # 167 steps, err=0.0185%
    "J": ("(0.1,0.3,0) rtol=1e-5 atol=1e-8", 0.1, 0.3, 0.0, 1e-5, 1e-8),    # 167 steps, err=0.0211%
    "K": ("(0.3,0.3,0) rtol=1e-6 atol=1e-7", 0.3, 0.3, 0.0, 1e-6, 1e-7),    # 189 steps, err=0.0032%
    "L": ("(0.1,0.3,0) rtol=1e-6 atol=1e-7", 0.1, 0.3, 0.0, 1e-6, 1e-7),    # 189 steps, err=0.0056%
    "M": ("(0.4,0.3,0) rtol=1e-6 atol=1e-7", 0.4, 0.3, 0.0, 1e-6, 1e-7),    # 189 steps, err=0.0092%
    "N": ("(0.2,0.4,0) rtol=1e-6 atol=1e-8", 0.2, 0.4, 0.0, 1e-6, 1e-8),    # 200 steps, err=0.0045%
    "O": ("(0.1,0.3,0) rtol=1e-6 atol=1e-8", 0.1, 0.3, 0.0, 1e-6, 1e-8),    # 234 steps, err=0.0035%
    "P": ("(0.3,0.3,0) rtol=1e-6 atol=1e-8", 0.3, 0.3, 0.0, 1e-6, 1e-8),    # 234 steps, err=0.0047%
    "Q": ("(0.4,0.3,0) rtol=1e-6 atol=1e-8", 0.4, 0.3, 0.0, 1e-6, 1e-8),    # 234 steps, err=0.0047%
}
HARD_CAP = 20_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--n_keep", type=int, default=None,
                    help="Keep only the top N (by steps) surviving conditions; omit to keep all.")
    a = ap.parse_args()
    label, pcoeff, icoeff, dcoeff, rtol, atol = CANDIDATES[a.candidate]

    out_dir = HERE / "candidate_data_v2" / a.candidate
    _, _, _, _, scaling_groups = build_ode_system_from_reactions(RX_DIR)
    sys_ = gcd.ChainSystem(RX_DIR, rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                          scaling_group_overrides=gcd.nominal_scaling_group_overrides(scaling_groups))
    targets = sys_.targets(gcd.UNSAT_PATTERN)

    print(f"[{a.candidate}] generating with two-phase search (self-referential strict tolerance, "
          f"n_keep={a.n_keep}) ...", flush=True)
    ts_df = gcd.export_timeseries(sys_, targets, out_dir, max_steps=HARD_CAP, n_points=12, min_observable=1e-8)

    result = gcd.export_sweep_ranked(
        sys_, targets, out_dir, max_steps=HARD_CAP,
        min_log_diff=0.2, max_steps_relative_to_baseline=1.5,
        strict_rtol=1e-10, strict_atol=1e-12, strict_probe_max_steps=200_000,
        n_keep=a.n_keep,
    )

    out_json = HERE / "results_v2" / f"{a.candidate}.json"
    out_json.parent.mkdir(exist_ok=True, parents=True)
    with open(out_json, "w") as f:
        json.dump(dict(candidate=a.candidate, label=label, baseline=result["baseline"],
                       kept=result["kept"], n_found=result["n_found"], n_bad=result["n_bad"],
                       n_similar=result["n_similar"], n_stiff=result["n_stiff"],
                       summary=result["summary"]), f, indent=2)
    print(f"[{a.candidate}] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
