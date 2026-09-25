"""Regenerate ONE (system, candidate) pair's chain-ladder data with the current
self-referential export_sweep_ranked search (n_keep=None, keep every survivor),
then fail loudly (nonzero exit) if fewer than --min_total_kept conditions --
kept survivors plus baseline -- were found. Used as the per-system step of a
sequential afterok job chain across the other 13 ladder rungs: if a candidate
doesn't generalize to some rung, this exits nonzero and SLURM never starts the
next system in that candidate's chain.

    python generate_and_score_ladder2.py --system C12+unsat --candidate B --min_total_kept 10
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

# system name -> reactions dir name; C4_NoFB is the one exception, its "no
# feedback" YAML set lives under the plain C4 directory.
SYSTEM_TO_RXDIR = {
    "C4_NoFB": "C4", "C6": "C6", "C8": "C8", "C10": "C10", "C12": "C12",
    "C12+unsat": "C12+unsat", "C14": "C14", "C14+unsat": "C14+unsat",
    "C16": "C16", "C16+unsat": "C16+unsat", "C18": "C18", "C18+unsat": "C18+unsat",
    "C20": "C20",
}

# Same (PID, tolerance) tuples as the C20+unsat A-Q comparison's candidates
# B, C, D, E -- the four survivors of that comparison being tested for
# generalization across the rest of the ladder.
CANDIDATES = {
    "B": ("(0.1,0.3,0) rtol=1e-6 atol=1e-6", 0.1, 0.3, 0.0, 1e-6, 1e-6),
    "C": ("(0.4,0.3,0) rtol=1e-5 atol=1e-7", 0.4, 0.3, 0.0, 1e-5, 1e-7),
    "D": ("(0.3,0.3,0) rtol=1e-5 atol=1e-7", 0.3, 0.3, 0.0, 1e-5, 1e-7),
    "E": ("(0.1,0.3,0) rtol=1e-5 atol=1e-7", 0.1, 0.3, 0.0, 1e-5, 1e-7),
    "Q": ("(0.4,0.3,0) rtol=1e-6 atol=1e-8", 0.4, 0.3, 0.0, 1e-6, 1e-8),
    "H": ("(0.4,0.3,0) rtol=1e-5 atol=1e-8", 0.4, 0.3, 0.0, 1e-5, 1e-8),
    "M": ("(0.4,0.3,0) rtol=1e-6 atol=1e-7", 0.4, 0.3, 0.0, 1e-6, 1e-7),
    "J": ("(0.1,0.3,0) rtol=1e-5 atol=1e-8", 0.1, 0.3, 0.0, 1e-5, 1e-8),
    "N": ("(0.2,0.4,0) rtol=1e-6 atol=1e-8", 0.2, 0.4, 0.0, 1e-6, 1e-8),
    "K": ("(0.3,0.3,0) rtol=1e-6 atol=1e-7", 0.3, 0.3, 0.0, 1e-6, 1e-7),
    "L": ("(0.1,0.3,0) rtol=1e-6 atol=1e-7", 0.1, 0.3, 0.0, 1e-6, 1e-7),
    "O": ("(0.1,0.3,0) rtol=1e-6 atol=1e-8", 0.1, 0.3, 0.0, 1e-6, 1e-8),
    "P": ("(0.3,0.3,0) rtol=1e-6 atol=1e-8", 0.3, 0.3, 0.0, 1e-6, 1e-8),
    "I": ("(0.3,0.3,0) rtol=1e-5 atol=1e-8", 0.3, 0.3, 0.0, 1e-5, 1e-8),
}
HARD_CAP = 20_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(SYSTEM_TO_RXDIR))
    ap.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    ap.add_argument("--min_total_kept", type=int, default=10,
                    help="Fail (nonzero exit) if kept+baseline is below this.")
    a = ap.parse_args()
    label, pcoeff, icoeff, dcoeff, rtol, atol = CANDIDATES[a.candidate]

    rx_dir = ROOT / "Reactions" / "EC_FAS_ME1" / SYSTEM_TO_RXDIR[a.system]
    out_dir = HERE / "ladder_compare_v2" / a.system / a.candidate
    _, _, _, _, scaling_groups = build_ode_system_from_reactions(rx_dir)
    sys_ = gcd.ChainSystem(rx_dir, rtol=rtol, atol=atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,
                          scaling_group_overrides=gcd.nominal_scaling_group_overrides(scaling_groups))
    pattern = gcd.UNSAT_PATTERN if "+unsat" in a.system else gcd.SAT_PATTERN
    targets = sys_.targets(pattern)
    if not targets:
        raise RuntimeError(f"no targets matched for system={a.system} pattern={pattern}")

    print(f"[{a.system}/{a.candidate}] generating with self-referential export_sweep_ranked "
          f"({label}), min_total_kept={a.min_total_kept} ...", flush=True)
    gcd.export_timeseries(sys_, targets, out_dir, max_steps=HARD_CAP, n_points=12, min_observable=1e-8)

    result = gcd.export_sweep_ranked(
        sys_, targets, out_dir, max_steps=HARD_CAP,
        min_log_diff=0.2, max_steps_relative_to_baseline=1.5,
        strict_rtol=1e-10, strict_atol=1e-12, strict_probe_max_steps=200_000,
        n_keep=None,
    )

    total_kept = len(result["kept"]) + 1  # +1 for baseline
    print(f"[{a.system}/{a.candidate}] found {result['n_found']} usable condition(s) "
          f"({result['n_bad']} non-converged, {result['n_similar']} too similar, "
          f"{result['n_stiff']} stiff at strict tolerance); total incl. baseline = {total_kept}", flush=True)

    out_json = HERE / "ladder_compare_v2_results" / f"{a.system}_{a.candidate}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(dict(system=a.system, candidate=a.candidate, label=label,
                       baseline=result["baseline"], kept=result["kept"],
                       n_found=result["n_found"], n_bad=result["n_bad"],
                       n_similar=result["n_similar"], n_stiff=result["n_stiff"],
                       summary=result["summary"], total_kept=total_kept), f, indent=2)
    print(f"[{a.system}/{a.candidate}] wrote {out_json}", flush=True)

    if total_kept < a.min_total_kept:
        print(f"[{a.system}/{a.candidate}] FAIL: total_kept={total_kept} < min_total_kept={a.min_total_kept}",
              flush=True)
        sys.exit(1)
    print(f"[{a.system}/{a.candidate}] PASS: total_kept={total_kept} >= min_total_kept={a.min_total_kept}",
          flush=True)


if __name__ == "__main__":
    main()
