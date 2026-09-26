"""Generate one inference config per (system, parameter) cell of the benchmark grid.

The experiment: run inference many times, each freeing exactly ONE scaling group, and
measure how the parameter's character (influence, footprint, identifiability structure)
predicts the cost of inferring it. This script builds the configs; bench_metrics.py
reads the results back out.

PRIOR CHOICE -- why tailored rather than universal

A single universal prior width would not be neutral. A parameter whose informative
range is much narrower than that width would spend most of its prior mass in flat
likelihood territory and look slow; one whose range exceeds it would be silently
truncated and look fast. Either way "draws to converge" would partly measure how well
an arbitrary width happened to suit each parameter rather than anything about the
parameter. So each gets a prior matched to its own measured range.

The tailoring is MECHANICAL, applied identically to every parameter, so it introduces
no per-parameter judgement:

    lower, upper = the USABLE range from group_range_sweep.py
                   (informative range intersected with solvable range),
                   intersected across every system in the grid
    prior        = LogNormal, median pinned at 1.0, `mass` of its probability
                   inside [lower, upper]

Because the tailoring is a rule rather than a choice, prior width becomes a MEASURED
covariate: bench_metrics.py records it, so the analysis can test directly whether
width explains cost instead of having to assume it away.

LogNormal with a pinned median requires log-symmetric bounds (upper == 1/lower); the
maxent solver degenerates otherwise. Asymmetric measured ranges are therefore reduced
to their tighter side, and the discarded side is recorded in `prior_note` so the
asymmetry is visible in the results rather than silently lost.

Usage:
    python bench_configs.py --ranges group_range_sweep.json --out-root <dir> [--dry-run]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path("/projects/anth4580/Bayesian")
CFG_ROOT = ROOT / "Results" / "Chain Scaling Tests"

# Chosen to span the predictor space rather than to pick winners -- see the Morris
# ranking. Each tests a different hypothesis about what drives inference cost.
BENCH_PARAMS = {
    "a1": "top for total production; ZERO footprint growth (4 constants at every chain length)",
    "a2": "top for chain length and unsat fraction; largest footprint; numerically fragile",
    "b3": "consistent rank 3 across all three objectives",
    "c1": "rank 4-5; single constant, single enzyme -- tests footprint vs influence",
    "d1": "additive-in-exp; provably degenerate with d2 below C14",
    "b1": "30 constants at C20 but rank 12-13 and below-noise SNR -- footprint/influence dissociation",
    "x2": "mu* 3-4 orders below a2 on every system; the null control",
}
BENCH_SYSTEMS = ["C4_NoFB", "C6", "C8"]

# Sampler settings held identical across every cell so the only thing varying is the
# parameter and the system.
SAMPLER = dict(draws=5000, tune=1000, chains=8, target_accept=0.8, random_seed=42,
               rhat_threshold=1.01, rhat_check_every=100,
               convergence_consecutive_checks=2, post_convergence_checks=1,
               rank_ecdf_prob=0.95, checkpoint_every_steps=5)
PRIOR_MASS = 0.95
PRIOR_DRAWS = 10000


def log_symmetric(lower, upper):
    """Tighten an asymmetric range to the log-symmetric one a pinned-median LogNormal
    can represent. Returns (lo, hi, note)."""
    if lower <= 0 or upper <= 0 or lower >= upper:
        return None, None, f"unusable measured range [{lower:g}, {upper:g}]"
    dlo, dhi = abs(math.log(lower)), abs(math.log(upper))
    if abs(dlo - dhi) < 1e-9:
        return lower, upper, ""
    d = min(dlo, dhi)
    lo, hi = math.exp(-d), math.exp(d)
    dropped = "upper" if dhi > dlo else "lower"
    return lo, hi, (f"measured [{lower:g}, {upper:g}] was asymmetric in log space; "
                    f"tightened to [{lo:.4g}, {hi:.4g}], {dropped} side reduced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", required=True, help="group_range_sweep.json")
    ap.add_argument("--out-root", default=str(CFG_ROOT))
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sweep = json.loads(Path(a.ranges).read_text())
    per = sweep["per_system"]

    print(f"{'system':<10}{'param':<6}{'usable range':>26}{'prior [lo,hi]':>24}  note")
    print("-" * 104)
    made, skipped = 0, 0
    for system in BENCH_SYSTEMS:
        base_path = CFG_ROOT / f"Chain {system} - a1 tightest" / "solver_params.json"
        base = json.loads(base_path.read_text())
        for param in BENCH_PARAMS:
            if param not in base.get("scaling_groups", {}):
                print(f"{system:<10}{param:<6}{'absent from this system':>26}")
                skipped += 1
                continue
            entry = per.get(param, {}).get(system)
            if entry is None:
                print(f"{system:<10}{param:<6}{'NO RANGE MEASURED':>26}   (skipped)")
                skipped += 1
                continue
            ulo, uhi = entry["usable"]
            lo, hi, note = log_symmetric(ulo, uhi)
            if lo is None:
                print(f"{system:<10}{param:<6}{f'[{ulo:g}, {uhi:g}]':>26}{'COLLAPSED':>24}  {note}")
                skipped += 1
                continue

            cfg = json.loads(json.dumps(base))          # deep copy
            cfg["free_kinetic_params"] = [dict(
                rxn_name=None, param_name=param,
                prior_dist_params=dict(distribution="LogNormal", lower=lo, upper=hi,
                                       mass=PRIOR_MASS, fixed_stat=["median", 1.0]))]
            cfg["posterior_sampling"] = dict(SAMPLER)
            cfg["prior_sampling"] = dict(draws=PRIOR_DRAWS, random_seed=0)
            save_dir = f"Results/Chain Scaling Tests/Chain {system} - {a.tag} {param}"
            cfg["output_paths"]["results_save_dir"] = save_dir
            # Provenance: what the range was before tightening, and why this prior.
            cfg["benchmark_meta"] = dict(
                system=system, parameter=param, rationale=BENCH_PARAMS[param],
                measured_informative=entry["informative"],
                measured_solvable=entry["solvable"],
                measured_usable=entry["usable"],
                prior_lower=lo, prior_upper=hi, prior_mass=PRIOR_MASS,
                prior_width_decades=round(math.log10(hi) - math.log10(lo), 4),
                prior_note=note)

            out = Path(a.out_root) / f"Chain {system} - {a.tag} {param}" / "solver_params.json"
            if not a.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(cfg, indent=2))
            made += 1
            print(f"{system:<10}{param:<6}{f'[{ulo:.4g}, {uhi:.4g}]':>26}"
                  f"{f'[{lo:.4g}, {hi:.4g}]':>24}  {note}")

    print("-" * 104)
    print(f"{made} configs {'would be' if a.dry_run else ''} written, {skipped} skipped")


if __name__ == "__main__":
    main()
