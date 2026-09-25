"""Tier-1 synthetic data: a C16 Equivalents time series plus per-species endpoints, with noise.

Everything is simulated to 720 s: the time series spans 0-720 s and endpoints are
read at 720 s.

Per system:
  1. Endpoint conditions are selected by generate_chain_data.export_sweep_ranked from
     one-at-a-time titration candidates (make_titration_sweep: each swept species at
     0.5, 2, 0.2, 5, 0.1 and 10x baseline, everything else at baseline; --factors) with the
     passing criteria the ladder data used (the system's own solver settings from its
     config, 1.5x-baseline step cap, strict 1e-9/1e-11 re-solve, 0.2-decade diversity,
     cheapest first), plus two Tier-1 rules:
       - total C16 Equivalents at the endpoint must be >= 10% of the baseline's, so no
         condition asks the fit to match fatty acid far below the measured range;
       - the diversity filter starts from the baseline's output, so no condition can
         duplicate the baseline row (the ladder files for C14, C16+unsat, C20 and
         C20+unsat each carry the baseline twice).
  2. Noise-free files go to Data/Tier1/Chain_<sys>/clean/: the baseline row plus the 9
     cheapest usable conditions, and the baseline time series.
  3. Unless --clean_only: generate_synthetic_ground_truth.py --from_clean_dir
     --c16_equiv_timeseries --no_clip injects noise with sigma = 0.01 uM + 10% of each
     clean value (--noise_mode pointwise --noise_floor 0.01, the defaults) and writes the
     fitted files plus their sigma columns to Data/Tier1/Chain_<sys>/.

Solved without a negative-concentration floor, like every model in the project.
A system with fewer than 9 usable conditions is reported and gets no Tier-1 files.

Every candidate's outcome is written to tier1_conditions_<sys>.json and printed as a
species x factor grid.

--conditions restricts the candidates to named titrations, in the order given (step-count
ties break in that order), still through every passing criterion. The Tier-1 data uses the
9 conditions that passed on all four systems in the full 0.1-10x sweep, ranked by summed
solver steps across the four (TesA x0.1, the 10th shared pass, is the most expensive):
  FabH x0.1, ACP x0.1, ACP x10, ACP x5, C3_MalCoA x0.2, ACP x2, C3_MalCoA x0.1, TesA x5, FabF x10

Usage: python make_tier1_data.py C12 [--clean_only]
"""
import argparse
import json
import subprocess
import sys
import time
import zlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))

import numpy as np
import pandas as pd
import generate_chain_data as gcd

END_TIME = 720.0
gcd.TIME_RANGE = (0.0, END_TIME)
ROOT = gcd.project_root()
HARD_CAP = 20_000
N_KEEP = 9                  # + baseline = 10 conditions
MIN_C16_FRAC = 0.10
# Near-exact reference for the strict-tolerance check. The ladder used 1e-10/1e-12 at
# 150 s; run to 720 s that reference itself stalls between 600 and 720 s (200,000 steps,
# 80,000 rejected, floor or no floor), while 1e-9/1e-11 reaches 720 s in 1,452 steps and
# agrees with 1e-8/1e-10 to ~1e-9 relative (C8 baseline, tmp_tests/strict720.py).
STRICT_RTOL, STRICT_ATOL = 1e-9, 1e-11
NOISE_FRAC = 0.10
PY = sys.executable


def load_fa_conc():
    spec = spec_from_file_location("FA_conc", ROOT / "Calculation Files" / "Full_FAS" / "FA_conc.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRID_CODE = {"kept": "K", "usable beyond n_keep": "u", "below weighted-total floor": "c",
             "over loose step cap / non-converged": "s", "stiff at strict tolerance": "t",
             "too similar": "d"}


def print_grid(candidate_log, species, factors):
    by_label = {c["label"]: c for c in candidate_log}
    print("\n  outcome per candidate:  K kept  u usable beyond 9  c C16 Equivalents < 10% of baseline")
    print("                          s over step cap  t stiff at strict tolerance  d too similar")
    print("  " + f"{'':<11}" + "".join(f"{f'x{f:g}':>7}" for f in factors))
    for sp in species:
        row = [GRID_CODE[by_label[f"{sp} x{f:g}"]["result"]] for f in factors]
        print("  " + f"{sp:<11}" + "".join(f"{c:>7}" for c in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--clean_only", action="store_true", help="select conditions and write clean files; no noise")
    ap.add_argument("--no_files", action="store_true", help="selection report (json) only; write no data files")
    ap.add_argument("--data_root", default="Data/Tier1", help="data output root, relative to the project")
    ap.add_argument("--out_json", default=None, help="selection report path (default tier1_conditions_<sys>.json here)")
    ap.add_argument("--n_keep", type=int, default=N_KEEP, help="conditions to keep besides the baseline")
    ap.add_argument("--min_log_diff", type=float, default=0.2,
                    help="diversity rule: decades a condition must differ from every kept one and the "
                         "baseline. 0 disables it, to keep a condition that duplicates the baseline on "
                         "some systems (FabB x0.1 on the saturated ones) for a design shared across systems.")
    ap.add_argument("--conditions", default=None,
                    help='comma-separated candidate labels to use, e.g. "FabH x0.1,ACP x10" (default: all)')
    ap.add_argument("--factors", default=None,
                    help="comma-separated titration factors, nearest-first (default generate_chain_data.TITRATION_FACTORS)")
    # Per-point sigma: one column-mean sigma left 16-73% of ladder endpoint points below
    # SNR 1 while a few large values dominated, and it tied every point's sigma to which
    # conditions were in the set. The absolute floor keeps sub-detection values from
    # counting as precise and keeps sigma far above the solver's atol (1e-7).
    ap.add_argument("--noise_mode", choices=["relative_mean", "pointwise"], default="pointwise")
    ap.add_argument("--noise_floor", type=float, default=0.01, help="absolute sigma floor (uM) for pointwise")
    a = ap.parse_args()
    system = a.system
    t_start = time.time()

    cfg_path = next((p for p in (ROOT / "Results" / "Chain Scaling Tests" / f"Chain {system} - {n}" / "solver_params.json"
                                 for n in ("a1_0.1-10_no_floor", "a1 tightest nofloor-eqxnan")) if p.exists()), None)
    if cfg_path is None:
        raise SystemExit(f"no no-floor a1 config for {system}")
    cfg = json.loads(cfg_path.read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                           icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"], scaling_group_overrides=sg)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN)
    fa = load_fa_conc()
    weights = np.array([fa.c16_equiv_weight(t) for t in targets])
    print(f"=== {system}: rtol={ctrl['rtol']:g} atol={ctrl['atol']:g}  t_end={END_TIME:g} s  targets={targets}", flush=True)
    factors = tuple(float(f) for f in a.factors.split(",")) if a.factors else gcd.TITRATION_FACTORS
    candidates = gcd.make_titration_sweep(sys_, gcd.SWEEP_SPECIES, factors)
    if a.conditions:
        wanted = [c.strip() for c in a.conditions.split(",")]
        sweep_all, labels_all = candidates
        missing = [w for w in wanted if w not in labels_all]
        if missing:
            raise SystemExit(f"--conditions not in the titration sweep: {missing}")
        rows = [labels_all.index(w) for w in wanted]
        candidates = ({n: [v[i] for i in rows] for n, v in sweep_all.items()}, wanted)

    tier_dir = ROOT / a.data_root / f"Chain_{system}"
    clean_dir = tier_dir / "clean"
    stage_dir = HERE / "staging" / system
    res = gcd.export_sweep_ranked(
        sys_, targets, stage_dir, max_steps=HARD_CAP,
        min_log_diff=a.min_log_diff, max_steps_relative_to_baseline=1.5,
        strict_rtol=STRICT_RTOL, strict_atol=STRICT_ATOL, strict_probe_max_steps=200_000,
        n_keep=None, total_weights=weights, min_total_frac_of_baseline=MIN_C16_FRAC,
        seed_diversity_with_baseline=True, candidates=candidates)

    base = res["baseline"]
    base_c16 = base["weighted_total"]
    kept = res["kept"][:a.n_keep]
    kept_label_set = {k["label"] for k in kept}
    for entry in res["candidate_log"]:
        if entry["result"] == "kept" and entry["label"] not in kept_label_set:
            entry["result"] = "usable beyond n_keep"
    fracs = [float(weights @ np.asarray(k["out"])) / base_c16 for k in kept]
    summary = dict(
        system=system, rtol=ctrl["rtol"], min_c16_frac=MIN_C16_FRAC,
        usable_conditions_including_baseline=res["n_found"] + 1, needed=a.n_keep + 1,
        enough=res["n_found"] >= a.n_keep, min_log_diff=a.min_log_diff, n_non_converged=res["n_bad"],
        n_below_c16_floor=res["n_low_total"], n_too_similar=res["n_similar"],
        n_stiff_at_strict=res["n_stiff"], baseline_steps=base["steps"],
        baseline_strict_steps=base["strict_steps"], loose_cap=base["loose_cap"], strict_cap=base["strict_cap"],
        baseline_out=base["out"], targets=targets,
        baseline_c16_equivalents=base_c16, kept_steps=[k["steps"] for k in kept],
        kept_c16_frac_of_baseline=fracs, kept_max_err_pct=max((k["err_pct"] for k in kept), default=None),
        kept_labels=[k["label"] for k in kept],
        all_usable_labels=[k["label"] for k in res["kept"]],
        all_usable_c16_frac_of_baseline=[float(weights @ np.asarray(k["out"])) / base_c16 for k in res["kept"]],
        end_time_s=END_TIME, requested_conditions=a.conditions, strict_reference=[STRICT_RTOL, STRICT_ATOL], candidate_sweep="titration", titration_factors=list(factors),
        candidate_log=res["candidate_log"])

    print(f"\n  usable conditions: {res['n_found']} + baseline = {res['n_found'] + 1} (need {a.n_keep + 1})")
    print(f"  rejected: {res['n_bad']} non-converged/over step cap, {res['n_low_total']} below "
          f"{MIN_C16_FRAC:.0%} of baseline C16 Equivalents, {res['n_similar']} too similar, "
          f"{res['n_stiff']} stiff at strict tolerance")
    print(f"  baseline C16 Equivalents {base_c16:.4g} uM at {END_TIME:g} s; kept: "
          + ", ".join(f"{k['label']} ({f:.2f}x)" for k, f in zip(kept, fracs)))
    if not a.conditions:
        print_grid(res["candidate_log"], [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of], factors)

    if a.no_files:
        print("\n  --no_files: selection report only")
    elif not summary["enough"]:
        print(f"\n  NOT ENOUGH CONDITIONS: {system} gets no Tier-1 files.")
    else:
        clean_dir.mkdir(parents=True, exist_ok=True)
        sweep_names = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
        y0 = sys_.y0()
        rows = [[float(y0[sys_.index_of[n]]) for n in sweep_names] + base["out"]]
        rows += [k["combo"] + k["out"] for k in kept]
        pd.DataFrame(rows, columns=[f"{n} (uM)" for n in sweep_names] + [f"{t} (uM)" for t in targets]) \
            .to_csv(clean_dir / "init_vs_final_conc.csv", index=False)
        ts = gcd.export_timeseries(sys_, targets, clean_dir, max_steps=HARD_CAP,
                                   n_points=11, min_observable=1e-8)
        if not np.isclose(ts["Time (s)"].iloc[-1], END_TIME):
            raise RuntimeError(f"time series ends at {ts['Time (s)'].iloc[-1]} s, not {END_TIME} s")
        summary["timeseries_times_s"] = ts["Time (s)"].tolist()
        print(f"  wrote clean files to {clean_dir}")

    if summary["enough"] and not a.clean_only and not a.no_files:
        seed = zlib.crc32(system.encode()) % 2**31
        summary["noise_seed"] = seed
        cmd = [PY, str(ROOT / "Utilities" / "generate_synthetic_ground_truth.py"),
               "--from_clean_dir", str(clean_dir.relative_to(ROOT)),
               "--out_dir", str(tier_dir.relative_to(ROOT)),
               "--c16_equiv_timeseries", "--no_clip",
               "--seed", str(seed), "--noise_frac", str(NOISE_FRAC), "--noise_mode", a.noise_mode]
        if a.noise_floor is not None:
            cmd += ["--noise_floor", str(a.noise_floor)]
        summary["noise_mode"], summary["noise_floor"] = a.noise_mode, a.noise_floor
        subprocess.run(cmd, check=True)
        print(f"  wrote {tier_dir}")

    summary["seconds"] = round(time.time() - t_start, 1)
    out_json = Path(a.out_json) if a.out_json else HERE / f"tier1_conditions_{system}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
