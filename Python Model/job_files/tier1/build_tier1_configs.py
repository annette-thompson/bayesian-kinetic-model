"""Tier-1 run configs: one per (system, parameter set, variant), with early stopping ON.

Run IDs and their purpose are in Notes/tier1_experiment_plan.md. `--plan` writes every
config the plan needs before the SBC and data-type runs (R0, R1, R2, R3, R5, R6, R8);
the flags below build any single one.

Data: Data/Tier1_rates/<data_name>/ (make_tier1_rate_data.py), three datasets, each with
its sigma read straight from the file (noise_model "column"), so the likelihood uses exactly
the sd the noise was drawn with:

  time series    baseline, C16 equivalents, 72-720 s
  profile        baseline, every fatty-acid species at 720 s
  initial rates  five conditions, C16 equivalents accumulated by 150 s per minute

Sampler settings (outline 2.6):
  chains 4, tune 300, target_accept 0.8
  stop at r-hat <= 1.01 AND bulk ESS >= 100 x chains (ess_per_split_chain 50), holding on
  two consecutive checks; at least 3 chains must remain after stranded-chain exclusion
  prior  LogNormal [0.1, 10] with the median pinned to 1 for multiplicative groups; for
         d-type groups (additive inside an exponential) a Normal centred on 0 whose bounds
         give the same [0.1, 10] multiplicative span, sampled on the rate-multiplier scale

The model and solver blocks come from the system's completed single-parameter ladder
config, so the per-system tolerances are the published ones.

Usage:
  python build_tier1_configs.py --plan
  python build_tier1_configs.py --system C8 --params a1,c3
  python build_tier1_configs.py --system C8 --params d1,d2 --dense
  python build_tier1_configs.py --system C8 --params a1,c3 --data_name Chain_C8_noise20
  python build_tier1_configs.py --system C8 --params a1,c3 --prior_shift_sd 2
  python build_tier1_configs.py --system C14+unsat --reactions C14+unsat+c3split --params a1,c3s,c3l
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "Results" / "Tier1"
DATA = ROOT / "Data" / "Tier1_rates"
LOGNORMAL_SPAN = (0.1, 10.0)
LOGNORMAL_SIGMA = math.log(10.0) / 1.959964   # sd of log(x) for 95% in [0.1, 10]
# d-type parameters enter as 1/exp(n*d1 + d2): no-op at 0, Normal prior. n is 12 for chains
# up to C12; d2 has no coefficient. Bounds put exp(scale * bound) = 10, the same span as
# every LogNormal group.
D_SAMPLE_SCALE = {"d1": 12.0, "d2": 1.0}
RATE_COL = "Initial Rate (uM C16 Equivalents/min)"   # FA_conc.INITIAL_RATE_NAME
BASELINE = {"C3_MalCoA": 500.0, "C2_AcCoA": 500.0, "ACP": 10.0, "NADPH": 1000.0,
            "NADH": 1000.0, "FabD": 1.0, "FabH": 1.0, "FabG": 1.0, "FabZ": 1.0, "FabI": 1.0,
            "FabF": 1.0, "FabA": 1.0, "FabB": 1.0, "TesA": 10.0}


def base_config(system):
    p = ROOT / "Results" / "Chain Scaling Tests" / f"Chain {system} - a1_0.1-10_no_floor" / "solver_params.json"
    if not p.exists():
        raise SystemExit(f"no completed ladder config for {system}: {p}")
    return json.loads(p.read_text())


def prior_for(param, shift_sd=0.0):
    """Prior for one free parameter. shift_sd moves a LogNormal prior's median up by that
    many prior sds while the truth stays at 1 (the robustness figure's misspecification
    axis); the [lower, upper] window moves with it."""
    if param.startswith("d"):
        if shift_sd:
            raise SystemExit("prior shifts are defined for LogNormal groups only")
        scale = D_SAMPLE_SCALE.get(param, 1.0)
        bound = math.log(LOGNORMAL_SPAN[1]) / scale
        return {"distribution": "Normal", "lower": -bound, "upper": bound,
                "mass": 0.95, "sample_scale": scale}
    median = math.exp(shift_sd * LOGNORMAL_SIGMA)
    return {"distribution": "LogNormal", "lower": median * LOGNORMAL_SPAN[0],
            "upper": median * LOGNORMAL_SPAN[1], "mass": 0.95, "fixed_stat": ["median", median]}


def datasets_for(data_name):
    d = DATA / data_name
    if not d.exists():
        raise SystemExit(f"missing data {d}; run make_tier1_rate_data.py first")
    rel = f"Data/Tier1_rates/{data_name}"
    prof = pd.read_csv(d / "init_vs_final_conc.csv")
    fa = [c for c in prof.columns if re.fullmatch(r"C\d+_FA(_unsat)? \(uM\)", c)]
    inputs = [c for c in prof.columns if c.endswith(" (uM)") and c not in fa]
    init_cols = {c[:-5]: c for c in inputs}
    c16 = "C16 Equivalents (uM)"
    return [
        {"name": f"timeseries_{data_name}", "dataset_type": "timeseries",
         "data_file": f"{rel}/time_vs_conc.csv",
         "observables": {c16: c16},
         "noise_model": "column", "noise_params": {"column_mapping": {c16: f"{c16}_sigma"}},
         "enabled": True, "time_column": "Time (s)",
         "init_cond_overrides": dict(BASELINE)},
        {"name": f"profile_{data_name}", "dataset_type": "endpoint",
         "data_file": f"{rel}/init_vs_final_conc.csv",
         "observables": {c: c for c in fa},
         "noise_model": "column", "noise_params": {"column_mapping": {c: f"{c}_sigma" for c in fa}},
         "enabled": True, "time_values": [720],
         "init_cond_columns": init_cols, "init_cond_overrides": {}},
        {"name": f"rates_{data_name}", "dataset_type": "endpoint",
         "data_file": f"{rel}/init_vs_rate.csv",
         "observables": {RATE_COL: RATE_COL},
         "noise_model": "column", "noise_params": {"column_mapping": {RATE_COL: f"{RATE_COL}_sigma"}},
         "enabled": True, "time_values": [150],
         "init_cond_columns": init_cols, "init_cond_overrides": {}},
    ]


def build(system, params, reactions=None, data_name=None, tag=None, dense=False,
          target_accept=0.8, rtol=None, prior_shift_sd=0.0, chains=4, tune=300,
          draws=5000, seed=42, max_total_hours=24.0):
    reactions = reactions or system
    data_name = data_name or f"Chain_{reactions}"
    cfg = base_config(system)
    cfg.pop("initial_condition_floor", None)

    # Reaction set and its scaling groups (a variant set renames groups, e.g. c3 -> c3s/c3l).
    cfg["output_paths"] = dict(cfg["output_paths"])
    cfg["output_paths"]["reactions_source"] = [
        f"Reactions/EC_FAS_ME1/{reactions}/{Path(p).name}" for p in cfg["output_paths"]["reactions_source"]]
    for p in cfg["output_paths"]["reactions_source"]:
        if not (ROOT / p).exists():
            raise SystemExit(f"missing {p}")
    sys.path.insert(0, str(ROOT / "Utilities"))
    from reaction_model_builder import discover_scaling_groups
    groups = sorted(discover_scaling_groups([ROOT / p for p in cfg["output_paths"]["reactions_source"]]))
    cfg["scaling_groups"] = {g: (0.0 if g.startswith("d") else 1.0) for g in groups}
    missing = [p for p in params if p not in groups]
    if missing:
        raise SystemExit(f"{missing} are not scaling groups of {reactions}: {groups}")

    if rtol is not None:
        cfg["ODE_stepsize_controller"] = dict(cfg["ODE_stepsize_controller"], rtol=rtol)
    cfg["calculation_module"] = "Calculation Files/Full_FAS/FA_conc.py"
    cfg["free_kinetic_params"] = [
        {"rxn_name": None, "param_name": p, "prior_dist_params": prior_for(p, prior_shift_sd)}
        for p in params]
    cfg["prior_sampling"] = {"draws": 2000, "random_seed": 0}
    cfg["posterior_sampling"] = {
        "sampler": "blackjax",
        "draws": draws, "tune": tune, "chains": chains,
        "target_accept": target_accept, "random_seed": seed,
        "is_mass_matrix_diagonal": not dense,
        "rhat_threshold": 1.01, "ess_per_split_chain": 50,
        "min_chains_for_convergence": 3, "rhat_check_every": 100,
        "convergence_consecutive_checks": 2, "post_convergence_checks": 1,
        "checkpoint_every_steps": 5, "max_total_hours": max_total_hours,
    }
    cfg["datasets"] = datasets_for(data_name)
    label = data_name[len("Chain_"):] if data_name.startswith("Chain_") else data_name
    run = f"Tier1 {label} - {''.join(params)}" + (f" - {tag}" if tag else "")
    cfg["output_paths"]["results_save_dir"] = f"Results/Tier1/{run}"
    cfg["tier1_truth"] = json.loads((DATA / data_name / "ground_truth.json").read_text())["scaling_groups"]
    return run, cfg


def write(run, cfg, dry_run=False):
    out = RESULTS / run
    if dry_run:
        print(f"[dry run] {run}")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "solver_params.json").write_text(json.dumps(cfg, indent=2))
    ps = cfg["posterior_sampling"]
    print(f"wrote {run:<58} chains={ps['chains']} tune={ps['tune']} "
          f"target_accept={ps['target_accept']} diag={ps['is_mass_matrix_diagonal']}")


def plan_runs():
    """Every config in the plan that needs no further code (R4 SBC and R7 come later)."""
    runs = [
        ("R0", dict(system="C8", params=["a1", "c3"])),
        ("R1", dict(system="C14+unsat", params=["a1", "c3"])),
        ("R2", dict(system="C14+unsat", params=["a1", "c3", "a2"])),
        ("R3", dict(system="C8", params=["d1", "d2"])),
        ("R3", dict(system="C8", params=["d1", "d2"], dense=True, tag="dense")),
        ("R3", dict(system="C14+unsat", params=["d1", "d2"])),
        # R6: split model on standard data; both models on off-grouping data. The grouped
        # model on standard data is R1.
        ("R6", dict(system="C14+unsat", reactions="C14+unsat+c3split", params=["a1", "c3s", "c3l"],
                    data_name="Chain_C14+unsat")),
        ("R6", dict(system="C14+unsat", params=["a1", "c3"], data_name="Chain_C14+unsat+c3split_c3l3")),
        ("R6", dict(system="C14+unsat", reactions="C14+unsat+c3split", params=["a1", "c3s", "c3l"],
                    data_name="Chain_C14+unsat+c3split_c3l3")),
        ("R8", dict(system="C8", params=["a1", "c3"], target_accept=0.95, tag="ta0.95")),
        ("R8", dict(system="C8", params=["a1", "c3"], rtol=1e-5, tag="rtol1e-5")),
    ]
    for pct in (5, 20, 40):
        runs.append(("R5", dict(system="C8", params=["a1", "c3"], data_name=f"Chain_C8_noise{pct}")))
    for k in (1, 2, 3, 4):
        runs.append(("R5", dict(system="C8", params=["a1", "c3"], prior_shift_sd=k,
                                tag=f"prior+{k}sd")))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="write every plan config (see plan_runs)")
    ap.add_argument("--system")
    ap.add_argument("--params", help="comma-separated, e.g. a1,c3")
    ap.add_argument("--reactions", default=None, help="variant reaction folder, e.g. C14+unsat+c3split")
    ap.add_argument("--data_name", default=None, help="folder under Data/Tier1_rates (default Chain_<reactions>)")
    ap.add_argument("--tag", default=None, help="suffix for the run folder name")
    ap.add_argument("--dense", action="store_true", help="dense mass matrix")
    ap.add_argument("--target_accept", type=float, default=0.8)
    ap.add_argument("--rtol", type=float, default=None, help="override the system's ODE rtol")
    ap.add_argument("--prior_shift_sd", type=float, default=0.0)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--draws", type=int, default=5000, help="ceiling; convergence ends the run first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_total_hours", type=float, default=24.0,
                    help="runaway guard in A100-equivalent hours")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    if a.plan:
        for rid, kw in plan_runs():
            run, cfg = build(**kw)
            print(f"{rid}  ", end="")
            write(run, cfg, a.dry_run)
        return
    if not (a.system and a.params):
        ap.error("--system and --params are required without --plan")
    run, cfg = build(a.system, [p.strip() for p in a.params.split(",")], a.reactions, a.data_name,
                     a.tag, a.dense, a.target_accept, a.rtol, a.prior_shift_sd, a.chains, a.tune,
                     a.draws, a.seed, a.max_total_hours)
    write(run, cfg, a.dry_run)


if __name__ == "__main__":
    main()
