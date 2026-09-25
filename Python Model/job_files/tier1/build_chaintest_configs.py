"""Configs for the chain-count test: C12, a1+c3, one per chain count.

Protocol in Notes/tier1_experiment_plan.md section 1. Every run is identical except for
`chains`, so the only thing the comparison can attribute a difference to is chain count:

  data        Data/Tier1/Chain_<system>/ (Tier-1 design: C16 Equivalents time series +
              per-species endpoints at 720 s, sigma column read straight from the files)
  parameters  a1 + c3, LogNormal [0.1, 10] with median pinned at 1 -- the pair that
              converged on every pilot system and is well separated on this design
  warmup      tune=300: a complete BlackJAX window schedule (75 fast, 25/50/100 slow,
              50 fast) at low cost
  stopping    rhat_threshold null, so nothing stops early and each run samples until its
              budget runs out; draws set far above what the budget can reach. The
              convergence knobs are still recorded (ess_per_split_chain 50 -> 100 x chains,
              min_chains_for_convergence 3) so the rule can be applied after the fact and
              carried into the Tier-1 runs.
  budget      max_sampling_hours: equal A100-equivalent SAMPLING compute per run, with
              warmup outside the cap so warmup cost is measured rather than charged
              against sampling. max_total_hours is only a runaway guard.
  seed        identical; chains differ through the sampler's own key splitting

Usage: python build_chaintest_configs.py [--system C12] [--chains 4,8,16,32,64]
       [--tune 300] [--sampling_hours 2.0]
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "Results" / "Chain Count Test"
PRIOR = {"distribution": "LogNormal", "lower": 0.1, "upper": 10.0, "mass": 0.95,
         "fixed_stat": ["median", 1.0]}


def base_config(system):
    for name in (f"Chain {system} - a1_0.1-10_no_floor", f"Chain {system} - a1 tightest nofloor-eqxnan"):
        p = ROOT / "Results" / "Chain Scaling Tests" / name / "solver_params.json"
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit(f"no no-floor a1 config for {system}")


def datasets_for(system):
    """Tier-1 datasets, with sigma read from the data's own columns (noise_model 'column'),
    so the likelihood uses exactly the sigma the noise was drawn with."""
    import pandas as pd
    d = ROOT / "Data" / "Tier1" / f"Chain_{system}"
    ts, ep = pd.read_csv(d / "time_vs_conc.csv"), pd.read_csv(d / "init_vs_final_conc.csv")
    ts_obs = [c for c in ts.columns if c.endswith("(uM)") and not c.endswith("_sigma")]
    fa = [c for c in ep.columns if re.match(r"^C\d+_FA(_unsat)? \(uM\)$", c) and not c.endswith("_sigma")]
    init_cols = [c for c in ep.columns if c.endswith(" (uM)") and c not in fa and not c.endswith("_sigma")]
    base = base_config(system)
    ts_base, ep_base = base["datasets"][0], base["datasets"][1]
    rel = f"Data/Tier1/Chain_{system}"
    return [
        {"name": f"time_vs_conc_Tier1_{system}", "dataset_type": "timeseries",
         "data_file": f"{rel}/time_vs_conc.csv",
         "observables": {c: c for c in ts_obs},
         "noise_model": "column",
         "noise_params": {"column_mapping": {c: f"{c}_sigma" for c in ts_obs}},
         "enabled": True, "time_column": "Time (s)",
         "init_cond_overrides": ts_base["init_cond_overrides"]},
        {"name": f"sweep_conc_Tier1_{system}", "dataset_type": "endpoint",
         "data_file": f"{rel}/init_vs_final_conc.csv",
         "observables": {c: c for c in fa},
         "noise_model": "column",
         "noise_params": {"column_mapping": {c: f"{c}_sigma" for c in fa}},
         "enabled": True, "time_values": [720],
         "init_cond_columns": {c[:-5]: c for c in init_cols},
         "init_cond_overrides": ep_base.get("init_cond_overrides", {})},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="C12")
    ap.add_argument("--params", default="a1,c3")
    ap.add_argument("--chains", default="4,8,16,32,64")
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--sampling_hours", type=float, default=2.0)
    ap.add_argument("--max_total_hours", type=float, default=12.0)
    a = ap.parse_args()
    params = a.params.split(",")
    base = base_config(a.system)
    datasets = datasets_for(a.system)

    for n in (int(x) for x in a.chains.split(",")):
        run = f"Chain {a.system} - {''.join(params)}_{n}chains"
        cfg = dict(base)
        cfg["free_kinetic_params"] = [{"rxn_name": None, "param_name": p, "prior_dist_params": dict(PRIOR)}
                                      for p in params]
        cfg["prior_sampling"] = {"draws": 2000, "random_seed": 0}
        cfg["posterior_sampling"] = {
            "draws": 1_000_000,          # never reached; the sampling budget ends the run
            "tune": a.tune, "chains": n, "target_accept": 0.8, "random_seed": 42,
            "rhat_threshold": None, "ess_threshold": None, "rhat_check_every": 100,
            "convergence_consecutive_checks": 2, "post_convergence_checks": 1,
            # Vehtari et al. 2021 state the ESS requirement per split chain (>= 50 each);
            # with two splits per chain that is 100 x chains -- 400 at four chains, the
            # familiar number. Scaling it keeps the standard honest as chains change.
            "ess_per_split_chain": 50,
            # One stranded chain out of four leaves three survivors; at the default floor of
            # four that would block convergence for the whole run.
            "min_chains_for_convergence": 3,
            "checkpoint_every_steps": 5,
            "max_total_hours": a.max_total_hours, "max_sampling_hours": a.sampling_hours,
        }
        cfg["datasets"] = datasets
        cfg["output_paths"] = dict(base["output_paths"])
        cfg["output_paths"]["results_save_dir"] = f"Results/Chain Count Test/{run}"
        out = RESULTS / run
        out.mkdir(parents=True, exist_ok=True)
        (out / "solver_params.json").write_text(json.dumps(cfg, indent=2))
        print(f"wrote {out / 'solver_params.json'}  (chains={n}, tune={a.tune}, "
              f"sampling budget {a.sampling_hours} A100-h)")


if __name__ == "__main__":
    main()
