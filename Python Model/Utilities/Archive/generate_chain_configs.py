"""Write one solver_params.json per chain-length rung.

Replaces nine hand-copied configs with a loop over one template. The copies had already
drifted: the 3-enzyme config declares FabD/FabH/FabG = 0.05 in its timeseries
init_cond_overrides while its data was generated at 0.01, so the config and its own
training data described different experiments. Generating both from the same constants
(``generate_chain_data.INITIAL_CONDITIONS``) makes that class of drift impossible.

    python Utilities/generate_chain_configs.py --systems C4
    python Utilities/generate_chain_configs.py             # every rung that has data

Only rungs whose CSVs already exist are written -- a config pointing at missing data
fails deep inside model building with a much less obvious message.

Stdlib only apart from the shared constants, so this can run without jax.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_chain_systems import SAT_CAPS, UNSAT_CAPS, variant_dir, project_root  # noqa: E402

ENZYMES = ["FabD", "FabH", "FabG", "FabZ", "FabI", "TesA", "FabF", "FabA", "FabB"]

# Kept identical to the existing scaling configs so timings stay comparable: one free
# scaling parameter, same prior bounds, same solver and sampler settings.
FREE_PARAMS = [{
    "rxn_name": None, "param_name": "a2",
    "prior_dist_params": {"distribution": "LogNormal", "lower": 0.01, "upper": 100.0,
                          "mass": 0.95, "fixed_stat": None},
}]
PRIOR_SAMPLING = {"draws": 100000, "random_seed": 0}
POSTERIOR_SAMPLING = {"draws": 5000, "tune": 1000, "chains": 4,
                      "target_accept": 0.8, "random_seed": 42}
ODE_SOLVER = {"solver_name": "Kvaerno5", "max_steps": 20000, "dt0": 1e-06,
              "stepsize_controller": "PIDController"}
STEPSIZE_CONTROLLER = {"rtol": 0.0001, "atol": 1e-08,
                       "pcoeff": 0.2, "icoeff": 0.4, "dcoeff": 0.0}
NOISE = {"noise_model": "relative_mean", "noise_params": {"frac": 0.1}}

# The FA_conc module selects species by regex (^C(\d+)_FA(_unsat)?$), so it adapts to
# whatever the network contains. One module serves all 14 rungs; no per-rung copies.
CALCULATION_MODULE = "Calculation Files/Full_FAS/FA_conc.py"

CONFIG_ROOT = "Results/Chain Scaling Tests"
ENDPOINT_TIME = 720


def observable_columns(csv_path: Path) -> list[str]:
    """FA columns actually present in the generated data, in chain order.

    Read from the CSV rather than predicted from the cap: the data is ground truth for
    what this network can make, and a config naming a column its CSV lacks fails late.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    cols = [c for c in header if re.fullmatch(r"C\d+_FA(_unsat)? \(uM\)", c)]
    return sorted(cols, key=lambda c: (int(re.match(r"C(\d+)", c).group(1)), "unsat" in c))


def build_config(cap: int, unsat: bool, data_dir: str, observables: list[str],
                 init_conditions: dict) -> dict:
    label = variant_dir(cap, unsat)
    reactions = [f"Reactions/EC_FAS_ME1/{label}/{e}.yaml" for e in ENZYMES]
    obs_map = {c: c for c in observables}
    # Cofactors are held fixed rather than swept, matching every existing config.
    swept = [s for s in init_conditions if s not in ("NADPH", "NADH")]

    return {
        "free_kinetic_params": FREE_PARAMS,
        "prior_sampling": PRIOR_SAMPLING,
        "posterior_sampling": POSTERIOR_SAMPLING,
        "ODE_solver": ODE_SOLVER,
        "ODE_stepsize_controller": STEPSIZE_CONTROLLER,
        "calculation_module": CALCULATION_MODULE,
        "datasets": [
            {
                "name": f"time_vs_conc_Chain_{label}",
                "dataset_type": "timeseries",
                "data_file": f"Data/{data_dir}/time_vs_conc.csv",
                "observables": obs_map,
                **NOISE,
                "enabled": True,
                "time_column": "Time (s)",
                "init_cond_overrides": dict(init_conditions),
            },
            {
                "name": f"sweep_conc_Chain_{label}",
                "dataset_type": "endpoint",
                "data_file": f"Data/{data_dir}/init_vs_final_conc.csv",
                "observables": obs_map,
                **NOISE,
                "enabled": True,
                "time_values": [ENDPOINT_TIME],
                "init_cond_columns": {s: f"{s} (uM)" for s in swept},
                "init_cond_overrides": {"NADPH": init_conditions["NADPH"],
                                        "NADH": init_conditions["NADH"]},
            },
        ],
        # Three levels up from Results/Chain Scaling Tests/<label>/ is "Python Model".
        # Every data_file and reactions_source above resolves against this.
        "path_base": "../../..",
        "output_paths": {
            "reactions_source": reactions,
            "results_save_dir": f"{CONFIG_ROOT}/Chain {label} - a2",
            "prior_samples_file": "prior_samples_pm.nc",
            "posterior_samples_file": "posterior_samples_pm.nc",
            "trace_plot_file": "trace_plot.png",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None, help="comma-separated rungs, e.g. C4,C12+unsat")
    a = ap.parse_args()

    # Imported here so --help works without jax on the path.
    from generate_chain_data import INITIAL_CONDITIONS, data_dir_name

    rungs = [(c, False) for c in SAT_CAPS] + [(c, True) for c in UNSAT_CAPS]
    if a.systems:
        want = {s.strip() for s in a.systems.split(",")}
        rungs = [r for r in rungs if variant_dir(*r) in want]

    root = project_root()
    written = skipped = 0
    for cap, unsat in rungs:
        label = variant_dir(cap, unsat)
        data_dir = data_dir_name(cap, unsat)
        ts = root / "Data" / data_dir / "time_vs_conc.csv"
        ep = root / "Data" / data_dir / "init_vs_final_conc.csv"
        if not (ts.exists() and ep.exists()):
            print(f"  {label:<12} skipped -- no data yet (run generate_chain_data.py --systems {label})")
            skipped += 1
            continue

        observables = observable_columns(ts)
        cfg = build_config(cap, unsat, data_dir, observables, INITIAL_CONDITIONS)
        out = root / CONFIG_ROOT / f"Chain {label} - a2" / "solver_params.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")

        missing = [s for s in cfg["output_paths"]["reactions_source"] if not (root / s).exists()]
        flag = f"  MISSING REACTIONS: {missing}" if missing else ""
        print(f"  {label:<12} -> {out.parent.name}   observables={len(observables)} {observables}{flag}")
        written += 1

    print(f"\n  {written} configs written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
