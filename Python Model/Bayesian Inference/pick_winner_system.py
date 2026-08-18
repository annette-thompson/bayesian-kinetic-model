"""
Ranks the reaction-count-axis GPU scaling configs (Results/GPU Scaling
Tests/Test * - a2/) to help pick "the winner system" -- test 3 in the
scaling test plan, the fixed base network for tests 4-7 (data amount/type,
free-param count/type, species tracked).

Criteria (see the plan's winner-picking discussion for the full reasoning):

1. Converges within budget (mean r-hat <= RHAT_THRESHOLD, min ESS bulk >=
   ESS_THRESHOLD) -- hard filter. A system that doesn't converge here won't
   converge once tests 4-7 make it harder (more data, more free params).
2. Has enough distinct scaling groups available (not just how many were
   configured free for *this* benchmark run -- all of these configs use
   only `a2`) to support test 6's multi-point free-param ladder -- hard
   filter, MIN_SCALING_GROUPS.
3. Among survivors, lowest measured sec/draw (posterior_sampling_sec /
   (tune + draws), from the actual completed run -- not the optional
   throughput-probe estimate) -- primary ranking key, since tests 4-7
   multiply this cost across many follow-up configs.
4. Fewest divergences -- tiebreaker; a cleaner posterior is less likely to
   introduce confounding artifacts into the follow-up comparisons.

Usage (from the "Python Model" directory, needs the same jax/arviz
environment as compare_sampler_benchmarks.py -- this machine's local
checkout won't have that):
    python "Bayesian Inference/pick_winner_system.py"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "Utilities"))

from compare_sampler_benchmarks import RESULTS_DIR, main as compare_main  # noqa: E402
from reaction_model_builder import build_ode_system_from_reactions  # noqa: E402

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0
MIN_SCALING_GROUPS = 5  # need room for a several-point free-param ladder (test 6)


def _param_counts(config_path: Path) -> tuple[int | None, int | None]:
    """(n_scaling_groups, n_raw_params) available in a config's FULL reaction
    network -- not how many were configured free for the benchmark run."""
    solver_params = json.loads(config_path.read_text())
    output_paths = solver_params["output_paths"]
    path_base = solver_params.get("path_base", ".")
    base_dir = (config_path.parent / path_base).resolve()
    reactions_source = output_paths["reactions_source"]
    resolved = [
        (Path(p) if Path(p).is_absolute() else (base_dir / p)).resolve()
        for p in reactions_source
    ]
    try:
        _network, _species, params, _param_values, scaling_groups = build_ode_system_from_reactions(resolved)
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"[warn] {config_path.parent.name}: could not load reactions for param counts ({exc})")
        return None, None
    n_scaling_groups = len(scaling_groups)
    n_raw_params = len(params) - n_scaling_groups
    return n_scaling_groups, n_raw_params


def main() -> pd.DataFrame:
    df = compare_main()
    if df.empty:
        return df

    axis = df[df["config"].str.endswith(" - a2") & df["config_path"].str.contains("GPU Scaling Tests")].copy()
    if axis.empty:
        print("\nNo completed 'Results/GPU Scaling Tests/Test * - a2' runs found yet.")
        return axis

    records = []
    for _, row in axis.iterrows():
        config_path = Path(row["config_path"])
        n_scaling_groups, n_raw_params = _param_counts(config_path)

        converged = bool(row["mean_r_hat"] <= RHAT_THRESHOLD and row["min_ess_bulk"] >= ESS_THRESHOLD)
        enough_params = n_scaling_groups is not None and n_scaling_groups >= MIN_SCALING_GROUPS

        sec_per_draw = None
        n_tune, n_draws = row.get("n_tune"), row.get("n_draws")
        if row["posterior_sampling_sec"] and n_tune is not None and n_draws is not None:
            total_draws = n_tune + n_draws
            if total_draws:
                sec_per_draw = row["posterior_sampling_sec"] / total_draws

        records.append({
            "config": row["config"],
            "n_reactions": row["n_reactions"],
            "n_scaling_groups": n_scaling_groups,
            "n_raw_params": n_raw_params,
            "mean_r_hat": row["mean_r_hat"],
            "min_ess_bulk": row["min_ess_bulk"],
            "n_divergences": row["n_divergences"],
            "sec_per_draw": sec_per_draw,
            "converged": converged,
            "enough_params_for_test6": enough_params,
            "passes_hard_filters": converged and enough_params,
        })

    result = pd.DataFrame(records).sort_values(
        ["passes_hard_filters", "sec_per_draw"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    print("\n=== Winner-System Candidates (reaction-count axis) ===")
    print(result.to_string(index=False))

    survivors = result[result["passes_hard_filters"]]
    if survivors.empty:
        print(
            "\nNo config passes both hard filters (converged within budget AND "
            f">= {MIN_SCALING_GROUPS} scaling groups available). Check the table above "
            "for which filter each config failed, and consider raising the draw budget "
            "or lowering MIN_SCALING_GROUPS."
        )
    else:
        survivors = survivors.sort_values(
            ["sec_per_draw", "n_divergences"], ascending=[True, True], na_position="last"
        )
        winner = survivors.iloc[0]
        print(f"\nRecommended winner: {winner['config']}")
        print(
            f"  sec/draw={winner['sec_per_draw']:.4g}  n_divergences={winner['n_divergences']}  "
            f"mean_r_hat={winner['mean_r_hat']:.4g}  min_ess_bulk={winner['min_ess_bulk']:.1f}  "
            f"n_scaling_groups={winner['n_scaling_groups']}  n_raw_params={winner['n_raw_params']}"
        )

    output_path = RESULTS_DIR / "winner_system_candidates.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved candidate table to {output_path}")

    return result


if __name__ == "__main__":
    main()
