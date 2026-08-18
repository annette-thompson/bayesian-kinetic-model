"""
Compares timing and convergence diagnostics across every Test* scaling config
under Results/ -- both the reaction-network-size axis (FabD, FabD+FabH,
FabD+FabH+FabG, ...) and the free-parameter-count axis (a1, a1 c1, a1 a2 c1,
...). Each config's own solver_params.json (Results/Test .../solver_params.json)
points at where its outputs land via output_paths.results_save_dir, so runs are
discovered by globbing for configs rather than assuming a fixed directory
layout -- add a new Results/Test .../solver_params.json and it's picked up
automatically.

n_enzymes (how many reaction YAML files a config composes) is not the real
cost-scaling axis -- each enzyme's YAML expands (via chain templates) into a
different number of elementary reactions. n_reactions/n_species are recomputed
here directly from the reaction files via load_elementary_reactions, so the
comparison table's primary sort key is the actual reaction count, not the
enzyme count.

Run after the corresponding Alpine job chains (submitted via
submit_inference_chain.sh, backed by run_inference_segment.sh /
run_inference_segment_gpu.sh) have finalized and results have been synced back
locally (Sync/sync_from_cluster.sh). Skips any run that hasn't produced output
yet.

Usage (from the "Python Model" directory):
    python "Bayesian Inference/compare_sampler_benchmarks.py"
"""
import json
import sys
from pathlib import Path

import arviz as az
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "Results"

sys.path.insert(0, str(RESULTS_DIR.parent / "Utilities"))
from reaction_model_builder import load_elementary_reactions  # noqa: E402


def _load_run(config_path: Path) -> dict | None:
    config_label = config_path.parent.name

    with open(config_path, "r", encoding="utf-8") as fh:
        solver_params = json.load(fh)

    output_paths = solver_params.get("output_paths", {})
    results_save_dir = output_paths.get("results_save_dir")
    if not results_save_dir:
        print(f"[skip] {config_label}: solver_params.json has no output_paths.results_save_dir")
        return None

    # config_path's own parent directory is always where a run's outputs
    # actually live (solver_params.json and its sibling outputs move together
    # as a unit if Results/ gets reorganized into subfolders), so anchor on it
    # directly rather than hopping a fixed number of parents up to
    # RESULTS_DIR -- that hop count broke once configs moved a level deeper
    # under Results/CPU Scaling Tests/ and Results/GPU Scaling Tests/.
    run_dir = config_path.parent
    timing_path = run_dir / "timing.json"
    posterior_path = run_dir / str(output_paths.get("posterior_samples_file", "posterior_samples_pm.nc"))

    if not timing_path.exists() or not posterior_path.exists():
        print(f"[skip] {config_label}: run not completed yet (missing timing.json or posterior samples under {run_dir})")
        return None

    with open(timing_path, "r", encoding="utf-8") as fh:
        timing = json.load(fh)

    # n_enzymes is just how many reaction files were composed; each enzyme
    # contributes a different (chain-template-expanded) number of elementary
    # reactions, so n_reactions/n_species are the actual cost-scaling axis
    # and are recomputed here rather than trusted from a stale config field.
    reactions_source = output_paths.get("reactions_source", [])
    if isinstance(reactions_source, str):
        reactions_source = [reactions_source]
    n_enzymes = len(reactions_source)

    # Anchor on RESULTS_DIR.parent (the project root, derived from this
    # script's own location) rather than the config's own path_base field --
    # reactions_source paths ("Reactions/EC_FAS_ME1/...") never moved when
    # Results/ got reorganized, but a config's path_base can go stale from
    # that move (see path_base note in the plan), so it's not a reliable
    # anchor for resolving other paths either.
    base_dir = RESULTS_DIR.parent
    resolved_reactions_source = [
        (Path(p) if Path(p).is_absolute() else (base_dir / p)).resolve()
        for p in reactions_source
    ]
    try:
        rxns = load_elementary_reactions(resolved_reactions_source)
        n_reactions = len(rxns)
        n_species = len(rxns.species)
    except Exception as exc:  # noqa: BLE001 - reaction count is informational only
        print(f"[warn] {config_label}: could not load reactions for n_reactions/n_species ({exc})")
        n_reactions = n_species = None

    free_params = [spec["param_name"] for spec in solver_params.get("free_kinetic_params", [])]
    inf_data = az.from_netcdf(posterior_path)

    summary = az.summary(inf_data, var_names=free_params, round_to=4)
    mean_rhat = float(summary["r_hat"].mean()) if "r_hat" in summary else float("nan")
    min_ess_bulk = float(summary["ess_bulk"].min()) if "ess_bulk" in summary else float("nan")

    n_divergences = None
    if "/sample_stats" in inf_data.groups and "diverging" in inf_data.sample_stats:
        n_divergences = int(inf_data.sample_stats["diverging"].values.sum())

    posterior_sampling = solver_params.get("posterior_sampling", {})

    # Resumable BlackJAX extras: segment count from the checkpoint, and the
    # preflight throughput probe if one was run (both optional).
    segments = None
    status_path = run_dir / "checkpoint" / "status.json"
    if status_path.exists():
        try:
            segments = json.loads(status_path.read_text()).get("n_invocations")
        except (json.JSONDecodeError, OSError):
            pass

    sampling_draws_per_hr = mean_leapfrog = sec_per_gradient = None
    probe_path = run_dir / "throughput_probe.json"
    if probe_path.exists():
        try:
            probe = json.loads(probe_path.read_text())
            sampling_draws_per_hr = probe.get("sampling_draws_per_hr_per_chain")
            mean_leapfrog = probe.get("mean_leapfrog_per_draw")
            sec_per_gradient = probe.get("sampling_sec_per_gradient_eval")
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "config": config_label,
        "config_path": str(config_path),
        "n_enzymes": n_enzymes,
        "n_reactions": n_reactions,
        "n_species": n_species,
        "sampler": posterior_sampling.get("sampler") or posterior_sampling.get("nuts_sampler"),
        "free_params": ",".join(free_params),
        "n_free_params": len(free_params),
        "n_tune": posterior_sampling.get("tune"),
        "n_draws": posterior_sampling.get("draws"),
        "prior_sampling_sec": timing.get("prior_sampling_sec"),
        "posterior_sampling_sec": timing.get("posterior_sampling_sec"),
        "posterior_predictive_sec": timing.get("posterior_predictive_sec"),
        "total_sec": timing.get("total_sec"),
        "segments": segments,
        "sampling_draws_per_hr_per_chain": sampling_draws_per_hr,
        "mean_leapfrog_per_draw": mean_leapfrog,
        "sampling_sec_per_gradient_eval": sec_per_gradient,
        "mean_r_hat": mean_rhat,
        "min_ess_bulk": min_ess_bulk,
        "n_divergences": n_divergences,
    }


def main() -> pd.DataFrame:
    # Recursive glob: configs now live nested under Results/CPU Scaling
    # Tests/, Results/GPU Scaling Tests/, etc., not just directly in Results/.
    config_paths = sorted(RESULTS_DIR.glob("**/Test*/solver_params.json"))

    rows = []
    for config_path in config_paths:
        row = _load_run(config_path)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"\nNo completed runs found yet under {RESULTS_DIR} (searched recursively for Test*/solver_params.json).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["n_reactions", "n_enzymes", "n_free_params", "config"]).reset_index(drop=True)

    print("\n=== Sampler Benchmark Comparison ===")
    print(df.to_string(index=False))

    output_path = RESULTS_DIR / "sampler_benchmark_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved comparison table to {output_path}")

    return df


if __name__ == "__main__":
    main()
