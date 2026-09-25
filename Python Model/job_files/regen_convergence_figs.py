"""Redraw the two figures whose content depends on the ESS threshold, and report the shift.

`convergence_diagnostics.png` (r-hat/ESS trajectory with the threshold line and the
"criteria met" marker) and `trace_plot.png` (same marker on the trace) were both drawn
against a flat ESS >= 400. Vehtari et al. 2021 give the requirement per split chain, which
is 100 x chains, so every eight-chain run was marked converged at half the standard it
should have met. Everything else in a run's figure set -- posteriors, energy, rank ECDF,
LOO -- is independent of this threshold and is deliberately left alone.

For each run this prints the criteria-met draw under the old flat bar and the new scaled
one, so the change to each figure is recorded rather than silently applied.

Usage: python job_files/regen_convergence_figs.py [--base "Results/Chain Scaling Tests"]
       [--dry_run] [--only SUBSTRING]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/projects/anth4580/Bayesian")
sys.path.insert(0, str(ROOT / "Utilities"))

DEFAULT_BASES = ("Results/Chain Scaling Tests", "Results/Chain Count Test")
OLD_FLAT_ESS = 400.0
PROD_BLOCK = 100      # rhat_check_every: the sampler only tests on this grid
PROD_POST_BLOCKS = 1  # post_convergence_checks: one further block after the streak


def regen(run_dir, dry_run=False):
    import arviz as az
    from inference_plotting import (compute_convergence_criteria_met_at, ess_threshold_for,
                                    plot_convergence_diagnostics, plot_posterior_trace_diagnostics)
    from inference_runner import import_solver_params

    imported = import_solver_params(str(run_dir / "solver_params.json"))
    cfg = imported.solver_params.get("posterior_sampling", {})
    free_params = [p["param_name"] for p in imported.solver_params.get("free_kinetic_params", [])]
    nc = imported.results_save_dir / imported.posterior_samples_file
    if not nc.exists():
        return {"run": run_dir.name, "skipped": "no posterior netcdf"}

    inf = az.from_netcdf(nc)
    n_chains = int(inf.posterior.sizes["chain"])
    rhat_threshold = float(cfg.get("rhat_threshold") or 1.01)
    consecutive = int(cfg.get("convergence_consecutive_checks", 1))
    new_ess, source = ess_threshold_for(cfg, n_chains)

    # Both numbers on sampling draws only, so they are directly comparable.
    kw = dict(rhat_threshold=rhat_threshold, consecutive=consecutive)
    old_at = compute_convergence_criteria_met_at(inf, free_params, ess_threshold=OLD_FLAT_ESS, **kw)
    new_at = compute_convergence_criteria_met_at(inf, free_params, ess_threshold=new_ess, **kw)
    # The figures use a step-10 grid, which is finer than the sampler can actually act on.
    # CONVERGED_SAMPLING_DONE_OVERRIDE in chain_scaling_analysis.ipynb needs the draw the
    # production rule would really have stopped at: the sampler only tests every
    # rhat_check_every draws, and takes one further block after the streak completes.
    n_draws = int(inf.posterior.sizes["draw"])
    prod_kw = dict(rhat_threshold=rhat_threshold, consecutive=2, step=PROD_BLOCK)
    prod_streak_old = compute_convergence_criteria_met_at(inf, free_params, ess_threshold=OLD_FLAT_ESS, **prod_kw)
    prod_streak_new = compute_convergence_criteria_met_at(inf, free_params, ess_threshold=new_ess, **prod_kw)

    def _prod(streak_end):
        if streak_end is None:
            return None
        fired = streak_end + PROD_POST_BLOCKS * PROD_BLOCK
        return fired if fired <= n_draws else None

    rec = {"run": run_dir.name, "chains": n_chains, "ess_old": OLD_FLAT_ESS, "ess_new": new_ess,
           "ess_source": source, "criteria_at_old": old_at, "criteria_at_new": new_at,
           "production_draw_old": _prod(prod_streak_old), "production_draw_new": _prod(prod_streak_new),
           "sampling_draws": n_draws,
           "moved": bool(old_at != new_at), "figures": []}
    if dry_run:
        return rec

    plot_convergence_diagnostics(
        inf_data=inf, free_params=free_params, rhat_threshold=rhat_threshold,
        ess_threshold=new_ess, step=10,
        save_file=str(imported.results_save_dir / "convergence_diagnostics.png"),
        system_name=imported.results_save_dir.name, show=False)
    rec["figures"].append("convergence_diagnostics.png")

    n_tune = int(inf.warmup_posterior.sizes["draw"]) if hasattr(inf, "warmup_posterior") else 0
    plot_posterior_trace_diagnostics(
        inf_data=inf, free_params=free_params,
        save_file=str(imported.results_save_dir / "trace_plot.png"),
        include_tuning=n_tune > 0,
        criteria_met_at=None if new_at is None else new_at + n_tune,
        system_name=imported.results_save_dir.name, show=False, use_log_param_axis=True)
    rec["figures"].append("trace_plot.png")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="append", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = []
    for b in (a.base or DEFAULT_BASES):
        d = ROOT / b
        if d.is_dir():
            runs += sorted(x for x in d.iterdir() if (x / "solver_params.json").exists())
    if a.only:
        runs = [r for r in runs if a.only in r.name]

    out = []
    for r in runs:
        try:
            rec = regen(r, dry_run=a.dry_run)
        except Exception as e:
            rec = {"run": r.name, "error": f"{type(e).__name__}: {e}"}
        out.append(rec)
        if "skipped" in rec:
            continue
        if "error" in rec:
            print(f"  !! {rec['run'][:50]:<50} {rec['error'][:60]}", flush=True)
            continue
        flag = "MOVED" if rec["moved"] else "same "
        print(f"  {flag} {rec['run'][:44]:<44} ch={rec['chains']:<3} "
              f"ESS {rec['ess_old']:.0f}->{rec['ess_new']:.0f}  "
              f"criteria {rec['criteria_at_old']}->{rec['criteria_at_new']}  "
              f"production {rec['production_draw_old']}->{rec['production_draw_new']}", flush=True)

    done = [r for r in out if "criteria_at_new" in r]
    print(f"\n{len(done)} runs with posteriors; {sum(r['moved'] for r in done)} moved; "
          f"{sum(1 for r in out if 'error' in r)} errors; "
          f"{sum(1 for r in out if 'skipped' in r)} without a netcdf")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
