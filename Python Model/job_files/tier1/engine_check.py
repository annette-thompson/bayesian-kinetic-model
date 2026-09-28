"""Second-engine check (SI): refit one Tier-1 run's exact PyMC model with NumPyro's NUTS in place
of the resumable BlackJAX sampler, and compare the two posteriors in Monte Carlo error units.

Everything but the sampler is shared: the model, priors, transforms, data and likelihood come from
the reference run's solver_params.json through the same model builder inference_runner.py uses, and
PyMC's jittered starting points are used by both. NumPyro brings its own NUTS and windowed warmup.
It cannot resume mid-run, so this needs one uninterrupted job (engine_check.sbatch, Alpine). The
chains, warmup length, draw count, target acceptance and seed are the reference run's; stranded-chain
exclusion is not applied here (none was needed in the reference run).

  python engine_check.py --run "Tier1 C8 - a1c3"        # writes "Results/Tier1/Tier1 C8 - a1c3 - numpyro"

Output, in the new run folder: posterior_samples_pm.nc, solver_params.json (the reference config
with its save folder changed) and engine_check.json, per parameter on the scale its prior is Normal
on (log for LogNormal groups): mean difference over the combined Monte Carlo error, the sd ratio,
r-hat and bulk ESS for each engine, plus divergences and wall time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "Utilities"))

import numpy as np  # noqa: E402


def _scale(post, p, log):
    x = np.asarray(post[p].values, float)
    return np.log(x) if log else x


def compare(ref, new, params):
    import arviz as az
    out = {}
    for p in params:
        log = bool(np.all(ref.posterior[p].values > 0))
        a, b = _scale(ref.posterior, p, log), _scale(new.posterior, p, log)
        mcse = [float(az.mcse(x[None] if x.ndim == 1 else x)) for x in (a, b)]
        rhat = [float(az.rhat(x)) for x in (a, b)]
        ess = [float(az.ess(x)) for x in (a, b)]
        out[p] = {"scale": "log" if log else "natural",
                  "mean_blackjax": float(a.mean()), "mean_numpyro": float(b.mean()),
                  "sd_blackjax": float(a.std()), "sd_numpyro": float(b.std()),
                  "mean_diff_over_mc_error": float((b.mean() - a.mean()) / np.hypot(*mcse)),
                  "sd_ratio_numpyro_over_blackjax": float(b.std() / a.std()),
                  "rhat": {"blackjax": rhat[0], "numpyro": rhat[1]},
                  "ess_bulk": {"blackjax": ess[0], "numpyro": ess[1]}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="Tier1 C8 - a1c3", help="reference run (BlackJAX, finalized)")
    ap.add_argument("--tag", default="numpyro", help="suffix of the new run folder")
    ap.add_argument("--draws", type=int, default=None, help="per chain (default: the reference run's count)")
    ap.add_argument("--tune", type=int, default=None, help="warmup steps (default: the reference run's)")
    a = ap.parse_args()

    import arviz as az
    import pymc as pm  # noqa: F401  (registers the model classes)
    import inference_runner as ir

    ref_dir = PROJECT / "Results" / "Tier1" / a.run
    cfg = json.loads((ref_dir / "solver_params.json").read_text())
    ps = cfg["posterior_sampling"]
    ref = az.from_netcdf(ref_dir / "posterior_samples_pm.nc")
    draws = a.draws or int(ref.posterior.sizes["draw"])
    tune = a.tune if a.tune is not None else int(ps.get("tune", 300))
    chains = int(ps.get("chains", 4))
    seed = int(ps.get("random_seed", 42))

    out_run = f"{a.run} - {a.tag}"
    out_dir = PROJECT / "Results" / "Tier1" / out_run
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = json.loads(json.dumps(cfg))
    cfg_out["output_paths"]["results_save_dir"] = f"Results/Tier1/{out_run}"
    cfg_out["posterior_sampling"]["nuts_sampler"] = "numpyro"
    (out_dir / "solver_params.json").write_text(json.dumps(cfg_out, indent=4) + "\n")

    bundle = ir._build_model_bundle(ir.import_solver_params(out_dir / "solver_params.json"))
    params = list(bundle.free_params)
    print(f"==> NumPyro NUTS on {a.run}: {chains} chains, tune {tune}, draws {draws}, "
          f"target_accept {ps.get('target_accept', 0.8)}, seed {seed}, params {params}", flush=True)
    t0 = time.time()
    # PyMC's own NumPyro path (what pm.sample(nuts_sampler="numpyro") calls), called directly because
    # pm.sample does not pass chain_method on: vectorized runs the four chains together on one GPU,
    # as the BlackJAX runs do, instead of one after another.
    from pymc.sampling.jax import sample_jax_nuts
    with bundle.pm_model:
        idata = sample_jax_nuts(draws=draws, tune=tune, chains=chains, target_accept=float(ps.get("target_accept", 0.8)),
                                random_seed=seed, initvals=ps.get("initial_values"), model=bundle.pm_model,
                                var_names=params, progressbar=False, nuts_sampler="numpyro",
                                chain_method="vectorized", compute_convergence_checks=False,
                                idata_kwargs={"log_likelihood": False})
    seconds = time.time() - t0
    idata.to_netcdf(out_dir / "posterior_samples_pm.nc")

    result = {"reference": a.run, "engine": "numpyro", "chains": chains, "tune": tune, "draws": draws,
              "seconds": round(seconds, 1),
              "divergences": int(np.asarray(idata.sample_stats["diverging"].values).sum())
              if "diverging" in idata.sample_stats else None,
              "params": compare(ref, idata, params)}
    (out_dir / "engine_check.json").write_text(json.dumps(result, indent=1) + "\n")
    for p, r in result["params"].items():
        print(f"  {p} ({r['scale']}): mean {r['mean_blackjax']:+.4f} vs {r['mean_numpyro']:+.4f} "
              f"({r['mean_diff_over_mc_error']:+.2f} MC error), sd ratio {r['sd_ratio_numpyro_over_blackjax']:.3f}, "
              f"r-hat {r['rhat']['numpyro']:.4f}, ESS {r['ess_bulk']['numpyro']:.0f}")
    print(f"==> {seconds / 3600:.2f} h, {result['divergences']} divergences; wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
