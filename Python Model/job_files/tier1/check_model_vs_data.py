"""Pre-flight check for Tier-1 configs: does the fitting model reproduce its own data?

For each config it builds the model exactly as inference_runner.py does, solves it at the
data's true scaling values, and reports three things per dataset:

  clean rel err   max |model - clean data| / |clean data|, against the noise-free copy in
                  clean/. The data were generated at tight tolerance and the fit runs at the
                  system's working tolerance, so this is the fit's own solver error; it should
                  be far below the 10% noise. A large value means data and model disagree
                  about the parameterisation, and no parameter value could fit.
  noise z         (noisy data - model) / sigma: mean and sd should be ~0 and ~1 if the sigma
                  columns describe the noise that was actually drawn (few points per dataset,
                  so expect scatter).
  logp / grad     (--grad) log posterior and its gradient at the sampler's un-jittered
                  starting point (PyMC's initial point, i.e. the prior mean -- not the truth),
                  through the full PyMC/JAX path. A non-finite gradient is a run that cannot
                  start.

The truth is taken from the config's `tier1_truth` block and translated into the model's own
groups: a split group (c3s, c3l) inherits its parent's value when the data were generated
with the group whole. When the truth cannot be expressed in the model's groups (the grouped
model fit to off-grouping data), the clean comparison is skipped -- that model cannot
reproduce those data by construction, which is the point of that run.

Usage: python check_model_vs_data.py "Tier1 C8 - a1c3" ["Tier1 C14+unsat - a1c3" ...] [--grad]
       python check_model_vs_data.py --all [--grad]
"""
import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "Utilities"))

import jax
import jax.numpy as jnp
import numpy as np

import inference_runner as ir
from experiment_framework import load_experiment_bundle
from reaction_model_builder import build_ode_system_from_reactions


def truth_in_model(truth, groups):
    """The data's truth expressed in the model's scaling groups, or None if impossible."""
    out = {}
    for g in groups:
        if g in truth:
            out[g] = truth[g]
        elif g[:-1] in truth:                       # c3s / c3l from a whole c3
            out[g] = truth[g[:-1]]
        else:
            children = {t: v for t, v in truth.items() if t[:-1] == g}
            if children and len(set(children.values())) == 1:   # whole c3 from equal halves
                out[g] = next(iter(children.values()))
            else:
                return None
    return out


def clean_config(cfg):
    c = copy.deepcopy(cfg)
    for d in c["datasets"]:
        p = Path(d["data_file"])
        d["data_file"] = str(p.parent / "clean" / p.name)
        d["noise_model"] = "relative_mean"          # clean files carry no sigma columns
        d["noise_params"] = {"frac": 0.1}
    return c


def predict(cfg_path, cfg, truth):
    imported = ir.import_solver_params(cfg_path)
    ode, species, names, values, _ = build_ode_system_from_reactions(
        imported.reactions_source, scaling_group=cfg["scaling_groups"])
    exp = load_experiment_bundle(solver_params=cfg, solver_params_file=str(cfg_path), species_names=species)
    sim = ir._build_simulator(ode, species, cfg, exp)
    theta = jnp.asarray([float(truth.get(n, values[n])) for n in names], dtype=jnp.float64)
    return np.ravel(np.asarray(sim(theta))), exp


def per_dataset(exp, arr):
    out, i = {}, 0
    for ds in exp.datasets:
        n = np.size(ds.observed_values)
        out[ds.name] = arr[i:i + n]
        i += n
    return out


def check(run, grad):
    cfg_path = ROOT / "Results" / "Tier1" / run / "solver_params.json"
    cfg = json.loads(cfg_path.read_text())
    truth = truth_in_model(cfg["tier1_truth"], cfg["scaling_groups"])
    print(f"\n=== {run}")

    pred, exp = predict(cfg_path, cfg, truth if truth is not None else cfg["scaling_groups"])
    obs = per_dataset(exp, np.ravel(exp.observed_values))
    sig = per_dataset(exp, np.ravel(exp.observed_sigma))
    pr = per_dataset(exp, pred)
    worst = 0.0
    if truth is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=cfg_path.parent, delete=False) as f:
            json.dump(clean_config(cfg), f)
            tmp = Path(f.name)
        try:
            _, exp_c = predict(tmp, clean_config(cfg), truth)
        finally:
            tmp.unlink()
        clean = per_dataset(exp_c, np.ravel(exp_c.observed_values))
    for name in obs:
        z = (obs[name] - pr[name]) / sig[name]
        line = f"  {name:<46} n={len(z):>2}  noise z mean {z.mean():+.2f} sd {z.std():.2f}"
        if truth is not None:
            rel = float(np.max(np.abs(pr[name] - clean[name]) / np.abs(clean[name])))
            worst = max(worst, rel)
            line += f"  clean rel err {rel:.1e}"
        print(line)
    if truth is None:
        print("  (truth not expressible in this model's groups -- e.g. grouped model on "
              "off-grouping data; clean check skipped)")

    if grad:
        import resumable_sampler as rs
        bundle = ir._build_model_bundle(ir.import_solver_params(cfg_path))
        bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=1, jitter=False)
        x0 = [jnp.asarray(p[0]) for p in bridge.initial_positions]
        lp, g = jax.value_and_grad(lambda xs: bridge.logdensity_fn(xs))(x0)
        gflat = np.concatenate([np.ravel(np.asarray(v)) for v in g])
        ok = bool(np.isfinite(lp)) and bool(np.all(np.isfinite(gflat)))
        print(f"  logp at start {float(lp):.3f}   grad {np.round(gflat, 3).tolist()}   "
              f"{'finite' if ok else 'NON-FINITE'}")
        if not ok:
            return False
    return worst < 1e-2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--grad", action="store_true")
    a = ap.parse_args()
    runs = sorted(p.parent.name for p in (ROOT / "Results" / "Tier1").glob("*/solver_params.json")) \
        if a.all else a.runs
    failed = []
    for r in runs:
        try:
            if not check(r, a.grad):
                failed.append(r)
        except Exception as e:                      # keep going; report at the end
            print(f"  ERROR {type(e).__name__}: {e}")
            failed.append(r)
    print(f"\n{len(runs) - len(failed)}/{len(runs)} passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
