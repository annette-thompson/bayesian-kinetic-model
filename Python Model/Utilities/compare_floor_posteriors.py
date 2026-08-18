"""Do different zero-concentration floors give the SAME POSTERIOR?

Gradient values differ between floors (up to ~14% at floor=1e-4), but that is the
wrong test: NUTS only uses the gradient to navigate, and two models with different
gradient magnitudes can still put the posterior in the same place. What matters is
whether the inferred parameter agrees within its own uncertainty.

Biologically the floor is not obviously an approximation either -- species are
rarely at exactly zero in vivo -- so a floored model may be the more faithful one.

Runs real blackjax NUTS (window adaptation + sampling) per floor and reports the
posterior mean/sd of the free parameter, so the floors can be compared on the
quantity that actually matters. Overlapping means within ~1 sd = same answer.
"""
from __future__ import annotations
import argparse, dataclasses, functools, sys
print = functools.partial(print, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--floors", default="0,0.0001,0.00001")
ap.add_argument("--rtol", type=float, default=1e-4)
ap.add_argument("--atol", type=float, default=1e-8)
ap.add_argument("--max-steps", type=int, default=20000)
ap.add_argument("--tune", type=int, default=30)
ap.add_argument("--draws", type=int, default=60)
a = ap.parse_args()
sys.path.insert(0, "Utilities")

import jax, jax.numpy as jnp, numpy as np       # noqa: E402
import blackjax                                  # noqa: E402
import inference_runner as ir                    # noqa: E402
import resumable_sampler as rs                   # noqa: E402

_orig_loader = ir.load_experiment_bundle

def run_floor(floor: float):
    if floor > 0:
        def _floored(*args, **kw):
            exp = _orig_loader(*args, **kw)
            cm = jnp.asarray(exp.condition_matrix_jax)
            return dataclasses.replace(exp, condition_matrix_jax=jnp.where(cm == 0, floor, cm))
        ir.load_experiment_bundle = _floored
    else:
        ir.load_experiment_bundle = _orig_loader

    imported = ir.import_solver_params(a.config)
    imported.solver_params["ODE_stepsize_controller"]["rtol"] = a.rtol
    imported.solver_params["ODE_stepsize_controller"]["atol"] = a.atol
    imported.solver_params["ODE_solver"]["max_steps"] = a.max_steps
    pc = imported.solver_params.get("posterior_sampling", {})
    bundle = ir._build_model_bundle(imported)
    bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=int(pc.get("chains", 4)),
                                  random_seed=int(pc.get("random_seed", 0)))
    logdensity = bridge.logdensity_fn
    start = jax.tree_util.tree_map(lambda x: x[0], [jnp.asarray(p) for p in bridge.initial_positions])

    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=0.8)
    (state, params), _ = warmup.run(sub, start, num_steps=a.tune)
    kernel = blackjax.nuts(logdensity, **(params if isinstance(params, dict) else params._asdict()))

    draws, divs = [], 0
    for _ in range(a.draws):
        key, sub = jax.random.split(key)
        state, info = kernel.step(sub, state)
        draws.append(float(jnp.ravel(jnp.asarray(state.position[0]))[0]))
        divs += int(getattr(info, "is_divergent", False))
    d = np.asarray(draws)
    # positions are unconstrained; exp() puts them back on the parameter's own scale
    return dict(floor=floor, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                mean_exp=float(np.exp(d).mean()), sd_exp=float(np.exp(d).std(ddof=1)),
                divergences=divs, n=len(d))

results = []
for f in [float(x) for x in a.floors.split(",")]:
    print(f"######## floor={f} ########")
    try:
        r = run_floor(f)
        results.append(r)
        print(f"[posterior] floor={r['floor']:<9g} a2 mean={r['mean_exp']:.5f} sd={r['sd_exp']:.5f} "
              f"(unconstrained {r['mean']:.5f} +/- {r['sd']:.5f})  divergences={r['divergences']}/{r['n']}")
    except Exception as exc:
        print(f"[posterior] floor={f} FAILED {type(exc).__name__}: {str(exc)[:200]}")

if len(results) > 1:
    base = results[0]
    print("")
    print("=== agreement with floor=0 (the question that matters) ===")
    for r in results[1:]:
        d_mean = abs(r["mean_exp"] - base["mean_exp"])
        pooled = (base["sd_exp"] ** 2 + r["sd_exp"] ** 2) ** 0.5
        print(f"  floor={r['floor']:<9g} shift={d_mean:.5f} = {d_mean / pooled:.2f} pooled sd  "
              f"-> {'SAME within error' if d_mean < pooled else 'DIFFERENT'}")
