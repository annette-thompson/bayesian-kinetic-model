"""Does a NaN gradient actually stop NUTS, or does it sample anyway?

Runs real BlackJAX warmup+sampling for a short burst and reports whether the
draws MOVE. A NaN gradient does not crash blackjax: the leapfrog produces NaN
energy, the proposal is marked divergent, and the chain stays put -- so the run
completes and writes draws that are all identical. This measures that directly
instead of inferring it.

Reports, per chain: number of unique draw values, spread, and divergence rate.
"""
from __future__ import annotations
import argparse, sys, functools
print = functools.partial(print, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--utils", default="Utilities")
ap.add_argument("--rtol", type=float); ap.add_argument("--atol", type=float)
ap.add_argument("--max-steps", type=int, default=20000)
ap.add_argument("--tune", type=int, default=30)
ap.add_argument("--draws", type=int, default=30)
a = ap.parse_args()
sys.path.insert(0, a.utils)

import jax, jax.numpy as jnp, numpy as np      # noqa: E402
import blackjax                                 # noqa: E402
import inference_runner as ir                   # noqa: E402
import resumable_sampler as rs                  # noqa: E402

imported = ir.import_solver_params(a.config)
if a.rtol: imported.solver_params["ODE_stepsize_controller"]["rtol"] = a.rtol
if a.atol: imported.solver_params["ODE_stepsize_controller"]["atol"] = a.atol
imported.solver_params["ODE_solver"]["max_steps"] = a.max_steps
pc = imported.solver_params.get("posterior_sampling", {})
n_chains = int(pc.get("chains", 4))

bundle = ir._build_model_bundle(imported)
bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=n_chains,
                              random_seed=int(pc.get("random_seed", 0)))
logdensity = bridge.logdensity_fn
pos0 = [jnp.asarray(p) for p in bridge.initial_positions]

# gradient at the start, for the record
val, grad = jax.vmap(jax.value_and_grad(logdensity))(pos0)
g0 = jnp.ravel(jnp.asarray(grad[0]))
print(f"[init] logp finite={bool(jnp.all(jnp.isfinite(val)))} "
      f"grad finite={bool(jnp.all(jnp.isfinite(g0)))} grad={[float(x) for x in g0[:2]]}")

# blackjax window adaptation FIRST, then sample. A fixed step size proves
# nothing: an arbitrary value can reject every proposal even when the gradient is
# perfectly healthy, which is exactly what the previous version of this test did
# -- it reported a stuck chain for the KNOWN-GOOD gradient, so it could not tell
# the two cases apart. resumable_sampler runs window adaptation, so this does too.
single0 = jax.tree_util.tree_map(lambda x: x[0], pos0)
key = jax.random.PRNGKey(0)

warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=0.8)
key, sub = jax.random.split(key)
try:
    (state, params), _ = warmup.run(sub, single0, num_steps=a.tune)
    step_size = float(params["step_size"]) if isinstance(params, dict) else float(params.step_size)
    print(f"[warmup] adapted step_size={step_size:.3e}")
except Exception as exc:
    import traceback
    print(f"[warmup] FAILED {type(exc).__name__}: {str(exc)[:200]}")
    traceback.print_exc()
    raise SystemExit(1)

kernel = blackjax.nuts(logdensity, **(params if isinstance(params, dict) else params._asdict()))
vals, divs, accepts = [], 0, []
for i in range(a.draws):
    key, sub = jax.random.split(key)
    state, info = kernel.step(sub, state)
    vals.append(float(jnp.ravel(jnp.asarray(state.position[0]))[0]))
    divs += int(getattr(info, "is_divergent", False))
    accepts.append(float(getattr(info, "acceptance_rate", jnp.nan)))

vals = np.asarray(vals)
uniq = len(np.unique(np.round(vals, 10)))
print(f"[sampling] {len(vals)} draws, unique values={uniq}, "
      f"min={vals.min():.6f} max={vals.max():.6f} spread={vals.max()-vals.min():.3e}")
print(f"[sampling] divergences={divs}/{a.draws}  mean_accept={np.nanmean(accepts):.3f}")
print(f"[verdict] {'CHAIN IS STUCK - all draws identical' if uniq <= 1 else 'chain moves - sampling works'}")
