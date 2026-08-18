"""Time one logp+gradient eval under a given precision / tolerance / init-floor.

Run as a separate process per variant: jax_enable_x64 is a global that has to be
set before any array is created, and inference_runner turns it ON at import, so
the only clean way to test float32 is a fresh interpreter that flips it back
immediately after that import.

The init floor exists because float32 plus exactly-zero starting concentrations
is the classic way to make a stiff solve fall over: relative error control has no
scale to work with at 0, and the first steps can drive species negative. Flooring
untouched species to a small positive value gives the controller something to
measure against. Species with an explicit initial condition are left alone.

Prints one JSON line so the driver can tabulate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import statistics
import sys
import time


p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--utils", default="Utilities")
p.add_argument("--max-steps", type=int, default=None)
p.add_argument("--dt0", default=None,
               help="initial step size; 'none' lets diffrax choose it from the system scale")
p.add_argument("--fp32", action="store_true")
p.add_argument("--floor", type=float, default=0.0, help="replace 0 initial concs with this")
p.add_argument("--rtol", type=float, default=None)
p.add_argument("--atol", type=float, default=None)
p.add_argument("--label", default="")
p.add_argument("--evals", type=int, default=10)
p.add_argument("--max-eval-seconds", type=float, default=None,
               help="stop adding timed evals once cumulative time exceeds this; "
                    "a single eval on the largest system on CPU can take tens of minutes")
args = p.parse_args()
sys.path.insert(0, args.utils)

import jax                                    # noqa: E402
import inference_runner as ir                 # noqa: E402  (sets x64 True at import)

# precision is baked into args.utils now; --fp32 only labels the row

import jax.numpy as jnp                       # noqa: E402
import resumable_sampler as rs                # noqa: E402

imported = ir.import_solver_params(args.config)
# The floor is a real config key now (experiment_framework.load_experiment_bundle),
# not a monkeypatch of the loader. That matters beyond tidiness: inference_runner
# reaches the same code path, so a benchmark timing and a real inference run are
# provably measuring the same model -- which a patch applied only inside this
# script could never guarantee.
if args.floor > 0:
    imported.solver_params["initial_condition_floor"] = float(args.floor)
if args.rtol is not None:
    imported.solver_params["ODE_stepsize_controller"]["rtol"] = args.rtol
if args.atol is not None:
    imported.solver_params["ODE_stepsize_controller"]["atol"] = args.atol
if args.max_steps is not None:
    imported.solver_params["ODE_solver"]["max_steps"] = args.max_steps
if args.dt0 is not None:
    # dt0=None makes diffrax pick the first step from the problem itself. A fixed
    # dt0 applies the same first step to every initial condition regardless of
    # scale, which the dilute rows may not tolerate.
    imported.solver_params["ODE_solver"]["dt0"] = (
        None if args.dt0.lower() in ("none", "auto", "null") else float(args.dt0))

pc = imported.solver_params.get("posterior_sampling", {})
n_chains = int(pc.get("chains", 4))

out = {
    "label": args.label,
    "fp32": args.fp32,
    "floor": args.floor,
    "rtol": imported.solver_params["ODE_stepsize_controller"]["rtol"],
    "atol": imported.solver_params["ODE_stepsize_controller"]["atol"],
    "x64_enabled": jax.config.read("jax_enable_x64"),
    "utils": args.utils,
    "max_steps": imported.solver_params["ODE_solver"]["max_steps"],
    "config_label": pathlib.Path(args.config).parent.name,
    "hostname": socket.gethostname(),
    "n_chains": n_chains,
    "dt0": imported.solver_params["ODE_solver"].get("dt0"),
}

# n_reactions/n_species are the actual cost axis (enzyme count is not: each
# enzyme expands into a different number of elementary reactions). Same
# recomputation benchmark_throughput.grad_eval_probe does.
try:
    from reaction_model_builder import load_elementary_reactions
    _rx = load_elementary_reactions(imported.reactions_source)
    out["n_reactions"] = len(_rx)
    out["n_species"] = len(_rx.species)
except Exception:
    out["n_reactions"] = out["n_species"] = None

try:
    t0 = time.perf_counter()
    bundle = ir._build_model_bundle(imported)
    bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=n_chains,
                                  random_seed=int(pc.get("random_seed", 0)))
    vg = jax.jit(jax.vmap(jax.value_and_grad(bridge.logdensity_fn)))
    pos = [jnp.asarray(x) for x in bridge.initial_positions]
    val, grad = vg(pos)
    jax.block_until_ready((val, grad))
    out["build_compile_sec"] = round(time.perf_counter() - t0, 1)

    leaves = jax.tree_util.tree_leaves((val, grad))
    out["dtype"] = sorted({str(jnp.dtype(x.dtype)) for x in leaves})
    out["device"] = sorted({str(getattr(x, "device", "?")) for x in leaves})
    # Does it produce usable numbers, not just fast ones?
    out["logp"] = [round(float(v), 6) for v in jnp.ravel(val)[:4]]
    out["grad"] = [round(float(g), 6) for g in jnp.ravel(jnp.asarray(grad[0]))[:4]]
    # TRULY per-chain: evaluate each chain SEPARATELY, not by slicing the vmapped
    # result. Under vmap one bad chain turns the whole batch NaN, so per-element
    # finiteness of the vmapped gradient is all-or-nothing and tells you nothing
    # about the individual chains. Measured difference on nate CPU at
    # rtol1e-4/atol1e-8: vmapped says 0/4 finite, separate evaluation says 3/4.
    per_chain, per_chain_grad = [], []
    for i in range(n_chains):
        single = jax.tree_util.tree_map(lambda x, i=i: x[i], pos)
        try:
            _, gi = jax.value_and_grad(bridge.logdensity_fn)(single)
            gg = jnp.ravel(jnp.asarray(gi[0]))
            per_chain.append(bool(jnp.all(jnp.isfinite(gg))))
            per_chain_grad.append(round(float(gg[0]), 6) if bool(jnp.all(jnp.isfinite(gg))) else None)
        except Exception:
            per_chain.append(False); per_chain_grad.append(None)
    out["grad_finite_per_chain"] = per_chain
    out["grad_per_chain"] = per_chain_grad
    out["n_chains_finite"] = int(sum(per_chain))
    out["finite_any_chain"] = any(per_chain)
    # kept for continuity with earlier runs: the strict all-chains-finite flag
    out["finite"] = bool(jnp.all(jnp.isfinite(val)) and
                         all(bool(jnp.all(jnp.isfinite(jnp.asarray(g)))) for g in grad))

    ts = []
    _t_budget_start = time.perf_counter()
    for _ in range(args.evals):
        if (args.max_eval_seconds is not None and ts
                and (time.perf_counter() - _t_budget_start) > args.max_eval_seconds):
            break
        t0 = time.perf_counter()
        v, g = vg(pos)
        jax.block_until_ready((v, g))
        ts.append(time.perf_counter() - t0)
    out["ms_per_grad_eval"] = round(statistics.median(ts) * 1000, 1)
    out["n_evals_actual"] = len(ts)
    out["ok"] = True
except Exception as exc:                       # noqa: BLE001 - want the reason, not a traceback
    out["ok"] = False
    out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    # Walk the exception CHAIN, not just the surface exception. The visible error is
    # routinely a mask: numba cannot serialise an exception whose context holds a
    # jax.custom_vjp, so it dies with "custom_vjp.__new__() missing 1 required
    # positional argument: 'fun'" while the real failure -- e.g. "Non-finite values
    # returned by ODE forward solve" -- sits in __context__/__cause__ and is
    # discarded by str(exc). Recording only the surface cost hours of debugging the
    # gradient bridge instead of the numerics.
    chain, seen, cur = [], set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(f"{type(cur).__name__}: {str(cur)[:300]}")
        cur = cur.__cause__ or cur.__context__
    out["error_chain"] = chain
    if len(chain) > 1:
        # The outermost frame is the mask; the innermost is usually the real cause.
        out["root_error"] = chain[-1]
    import traceback as _tb
    out["traceback_tail"] = [l[:200] for l in
                             "".join(_tb.format_exception(exc)).splitlines()[-40:]]

print("RESULT " + json.dumps(out))
