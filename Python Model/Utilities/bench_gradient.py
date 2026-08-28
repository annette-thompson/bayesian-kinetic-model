"""Per-chain gradient finiteness check + gradient-evaluation microbenchmark.

Builds the PyMC model for a solver_params config, gets each chain's initial
(unconstrained) position via resumable_sampler.prepare_from_pymc, and for each
chain: reports whether logp and grad(logp) are finite, then times repeated
value_and_grad calls (discarding the first as JIT/compile warmup, blocking on
results since JAX GPU dispatch is asynchronous) and reports the median
per-call time. This is the fast, low-noise signal for comparing tolerance/
floor/precision/device choices -- a full MCMC run bundles in sampler behavior
(tree depth, acceptance) that has nothing to do with raw compute cost.

Deliberately NOT vmapped across chains: vmapping can make one bad chain's
non-finite gradient look like every chain failed.

Usage (from the "Python Model" directory):
    python Utilities/bench_gradient.py --solver_params_file "Results/Benchmarks/<label>/solver_params.json"
    python Utilities/bench_gradient.py --solver_params_file ... --repeats 50 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import numpy as np

import inference_runner as ir
import resumable_sampler as rs


def check_and_bench(
    solver_params_file: str,
    n_chains: int = 4,
    seed: int = 42,
    jitter: bool = False,
    repeats: int = 30,
) -> dict:
    imported = ir.import_solver_params(solver_params_file)
    bundle = ir._build_model_bundle(imported)

    bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=n_chains, random_seed=seed, jitter=jitter)
    value_and_grad = jax.jit(jax.value_and_grad(bridge.logdensity_fn))

    per_chain = []
    for chain_idx in range(n_chains):
        position = [np.asarray(v)[chain_idx] for v in bridge.initial_positions]

        logp, grad = value_and_grad(position)
        jax.block_until_ready((logp, grad))
        logp_val = float(np.asarray(logp))
        grad_flat = np.concatenate([np.ravel(np.asarray(g)) for g in jax.tree_util.tree_leaves(grad)])
        logp_finite = bool(np.isfinite(logp_val))
        grad_finite = bool(np.all(np.isfinite(grad_flat)))

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = value_and_grad(position)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - t0)
        median_s = float(np.median(times))

        per_chain.append(
            {
                "chain": chain_idx,
                "logp": logp_val,
                "logp_finite": logp_finite,
                "grad_finite": grad_finite,
                "median_seconds_per_grad_eval": median_s,
                "grad_evals_per_sec": (1.0 / median_s) if median_s > 0 else None,
            }
        )

    controller_cfg = imported.solver_params.get("ODE_stepsize_controller", {})
    report = {
        "config": imported.results_save_dir.name,
        "rtol": controller_cfg.get("rtol"),
        "atol": controller_cfg.get("atol"),
        "initial_condition_floor": imported.solver_params.get("initial_condition_floor", 0.0),
        "n_chains": n_chains,
        "seed": seed,
        "jitter": jitter,
        "repeats": repeats,
        "jax_devices": [str(d) for d in jax.devices()],
        "per_chain": per_chain,
    }

    print(
        f"Config: {report['config']}  rtol={report['rtol']}  atol={report['atol']}  "
        f"floor={report['initial_condition_floor']}"
    )
    print(f"JAX devices: {report['jax_devices']}")
    for c in per_chain:
        status = "OK" if c["logp_finite"] and c["grad_finite"] else "BAD"
        rate = f"{c['grad_evals_per_sec']:.1f}/s" if c["grad_evals_per_sec"] else "n/a"
        print(
            f"  chain {c['chain']}: logp={c['logp']:.4f} logp_finite={c['logp_finite']} "
            f"grad_finite={c['grad_finite']}  median={c['median_seconds_per_grad_eval'] * 1000:.3f} ms "
            f"({rate})  [{status}]"
        )

    out_path = imported.results_save_dir / "gradient_bench.json"
    imported.results_save_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {out_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True)
    parser.add_argument("--n-chains", type=int, default=4, dest="n_chains")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="jitter initial points (default: off, so results are reproducible/comparable)",
    )
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    check_and_bench(
        args.solver_params_file,
        n_chains=args.n_chains,
        seed=args.seed,
        jitter=args.jitter,
        repeats=args.repeats,
    )
