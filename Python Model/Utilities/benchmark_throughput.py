"""Preflight throughput probe for the resumable BlackJAX sampler.

Runs a short, time-boxed burst of the REAL model (same config: chains, ODE
settings, priors) and measures how fast warmup and sampling draws are produced,
separating one-time JIT compile from the steady per-step cost. From that it
estimates how many 24h segments a full run needs, so you can size the job chain
(submit_inference_chain.sh --segments N) before committing.

Chains are vmapped on one device, so each step advances all chains together: the
per-step time already accounts for the config's chain count, and "draws/hr" below
is per chain (multiply by chains for total).

Caveat: warmup step cost grows as NUTS tree depth adapts, so the warmup estimate
from a short probe is approximate; the sampling estimate (fixed metric) is stable.

Usage (from the "Python Model" directory):
    python Utilities/benchmark_throughput.py --solver_params_file "Results/<run>/solver_params.json"
    python Utilities/benchmark_throughput.py --solver_params_file ... --minutes 5 --max-hours 23.5
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from inference_runner import _build_model_bundle, import_solver_params
import resumable_sampler as rs

import jax
import jax.numpy as jnp


def grad_eval_probe(solver_params_file: str, n_evals: int = 25) -> dict:
    """Time a single logp + gradient evaluation (one ODE solve + VJP), vmapped
    over chains -- the core simulation cost, independent of sampler efficiency.

    Far faster than the full NUTS probe on large/stiff models: it compiles once
    and times cheap gradient calls, with no warmup trees to wait through. This is
    the right tool for "how does sim time scale with #reactions / #free params",
    and it is the same quantity for nutpie or blackjax (only leapfrogs-per-draw
    differs between samplers).
    """
    imported = import_solver_params(solver_params_file)
    pc = imported.solver_params.get("posterior_sampling", {})
    n_chains = int(pc.get("chains", 4))

    print(f"Building model for {imported.results_save_dir.name} (chains={n_chains})...")
    bundle = _build_model_bundle(imported)
    bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=n_chains,
                                  random_seed=int(pc.get("random_seed", 0)))

    value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(bridge.logdensity_fn)))
    position = [jnp.asarray(p) for p in bridge.initial_positions]

    print("Compiling value_and_grad(logp)...")
    t0 = time.perf_counter()
    val, grad = value_and_grad(position)
    jax.block_until_ready((val, grad))
    compile_sec = time.perf_counter() - t0

    print(f"Timing {n_evals} gradient evals...")
    key = jax.random.PRNGKey(0)
    times = []
    for _ in range(n_evals):
        key, sub = jax.random.split(key)
        perturbed = [p + 0.01 * jax.random.normal(sub, p.shape) for p in position]
        t0 = time.perf_counter()
        val, grad = value_and_grad(perturbed)
        jax.block_until_ready((val, grad))
        times.append(time.perf_counter() - t0)

    reactions_source = imported.output_paths.get("reactions_source", [])
    if isinstance(reactions_source, str):
        reactions_source = [reactions_source]

    per_eval = float(statistics.median(times))
    report = {
        "config": imported.results_save_dir.name,
        "n_chains": n_chains,
        "n_enzymes": len(reactions_source),
        "n_free_params": len(bundle.free_params),
        "device": _device_str(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compile_sec": compile_sec,
        "sec_per_gradient_eval": per_eval,
        "gradient_evals_per_sec": (1.0 / per_eval) if per_eval else None,
        "n_evals_timed": n_evals,
    }
    out_path = imported.results_save_dir / "gradient_probe.json"
    imported.results_save_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== Gradient-Eval Probe ===")
    print(f"  config: {report['config']}   chains: {n_chains}   free params: {report['n_free_params']}   device: {report['device']}")
    print(f"  compile: {compile_sec:.0f}s")
    print(f"  {per_eval * 1000:.1f} ms / gradient-eval  ({report['gradient_evals_per_sec']:.1f}/s), vmapped over {n_chains} chains")
    print("  (compare across configs: rises with #reactions; #free params adds VJP cost)")
    print(f"\n  Wrote {out_path}")
    return report


def _steady_per_step(records: list[tuple[int, float]]) -> tuple[float | None, float | None, int]:
    """From [(steps, seconds), ...] return (steady_sec_per_step, compile_sec, n_steady_chunks).

    The first chunk carries JIT compilation; steady cost is the median per-step
    time over the remaining chunks, and compile is the first chunk's excess.
    """
    if not records:
        return None, None, 0
    if len(records) < 2:
        return None, None, 0  # only the compile-bearing chunk was measured
    per_step = statistics.median(dt / steps for steps, dt in records[1:])
    first_steps, first_dt = records[0]
    compile_sec = max(0.0, first_dt - first_steps * per_step)
    return per_step, compile_sec, len(records) - 1


def probe(solver_params_file: str, minutes: float, probe_tune: int,
          checkpoint_every: int, max_hours: float) -> dict:
    imported = import_solver_params(solver_params_file)
    pc = imported.solver_params.get("posterior_sampling", {})
    n_chains = int(pc.get("chains", 4))
    target_tune = int(pc.get("tune", 1000))
    target_draws = int(pc.get("draws", 1000))

    print(f"Building model for {imported.results_save_dir.name} (chains={n_chains})...")
    bundle = _build_model_bundle(imported)
    bridge = rs.prepare_from_pymc(bundle.pm_model, n_chains=n_chains,
                                  random_seed=int(pc.get("random_seed", 0)))

    tmp = Path(tempfile.mkdtemp(prefix="throughput_probe_"))
    spec = rs.SamplerSpec(
        n_tune=probe_tune,
        n_draws=10_000_000,            # effectively unbounded; the time budget stops it
        n_chains=n_chains,
        target_accept=float(pc.get("target_accept", 0.8)),
        is_mass_matrix_diagonal=bool(pc.get("is_mass_matrix_diagonal", True)),
        initial_step_size=float(pc.get("initial_step_size", 1.0)),
        checkpoint_every=checkpoint_every,
        random_seed=int(pc.get("random_seed", 0)),
    )
    sampler = rs.ResumableSampler(bridge.logdensity_fn, bridge.initial_positions,
                                  bridge.value_var_names, spec, tmp, bridge.config_hash)

    warmup_recs: list[tuple[int, float]] = []
    sampling_recs: list[tuple[int, float]] = []
    print(f"Probing for up to {minutes:.1f} min (probe_tune={probe_tune}, chunk={checkpoint_every})...")
    deadline = time.perf_counter() + minutes * 60.0
    prev_w = prev_s = 0
    mean_leapfrog = None
    try:
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            status = sampler.run(max_seconds=0.0)   # exactly one chunk
            dt = time.perf_counter() - t0
            dw, ds = status.warmup_done - prev_w, status.sampling_done - prev_s
            prev_w, prev_s = status.warmup_done, status.sampling_done
            if dw > 0:
                warmup_recs.append((dw, dt))
                print(f"  warmup   +{dw:>3} steps in {dt:6.1f}s  (total {status.warmup_done})")
            elif ds > 0:
                sampling_recs.append((ds, dt))
                print(f"  sampling +{ds:>3} draws in {dt:6.1f}s  (total {status.sampling_done})")
            if status.is_done:
                break
        # Mean leapfrog (integration) steps per sampling draw: separates ODE/VJP
        # compute cost from posterior difficulty. Read before the store is removed.
        samp_stats = sampler.read_stats("sampling")
        if "n_steps" in samp_stats and samp_stats["n_steps"].size:
            mean_leapfrog = float(np.mean(samp_stats["n_steps"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    warmup_step, warmup_compile, n_warm_steady = _steady_per_step(warmup_recs)
    sampling_step, sampling_compile, n_samp_steady = _steady_per_step(sampling_recs)
    sec_per_gradient = (sampling_step / mean_leapfrog) if (sampling_step and mean_leapfrog) else None

    report: dict = {
        "config": imported.results_save_dir.name,
        "n_chains": n_chains,
        "device": _device_str(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "probe_tune": probe_tune,
        "checkpoint_every": checkpoint_every,
        "warmup_sec_per_step": warmup_step,
        "warmup_compile_sec": warmup_compile,
        "warmup_steady_chunks": n_warm_steady,
        "sampling_sec_per_step": sampling_step,
        "sampling_compile_sec": sampling_compile,
        "sampling_steady_chunks": n_samp_steady,
        "mean_leapfrog_per_draw": mean_leapfrog,
        "sampling_sec_per_gradient_eval": sec_per_gradient,
        "warmup_draws_per_hr_per_chain": (3600.0 / warmup_step) if warmup_step else None,
        "sampling_draws_per_hr_per_chain": (3600.0 / sampling_step) if sampling_step else None,
        "target_tune": target_tune,
        "target_draws": target_draws,
        "max_hours_per_segment": max_hours,
    }
    report["sizing"] = _estimate_segments(report)

    out_path = imported.results_save_dir / "throughput_probe.json"
    imported.results_save_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _print_summary(report, out_path)
    return report


def _estimate_segments(r: dict) -> dict:
    ws, ss = r["warmup_sec_per_step"], r["sampling_sec_per_step"]
    if ws is None or ss is None:
        return {"note": "insufficient steady-state chunks measured; increase --minutes"}
    compile_total = (r["warmup_compile_sec"] or 0.0) + (r["sampling_compile_sec"] or 0.0)
    work_sec = r["target_tune"] * ws + r["target_draws"] * ss
    seg_budget = r["max_hours_per_segment"] * 3600.0
    useful = max(1.0, seg_budget - compile_total)   # each segment recompiles
    n_segments = max(1, math.ceil(work_sec / useful))
    wall_sec = work_sec + n_segments * compile_total
    return {
        "estimated_work_sec": round(work_sec, 1),
        "compile_overhead_per_segment_sec": round(compile_total, 1),
        "estimated_wall_hours": round(wall_sec / 3600.0, 2),
        "segments_needed": n_segments,
    }


def _device_str() -> str:
    try:
        import jax
        return ",".join(sorted({d.platform for d in jax.devices()}))
    except Exception:
        return "unknown"


def _print_summary(r: dict, out_path: Path) -> None:
    def rate(v):
        return f"{v:,.0f} draws/hr/chain" if v else "n/a (too few chunks)"

    print("\n=== Throughput Probe ===")
    print(f"  config: {r['config']}   chains: {r['n_chains']}   device: {r['device']}")
    print(f"  warmup:   {rate(r['warmup_draws_per_hr_per_chain'])}"
          f"   (compile ~{(r['warmup_compile_sec'] or 0):.0f}s, {r['warmup_steady_chunks']} steady chunks)")
    print(f"  sampling: {rate(r['sampling_draws_per_hr_per_chain'])}"
          f"   (compile ~{(r['sampling_compile_sec'] or 0):.0f}s, {r['sampling_steady_chunks']} steady chunks)")
    lf, spg = r["mean_leapfrog_per_draw"], r["sampling_sec_per_gradient_eval"]
    if lf and spg:
        print(f"    decompose: {r['sampling_sec_per_step']:.2f} s/draw = "
              f"{lf:.1f} leapfrog/draw x {spg * 1000:.1f} ms/gradient-eval")
        print(f"    (ms/gradient-eval scales with #reactions; leapfrog/draw with posterior difficulty)")
    s = r["sizing"]
    if "segments_needed" in s:
        print(f"\n  For target tune={r['target_tune']} draws={r['target_draws']}:")
        print(f"    estimated wall: {s['estimated_wall_hours']} h"
              f"  ->  {s['segments_needed']} segment(s) of {r['max_hours_per_segment']}h")
        print(f"    submit with:  submit_inference_chain.sh --solver-params <cfg> "
              f"--segments {s['segments_needed']}")
    else:
        print(f"\n  Sizing: {s['note']}")
    print(f"\n  Wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver_params_file", required=True)
    parser.add_argument("--minutes", type=float, default=10.0, help="probe wall-clock budget (default 10)")
    parser.add_argument("--probe-tune", type=int, default=40,
                        help="warmup steps to run in the probe before sampling (default 40)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="chunk size for per-chunk timing (default 5)")
    parser.add_argument("--max-hours", type=float, default=23.5,
                        help="per-segment wall budget used for the sizing estimate (default 23.5)")
    parser.add_argument("--grad-only", action="store_true",
                        help="skip NUTS; just time logp+gradient evals (fast for large/stiff "
                        "models). Use this to map sim-time scaling with #reactions / #free params.")
    parser.add_argument("--grad-evals", type=int, default=25,
                        help="number of gradient evals to time in --grad-only mode (default 25)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.grad_only:
        grad_eval_probe(args.solver_params_file, args.grad_evals)
    else:
        probe(args.solver_params_file, args.minutes, args.probe_tune, args.checkpoint_every, args.max_hours)
