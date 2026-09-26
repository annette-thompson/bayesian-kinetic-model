"""What actually makes a run cheap: per-gradient cost across dataset variants.

Cost per draw is (leapfrog steps) x (ODE solves per gradient) x (sec per solve). The
sampler terms are fixed by the posterior; this profiles the solve terms, which are the
ones dataset design controls. Measuring cost first is deliberate: it is minutes rather
than the hours a full inference run costs, and it narrows which variants are worth
paying for information on.

The key structural fact, verified in inference_runner._build_simulator: ONE SaveAt
carrying every requested time is applied to EVERY condition, and solve_all_conditions
maps over the condition matrix. So the solve count equals the number of unique INITIAL
CONDITIONS. Extra time points ride along on solves that already happen. That makes the
timeseries dataset nearly free and the endpoint sweep the expensive part, which is the
opposite of the intuitive reading.

Variants profiled:

  n_conditions    2, 4, 6, 10        the dominant cost term
  cheapest-first  pick conditions by measured step count rather than dataset order.
                  Conditions differ severalfold in stiffness, and for a TEST system
                  there is no requirement that they be experimentally representative.
  t1              50, 100, 150 s     shorter horizon means fewer steps. Note the real
                  experiment runs 720 s, so any speedup here is a test-system-only
                  convenience and must be flagged as such downstream.
  timeseries      on/off             tests whether dropping the intermediate save
                  points is actually free, or whether dense output costs something.

Reports sec/gradient and total steps, so variants are comparable independent of node.

Usage: python cost_profile.py --system C6 [--reps 3]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/projects/anth4580/Bayesian/Utilities")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd

ROOT = gcd.project_root()
CFG_ROOT = ROOT / "Results" / "Chain Scaling Tests"
HARD_CAP = 20_000


def build(system):
    cfg = json.loads((CFG_ROOT / f"Chain {system} - a1 tightest"
                      / "solver_params.json").read_text())
    sg = {k: float(v) for k, v in cfg["scaling_groups"].items()}
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    ctrl = cfg["ODE_stepsize_controller"]
    sys_ = gcd.ChainSystem(srcs, rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
                           icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"],
                           scaling_group_overrides=sg)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in system else gcd.SAT_PATTERN)
    return sys_, targets, ctrl


def all_conditions(sys_, system):
    df = pd.read_csv(ROOT / "Data" / f"Chain_{system}" / "init_vs_final_conc.csv")
    sweep = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
    y0s = []
    for _, r in df.iterrows():
        y0 = np.array(sys_.y0(), dtype=float)
        for n in sweep:
            col = f"{n} (uM)"
            if col in df.columns:
                y0[sys_.index_of[n]] = float(r[col])
        y0s.append(y0)
    return y0s


def steps_for(sys_, y0, ctrl, t1):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(), t0=0.0, t1=t1, dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(
            rtol=ctrl["rtol"], atol=ctrl["atol"], pcoeff=ctrl["pcoeff"],
            icoeff=ctrl["icoeff"], dcoeff=ctrl["dcoeff"]),
        max_steps=HARD_CAP, throw=False)
    return int(sol.stats["num_steps"])


def make_batch_solver(sys_, ctrl, ts, t1):
    """Mirror the production solver: one SaveAt for all times, mapped over conditions."""
    saveat = dfrx.SaveAt(ts=jnp.asarray(ts, dtype=jnp.float64))
    rhs = dfrx.ODETerm(sys_.network)
    controller = dfrx.PIDController(rtol=ctrl["rtol"], atol=ctrl["atol"],
                                    pcoeff=ctrl["pcoeff"], icoeff=ctrl["icoeff"],
                                    dcoeff=ctrl["dcoeff"])

    def one(y0, theta):
        return dfrx.diffeqsolve(rhs, dfrx.Kvaerno5(), t0=0.0, t1=t1, dt0=1e-6,
                                y0=y0, args=theta, saveat=saveat,
                                stepsize_controller=controller,
                                max_steps=HARD_CAP, throw=False).ys

    def batch(Y, theta):
        return jax.lax.map(lambda y: one(y, theta), Y)

    # Scalar summary so a gradient exists; magnitude is irrelevant, cost is the point.
    def loss(theta, Y):
        return jnp.sum(batch(Y, theta) ** 2)

    return jax.jit(jax.value_and_grad(loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="C6")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    sys_, targets, ctrl = build(a.system)
    y0s = all_conditions(sys_, a.system)
    print(f"=== cost profile: {a.system} ===")
    print(f"  {len(sys_.species)} species, {len(y0s)} available conditions, "
          f"rtol={ctrl['rtol']:g}\n", flush=True)

    # Per-condition cost at the production horizon. Conditions differ severalfold, so
    # for a test system the cheap ones are worth preferring over the dataset's order.
    print("  per-condition steps at t1=150 (dataset order; row 0 is baseline):")
    per = [steps_for(sys_, y0, ctrl, 150.0) for y0 in y0s]
    print("    " + "  ".join(f"{i}:{s}" for i, s in enumerate(per)))
    order_cheap = list(np.argsort(per))
    print(f"    cheapest-first order: {order_cheap}")
    print(f"    baseline is rank {order_cheap.index(0)} of {len(per)} "
          f"({per[0]} steps vs cheapest {min(per)})\n", flush=True)

    TS_FULL = list(np.linspace(15.0, 150.0, 10))
    rows = []
    for ncond, t1, use_ts, pick in itertools.product(
            [2, 4, 6, 10], [50.0, 100.0, 150.0], [True, False], ["order", "cheap"]):
        if ncond > len(y0s):
            continue
        idx = (list(range(ncond)) if pick == "order" else order_cheap[:ncond])
        Y = jnp.asarray(np.stack([y0s[i] for i in idx]), dtype=jnp.float64)
        ts = [t for t in TS_FULL if t <= t1] if use_ts else [t1]
        if not ts:
            ts = [t1]
        fn = make_batch_solver(sys_, ctrl, ts, t1)
        v, g = fn(sys_.theta, Y)          # warm up / compile
        jax.block_until_ready(g)
        t0 = time.time()
        for _ in range(a.reps):
            v, g = fn(sys_.theta, Y)
            jax.block_until_ready(g)
        dt = (time.time() - t0) / a.reps
        steps = sum(steps_for(sys_, y0s[i], ctrl, t1) for i in idx)
        rows.append(dict(ncond=ncond, t1=t1, ts=use_ts, pick=pick,
                         sec_per_grad=dt, total_steps=steps, n_saved=len(ts)))
        print(f"  ncond={ncond:<3} t1={t1:<6g} ts={'yes' if use_ts else 'no ':<4} "
              f"{pick:<6} -> {dt*1000:8.1f} ms/grad  {steps:6d} steps  "
              f"{len(ts)} saved pts", flush=True)

    out = HERE / f"cost_profile_{a.system}.json"
    out.write_text(json.dumps(dict(system=a.system, per_condition_steps=per,
                                   cheapest_order=[int(i) for i in order_cheap],
                                   rows=rows), indent=2))

    base = next(r for r in rows if r["ncond"] == 10 and r["t1"] == 150.0
                and r["ts"] and r["pick"] == "order")
    print(f"\n  === speedup vs production config "
          f"(10 cond, t1=150, timeseries on) = {base['sec_per_grad']*1000:.1f} ms ===")
    for r in sorted(rows, key=lambda r: r["sec_per_grad"])[:8]:
        print(f"    {base['sec_per_grad']/r['sec_per_grad']:5.2f}x  "
              f"ncond={r['ncond']} t1={r['t1']:g} ts={'y' if r['ts'] else 'n'} {r['pick']}")
    print(f"\nwrote {out}\nDONE")


if __name__ == "__main__":
    main()
