"""How many ODE solver steps did each Tier-1 run's accepted states need?

For every run with a checkpoint, this takes each chain's accepted positions -- all warmup
steps and all sampling draws in checkpoint/draws.zarr, repeats (rejected transitions)
dropped -- and solves the five Tier-1 conditions to 720 s at each one with the run's own
tolerances, the production PID controller and a 20000-step limit. It reports the most steps
any condition needed, per run and per phase, and how many positions needed more than 200, 500
and 1000. The question it answers: would a `max_steps` of 1000 have cut off any state the
chains actually visited (written 2026-09-27, when the C8 runs' stiff band made 1000 the
proposed limit).

Rejected proposals inside NUTS trees are not stored, so they are not checked; those are the
ones a lower limit is meant to cut short.

One process compiles one model (runs are grouped by reaction set and tolerances; see
--groups), since jaxlib's CPU JIT can abort on a second compile in a process.

    python accepted_solver_steps.py --groups                 # "GROUP<tab>key<tab>runs" lines
    python accepted_solver_steps.py --group "C8|0.001|1e-07" --json out.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Utilities"))

import numpy as np

RESULTS = ROOT / "Results" / "Tier1"
LIMIT = 20000
BATCH = {"C8": 1024, "C14+unsat": 512, "C14+unsat+c3split": 512, "C18": 256}


def reactions_folder(cfg_path):
    import inference_runner as ir
    src = ir.import_solver_params(cfg_path).reactions_source
    return Path(src[0] if isinstance(src, (list, tuple)) else src).parent.name


def run_groups():
    """{group key: [run name, ...]}; key = reactions folder | rtol | atol."""
    groups = {}
    for d in sorted(RESULTS.glob("Tier1 */")):
        cfg_path = d / "solver_params.json"
        if not (cfg_path.exists() and (d / "checkpoint" / "draws.zarr").exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        c = cfg["ODE_stepsize_controller"]
        key = f"{reactions_folder(cfg_path)}|{c['rtol']:g}|{c['atol']:g}"
        groups.setdefault(key, []).append(d.name)
    return groups


def natural(name, u):
    """A draws.zarr coordinate back on the parameter's own scale."""
    if name.endswith("_log__"):
        return name[:-6], np.exp(u)
    if "_x" in name and name.endswith("__"):
        base, factor = name[:-2].rsplit("_x", 1)
        return base, u / float(factor)
    return name, u


def accepted_positions(run):
    """{phase: (names, array (n, params), index (n, 2) of chain and step)} with repeats dropped."""
    import zarr
    root = zarr.open_group(store=zarr.storage.LocalStore(str(RESULTS / run / "checkpoint" / "draws.zarr")), mode="r")
    out = {}
    for phase in ("warmup", "sampling"):
        if phase not in root:
            continue
        grp = root[phase]
        cols, names = [], []
        for var in sorted(grp.keys()):
            name, x = natural(var, np.asarray(grp[var][...], dtype=float))
            names.append(name)
            cols.append(x)
        if not cols or cols[0].size == 0:
            continue
        chains, steps = cols[0].shape
        pts = np.stack([c.reshape(-1) for c in cols], axis=1)
        idx = np.stack(np.meshgrid(np.arange(chains), np.arange(steps), indexing="ij"), axis=-1).reshape(-1, 2)
        keep = np.all(np.isfinite(pts), axis=1)
        pts, idx = pts[keep], idx[keep]
        _, first = np.unique(pts, axis=0, return_index=True)
        first = np.sort(first)
        out[phase] = (names, pts[first], idx[first], int(keep.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", action="store_true", help="print the groups and exit")
    ap.add_argument("--group", help="one group key from --groups")
    ap.add_argument("--json", default=None, help="append results to this JSON file")
    ap.add_argument("--only", default=None, help="runs whose name contains this (testing)")
    ap.add_argument("--max_points", type=int, default=None, help="first N positions per phase (testing)")
    ap.add_argument("--batch", type=int, default=None, help="positions per compiled batch")
    a = ap.parse_args()

    groups = run_groups()
    if a.groups:
        for k, runs in groups.items():
            print(f"GROUP\t{k}\t{len(runs)}")        # tagged: importing the fit code prints to stdout
        return
    runs = [r for r in groups[a.group] if not a.only or a.only in r]
    folder, rtol, atol = a.group.split("|")
    rtol, atol = float(rtol), float(atol)

    import diffrax as dfrx
    import jax
    import jax.numpy as jnp
    from forward_model import CONDITIONS, PRODUCTION_PID, ForwardModel

    system = folder.replace("+c3split", "")
    fm = ForwardModel(system, times=[720.0], rtol=rtol, atol=atol, max_steps=LIMIT, reactions=folder)
    ctrl = dfrx.PIDController(rtol=rtol, atol=atol, pcoeff=PRODUCTION_PID[0], icoeff=PRODUCTION_PID[1],
                              dcoeff=PRODUCTION_PID[2])

    def solve(y0, theta):
        sol = dfrx.diffeqsolve(dfrx.ODETerm(fm._sys.network), dfrx.Kvaerno5(), t0=0.0, t1=720.0, dt0=1e-6,
                               y0=y0, args=theta, saveat=dfrx.SaveAt(t1=True), stepsize_controller=ctrl,
                               max_steps=LIMIT, throw=False)
        return sol.stats["num_steps"], sol.result == dfrx.RESULTS.successful

    batched = jax.jit(jax.vmap(jax.vmap(solve, in_axes=(0, None)), in_axes=(None, 0)))
    y0 = jnp.asarray(np.stack([fm.y0(ch) for _, ch in CONDITIONS]), dtype=jnp.float64)
    base = np.asarray(fm.theta(), dtype=float)
    B = a.batch or BATCH.get(folder, 256)
    print(f"== group {a.group}: {len(runs)} runs, {jax.devices()[0].device_kind}, batch {B}", flush=True)

    results = {}
    for run in runs:
        t0 = time.time()
        rec = {"group": a.group}
        for phase, (names, pts, idx, n_all) in accepted_positions(run).items():
            if a.max_points:
                pts, idx = pts[:a.max_points], idx[:a.max_points]
            cols = [fm._param_index[n] for n in names]
            steps, ok = [], []
            for i in range(0, len(pts), B):
                chunk = pts[i:i + B]
                th = np.tile(base, (len(chunk), 1))
                th[:, cols] = chunk
                th = np.concatenate([th, np.tile(th[-1:], (B - len(chunk), 1))])   # one compiled shape
                n, s = batched(y0, jnp.asarray(th))
                steps.append(np.asarray(n).max(axis=1)[:len(chunk)])
                ok.append(np.asarray(s).all(axis=1)[:len(chunk)])
            steps, ok = np.concatenate(steps), np.concatenate(ok)
            k = int(np.argmax(steps))
            rec[phase] = {
                "accepted": n_all, "unique": int(len(pts)), "max_steps": int(steps.max()),
                "median_steps": int(np.median(steps)), "over_200": int((steps > 200).sum()),
                "over_500": int((steps > 500).sum()), "over_1000": int((steps > 1000).sum()),
                "failed": int((~ok).sum()),
                "worst": {"chain": int(idx[k, 0]), "step": int(idx[k, 1]),
                          "values": {n: float(v) for n, v in zip(names, pts[k])}},
            }
        results[run] = rec
        parts = [f"{ph} {r['unique']:5d} pts max {r['max_steps']:5d} (>1000: {r['over_1000']}, failed {r['failed']})"
                 for ph, r in rec.items() if ph != "group"]
        print(f"{run:<46} " + " | ".join(parts) + f"  [{time.time() - t0:.0f} s]", flush=True)

    if a.json:
        p = Path(a.json)
        allres = json.loads(p.read_text()) if p.exists() else {}
        allres.update(results)
        p.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
