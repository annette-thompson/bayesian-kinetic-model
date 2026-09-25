"""Pick the cheapest ODE tolerance per chain system that still yields 10 endpoint
conditions, then regenerate that system's training data at the winning tolerance.

Motivation: the production data was generated at rtol=1e-5 (candidate C, PID
0.4/0.3/0.0). For scoping runs a looser tolerance is much cheaper per solve, and
the a2-era runs used rtol=1e-3. But looser is not automatically cheaper: as the
tolerance relaxes, the PID controller can start thrashing and the *rejected* step
count can rise faster than the accepted count falls. So rather than assume the
ordering, this measures all three candidate tolerances and picks the one that
actually minimizes steps while still producing a full 10-row endpoint dataset.

Selection rule, per system:
  * try rtol in {1e-3, 1e-4, 1e-5}, all with atol=1e-7 and PID (0.4, 0.3, 0.0)
  * a tolerance is ELIGIBLE if export_sweep_ranked keeps >= 9 conditions, which
    with the baseline row makes the required 10
  * among eligible tolerances, choose the one with the smallest TOTAL steps over
    the kept set plus baseline. Total (not max) is the right cost proxy because
    every logp evaluation during inference solves every condition.
  * if no tolerance is eligible, the system keeps its existing 1e-5 data untouched

Error reporting: err_pct is each condition's loose-vs-strict disagreement, where
"strict" is that same PID at rtol=1e-10/atol=1e-12. This is NOT part of the
selection rule (the user's criterion is steps subject to 10 conditions), but it is
printed prominently per tolerance because it is the number that decides whether
the choice is scientifically defensible: the noise model applied to these datasets
is 10% relative, so a tolerance whose err_pct approaches 10% is injecting
numerical error comparable to the observation noise, and a1 will absorb some of it.

Data is generated into a staging tree first and only promoted into Data/Chain_<sys>/
once a winner is known, so a failed or interrupted run cannot leave a system with a
half-written dataset.

Usage:
    python rtol_ladder.py                 # all 14 systems
    python rtol_ladder.py C6 C8           # a subset
    python rtol_ladder.py --dry-run       # measure and report, promote nothing
"""
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Utilities"))

import numpy as np
import pandas as pd
import jax.numpy as jnp
import diffrax as dfrx
import generate_chain_data as gcd

gcd.TIME_RANGE = (0.0, 150.0)
ROOT = gcd.project_root()

SYSTEMS = ["C4_NoFB", "C6", "C8", "C10", "C12", "C12+unsat", "C14", "C14+unsat",
           "C16", "C16+unsat", "C18", "C18+unsat", "C20", "C20+unsat"]

# Ordered loosest-first purely for readable logs; selection is by measured steps,
# not by position in this list.
RTOL_LADDER = [1e-3, 1e-4, 1e-5]
ATOL = 1e-7
PID = dict(pcoeff=0.4, icoeff=0.3, dcoeff=0.0)

N_KEEP = 9            # + baseline = the 10 rows the datasets must have
HARD_CAP = 20_000
STAGING = HERE / "rtol_ladder_staging"
# One results file per invocation, so the 14 systems can be split across several
# concurrent jobs without them clobbering each other's summary.
RESULTS_DIR = HERE / "rtol_ladder_results"

# The tolerance the existing production data was generated at. A system that
# fails every rung of the ladder keeps this and is left alone.
INCUMBENT_RTOL = 1e-5


def config_path(name):
    return (ROOT / "Results" / "Chain Scaling Tests" / f"Chain {name} - a1 tightest"
            / "solver_params.json")


def load_cfg(name):
    cfg = json.loads(config_path(name).read_text())
    sg = cfg.get("scaling_groups")
    if not isinstance(sg, dict) or not sg:
        # Never guess. d-prefixed groups are additive inside exp() so their nominal
        # value is 0.0 while ordinary groups are 1.0; defaulting everything to 1.0
        # corrupts TesA's rate by ~4.4e5x at C12. The authoritative values live in
        # each system's own config.
        raise ValueError(
            f"{config_path(name)} has no 'scaling_groups' block; refusing to "
            "generate data with guessed scaling values.")
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    return cfg, srcs, {k: float(v) for k, v in sg.items()}


def build(name, srcs, sg, rtol):
    return gcd.ChainSystem(srcs, rtol=rtol, atol=ATOL, scaling_group_overrides=sg, **PID)


def solve_baseline(sys_, rtol):
    """Baseline endpoint at this tolerance, returning (final_state, ok, steps)."""
    y0 = sys_.y0()
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(rtol=rtol, atol=ATOL, **PID),
        max_steps=HARD_CAP, throw=False)
    ok = bool(sol.result == dfrx.RESULTS.successful) and int(sol.stats["num_steps"]) < HARD_CAP
    final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
    return y0, final, ok, int(sol.stats["num_steps"])


def measure(name, srcs, sg, rtol, stage_dir):
    """Generate at one tolerance. Returns a stats dict (eligible flag included)."""
    t0 = time.time()
    out = dict(rtol=rtol, eligible=False, n_found=0, n_kept=0, total_steps=None,
               max_steps=None, max_err_pct=None, baseline_steps=None, error=None,
               seconds=None)
    try:
        sys_ = build(name, srcs, sg, rtol)
        targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in name else gcd.SAT_PATTERN)
        stage_dir.mkdir(parents=True, exist_ok=True)

        res = gcd.export_sweep_ranked(
            sys_, targets, stage_dir, max_steps=HARD_CAP,
            min_log_diff=0.2, max_steps_relative_to_baseline=1.5,
            strict_rtol=1e-10, strict_atol=1e-12, strict_probe_max_steps=200_000,
            n_keep=N_KEEP)
        kept = res["kept"][:N_KEEP]
        out["n_found"] = res["n_found"]
        out["n_kept"] = len(kept)

        y0, final, ok, base_steps = solve_baseline(sys_, rtol)
        out["baseline_steps"] = base_steps
        if not ok:
            out["error"] = "baseline endpoint solve did not converge"
            return out, None
        if len(kept) < N_KEEP:
            out["error"] = f"only {len(kept)} conditions, need {N_KEEP}"
            return out, None

        out["total_steps"] = int(base_steps + sum(k.get("steps", 0) for k in kept))
        out["max_steps"] = int(max([base_steps] + [k.get("steps", 0) for k in kept]))
        out["max_err_pct"] = float(max(k.get("err_pct", 0.0) for k in kept))
        out["eligible"] = True

        # Endpoint CSV, baseline row first, matching export_sweep's column layout.
        sweep_names = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
        base_combo = [float(y0[sys_.index_of[n]]) for n in sweep_names]
        base_out = [float(final[sys_.index_of[t]]) for t in targets]
        rows = [base_combo + base_out] + [k["combo"] + k["out"] for k in kept]
        cols = [f"{n} (uM)" for n in sweep_names] + [f"{t} (uM)" for t in targets]
        pd.DataFrame(rows, columns=cols).to_csv(stage_dir / "init_vs_final_conc.csv",
                                                index=False)

        # Timeseries must come from the SAME tolerance the inference will use.
        ts = gcd.export_timeseries(sys_, targets, stage_dir, max_steps=HARD_CAP,
                                   n_points=11, min_observable=1e-8)
        out["timeseries_rows"] = len(ts)
    except Exception as exc:  # noqa: BLE001 - one bad tolerance must not kill the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out["seconds"] = round(time.time() - t0, 1)
    return out, stage_dir


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    systems = argv or SYSTEMS

    all_results = {}
    for name in systems:
        print(f"\n{'=' * 70}\n=== {name} ===\n{'=' * 70}", flush=True)
        try:
            cfg, srcs, sg = load_cfg(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP: {exc}", flush=True)
            all_results[name] = dict(error=str(exc))
            continue
        print(f"  scaling groups: { {k: sg[k] for k in sorted(sg)} }", flush=True)

        per_tol = []
        for rtol in RTOL_LADDER:
            stage = STAGING / name / f"rtol{rtol:.0e}"
            print(f"\n  --- rtol={rtol:g} atol={ATOL:g} PID={tuple(PID.values())} ---",
                  flush=True)
            stats, _ = measure(name, srcs, sg, rtol, stage)
            per_tol.append(stats)
            if stats["eligible"]:
                print(f"  -> ELIGIBLE  kept={stats['n_kept']}/{N_KEEP}  "
                      f"total_steps={stats['total_steps']}  max_steps={stats['max_steps']}  "
                      f"max_err={stats['max_err_pct']:.3f}%  ({stats['seconds']}s)", flush=True)
            else:
                print(f"  -> not eligible: {stats['error']}  ({stats['seconds']}s)", flush=True)

        eligible = [s for s in per_tol if s["eligible"]]
        if eligible:
            winner = min(eligible, key=lambda s: s["total_steps"])
            print(f"\n  WINNER for {name}: rtol={winner['rtol']:g} "
                  f"(total_steps={winner['total_steps']}, max_err={winner['max_err_pct']:.3f}%)",
                  flush=True)
            if not dry_run:
                src = STAGING / name / f"rtol{winner['rtol']:.0e}"
                dst = ROOT / "Data" / f"Chain_{name}"
                dst.mkdir(parents=True, exist_ok=True)
                for fn in ("init_vs_final_conc.csv", "time_vs_conc.csv"):
                    if (src / fn).exists():
                        shutil.copy2(src / fn, dst / fn)
                print(f"  promoted {src} -> {dst}", flush=True)
        else:
            winner = None
            print(f"\n  NO eligible tolerance for {name}; keeping existing "
                  f"rtol={INCUMBENT_RTOL:g} data untouched", flush=True)

        all_results[name] = dict(
            per_tolerance=per_tol,
            chosen_rtol=(winner["rtol"] if winner else INCUMBENT_RTOL),
            promoted=bool(winner) and not dry_run)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_json = RESULTS_DIR / f"{systems[0]}_plus{len(systems) - 1}.json"
    results_json.write_text(json.dumps(all_results, indent=2))

    print(f"\n\n{'=' * 78}\nSUMMARY\n{'=' * 78}", flush=True)
    hdr = ("SYSTEM", "1e-3", "1e-4", "1e-5", "CHOSEN", "steps", "maxErr%")
    W = (12, 10, 10, 10, 8, 8, 8)
    print("  ".join(h.ljust(w) for h, w in zip(hdr, W)))
    print("-" * (sum(W) + 12))
    for name in systems:
        r = all_results.get(name, {})
        if "per_tolerance" not in r:
            print(f"{name.ljust(W[0])}  ERROR: {r.get('error', '?')}")
            continue
        by = {s["rtol"]: s for s in r["per_tolerance"]}
        cells = [name]
        for rt in RTOL_LADDER:
            s = by.get(rt)
            cells.append(str(s["total_steps"]) if s and s["eligible"] else "x")
        chosen = r["chosen_rtol"]
        cs = by.get(chosen)
        cells += [f"{chosen:g}",
                  str(cs["total_steps"]) if cs and cs["eligible"] else "-",
                  f"{cs['max_err_pct']:.2f}" if cs and cs["eligible"] else "-"]
        print("  ".join(c.ljust(w) for c, w in zip(cells, W)))
    print(f"\nwrote {results_json}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
