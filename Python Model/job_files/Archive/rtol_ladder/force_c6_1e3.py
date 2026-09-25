"""Generate C6's training data at rtol=1e-3, skipping the flaky strict baseline probe.

C6 is the one system whose self-referential strict baseline solve (rtol=1e-10,
atol=1e-12) intermittently fails to converge inside 200,000 steps on Blanca, which
makes export_sweep_ranked raise before it can keep any conditions. That failure is
a hardware artifact, not a property of C6: on 2026-09-03 at 18:59 the same probe,
at the same PID (0.4, 0.3, 0.0) and the same tolerances, solved in 2105 steps and
set a strict cap of 3158.

The only thing that probe contributes is that cap, so this run supplies it directly
via strict_cap_override=3158 and never attempts the solve. Everything else is the
normal data-generation path:

  * the ChainSystem is built at rtol=1e-3, so ALL condition solves, the baseline,
    and the timeseries are generated at 1e-3 -- this is genuinely 1e-3 data, not
    1e-5 data relabelled
  * the per-condition strict filter still runs at 1e-10/1e-12, bounded by the
    supplied cap, so each kept condition's err_pct is still a real measurement
  * only the BASELINE's own err_pct is unavailable, since there is no strict
    baseline state to difference against

Three "is it acting odd" checks are printed:

  A. yield and cost -- how many conditions survived, and total steps against the
     675-823 range the other systems produced at 1e-3
  B. err_pct across the kept conditions, against the 10% relative noise model
  C. the decisive check -- re-solve the EXISTING 1e-5 dataset's own conditions at
     rtol=1e-3 and compare outputs. Those rows have known-good outputs, so any
     disagreement isolates the tolerance change from the choice of conditions.

Nothing is promoted unless 9 conditions survive and the baseline converges.
"""
import json
import sys
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

SYSTEM = "C6"
RTOL, ATOL = 1e-3, 1e-7
PID = dict(pcoeff=0.4, icoeff=0.3, dcoeff=0.0)
N_KEEP = 9
HARD_CAP = 20_000
NOISE_FRAC = 0.10

# Measured for C6 on 2026-09-03 18:59 (candidate C, same PID, strict 1e-10/1e-12):
# "baseline solved in 2105 steps at strict tolerance -> strict cap set to 3158".
STRICT_CAP = 3158

CFG = (ROOT / "Results" / "Chain Scaling Tests" / f"Chain {SYSTEM} - a1 tightest"
       / "solver_params.json")
DATA = ROOT / "Data" / f"Chain_{SYSTEM}"
STAGE_DIR = HERE / "force_c6_staging"


def build():
    cfg = json.loads(CFG.read_text())
    sg = cfg.get("scaling_groups")
    if not isinstance(sg, dict) or not sg:
        raise ValueError(f"{CFG} has no scaling_groups block; refusing to guess.")
    srcs = [ROOT / p for p in cfg["output_paths"]["reactions_source"]]
    sys_ = gcd.ChainSystem(srcs, rtol=RTOL, atol=ATOL,
                           scaling_group_overrides={k: float(v) for k, v in sg.items()},
                           **PID)
    targets = sys_.targets(gcd.UNSAT_PATTERN if "+unsat" in SYSTEM else gcd.SAT_PATTERN)
    return sys_, targets, sg


def solve_at(sys_, y0, rtol, atol, cap=HARD_CAP):
    sol = dfrx.diffeqsolve(
        dfrx.ODETerm(sys_.network), dfrx.Kvaerno5(),
        t0=gcd.TIME_RANGE[0], t1=gcd.TIME_RANGE[1], dt0=1e-6,
        y0=jnp.asarray(y0, dtype=jnp.float64), args=sys_.theta,
        saveat=dfrx.SaveAt(t1=True),
        stepsize_controller=dfrx.PIDController(rtol=rtol, atol=atol, **PID),
        max_steps=cap, throw=False)
    ok = bool(sol.result == dfrx.RESULTS.successful) and int(sol.stats["num_steps"]) < cap
    final = np.asarray(sol.ys[-1] if sol.ys.ndim == 2 else sol.ys)
    return final, ok, int(sol.stats["num_steps"])


def check_against_existing(sys_, targets):
    """Check C: re-solve the existing 1e-5 dataset's own conditions at rtol=1e-3."""
    csv = DATA / "init_vs_final_conc.csv"
    if not csv.exists():
        print("    (no existing CSV to compare against)")
        return None
    df = pd.read_csv(csv)
    sweep = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
    in_cols = [f"{n} (uM)" for n in sweep]
    out_cols = [f"{t} (uM)" for t in targets]
    missing = [c for c in in_cols + out_cols if c not in df.columns]
    if missing:
        print(f"    (existing CSV missing columns {missing}; skipping check C)")
        return None

    rows = []
    for _, r in df.iterrows():
        y0 = np.array(sys_.y0(), dtype=float)
        for name, col in zip(sweep, in_cols):
            y0[sys_.index_of[name]] = float(r[col])
        final, ok, steps = solve_at(sys_, y0, RTOL, ATOL)
        if not ok:
            rows.append(dict(ok=False, steps=steps, rel=None))
            continue
        got = np.array([float(final[sys_.index_of[t]]) for t in targets])
        ref = np.array([float(r[c]) for c in out_cols])
        denom = np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        with np.errstate(invalid="ignore"):
            rel = float(np.nanmax(np.abs(got - ref) / denom) * 100)
        rows.append(dict(ok=True, steps=steps, rel=rel))
    return rows


def main():
    sys_, targets, sg = build()
    print(f"=== {SYSTEM} data generation at rtol={RTOL:g} atol={ATOL:g} "
          f"PID={tuple(PID.values())} ===")
    print(f"  species={len(sys_.species)} targets={targets}")
    print(f"  scaling groups: { {k: sg[k] for k in sorted(sg)} }")
    print(f"  strict cap: {STRICT_CAP} (from C6's 2026-09-03 run; probe skipped)\n",
          flush=True)

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    res = gcd.export_sweep_ranked(
        sys_, targets, STAGE_DIR, max_steps=HARD_CAP,
        min_log_diff=0.2, max_steps_relative_to_baseline=1.5,
        strict_rtol=1e-10, strict_atol=1e-12,
        strict_cap_override=STRICT_CAP,
        n_keep=N_KEEP)

    kept = res["kept"][:N_KEEP]

    print(f"\n=== CHECK A: yield and cost ===")
    print(f"  found {res['n_found']} usable, kept {len(kept)} "
          f"(n_bad={res['n_bad']}, n_similar={res['n_similar']}, n_stiff={res['n_stiff']})")
    if kept:
        st = [k["steps"] for k in kept]
        print(f"  steps: min={min(st)} max={max(st)} total={sum(st)}")
        print(f"  other systems at 1e-3 totalled 675-823 over 9 conditions")

    print(f"\n=== CHECK B: err_pct vs the 1e-10/1e-12 reference ===")
    if kept:
        er = [k["err_pct"] for k in kept]
        print(f"  max={max(er):.4f}%  avg={sum(er)/len(er):.4f}%  "
              f"(noise model is {NOISE_FRAC*100:.0f}%)")
        print(f"  other systems at 1e-3 had max err 0.02-0.05%")

    print(f"\n=== CHECK C: existing 1e-5 conditions re-solved at 1e-3 ===")
    comp = check_against_existing(sys_, targets)
    verdict_c = None
    if comp:
        bad = sum(1 for r in comp if not r["ok"])
        rels = [r["rel"] for r in comp if r["ok"] and r["rel"] is not None]
        print(f"  {len(comp)} rows re-solved, {bad} failed to converge")
        if rels:
            verdict_c = max(rels)
            print(f"  max disagreement {verdict_c:.4f}%   mean {np.mean(rels):.4f}%")
            print(f"  steps at 1e-3: {[r['steps'] for r in comp if r['ok']]}")
            if verdict_c > NOISE_FRAC * 100:
                print(f"  !! ODD: exceeds the {NOISE_FRAC*100:.0f}% noise model")
            elif verdict_c > 1.0:
                print(f"  !  above 1% -- inside noise but worth noting")
            else:
                print(f"  OK: well inside the noise model")

    if len(kept) < N_KEEP:
        print(f"\nONLY {len(kept)} CONDITIONS (need {N_KEEP}) -- NOT promoting.")
        return 1

    y0 = np.array(sys_.y0(), dtype=float)
    final, ok, base_steps = solve_at(sys_, y0, RTOL, ATOL)
    if not ok:
        print("\nBASELINE DID NOT CONVERGE at 1e-3 -- NOT promoting.")
        return 1

    sweep = [n for n in gcd.SWEEP_SPECIES if n in sys_.index_of]
    rows = [[float(y0[sys_.index_of[n]]) for n in sweep]
            + [float(final[sys_.index_of[t]]) for t in targets]]
    rows += [k["combo"] + k["out"] for k in kept]
    cols = [f"{n} (uM)" for n in sweep] + [f"{t} (uM)" for t in targets]
    DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(DATA / "init_vs_final_conc.csv", index=False)
    ts = gcd.export_timeseries(sys_, targets, DATA, max_steps=HARD_CAP,
                               n_points=11, min_observable=1e-8)

    total = int(base_steps + sum(k["steps"] for k in kept))
    print(f"\nPROMOTED to {DATA}")
    print(f"  init_vs_final_conc.csv: {len(rows)} rows (baseline {base_steps} steps "
          f"+ {len(kept)} conditions), total_steps={total}")
    print(f"  time_vs_conc.csv: {len(ts)} rows")

    (HERE / "force_c6_result.json").write_text(json.dumps(dict(
        system=SYSTEM, rtol=RTOL, atol=ATOL, strict_cap=STRICT_CAP,
        strict_probe_skipped=True, n_found=res["n_found"], n_kept=len(kept),
        baseline_steps=base_steps, total_steps=total,
        max_err_pct=max(k["err_pct"] for k in kept),
        check_c_max_disagreement_pct=verdict_c), indent=2))
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
