"""Does the chain-scaled ESS rule move any run's convergence point?

Vehtari et al. 2021 state the ESS requirement per split chain (>= 50 each), which is
100 x chains once each chain is split in two -- 400 at four chains, the familiar number,
but 800 at eight. Every finished run here used a flat 400 regardless of chain count, so
the eight-chain runs were held to half the standard they should have been.

For each run this walks the sampling draws in blocks of `rhat_check_every` and reports
the draw at which the production rule would have fired under

  old   rank-normalised r-hat <= 1.01 and bulk ESS >= 400          (flat, what ran)
  new   rank-normalised r-hat <= 1.01 and bulk ESS >= 100 x chains (scaled)

both with the same two-consecutive-checks-plus-one-block structure the sampler uses,
and the worst value across free parameters at each check.

Two things decide whether a run's line can actually move. A run whose config carries no
`ess_threshold` key ran under the hardcoded 400 and STOPPED when it fired, so draws past
that point were never taken: if the scaled rule wants more, the honest report is
"stopped short", not a later convergence draw. A run with `ess_threshold: null` sampled
to a compute budget with early stopping off, so its later draws are on disk and the
firing point is a real, recomputable number.

Usage: python ess_rule_recheck.py [--base "<results subdir>" ...] [--out file.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/projects/anth4580/Bayesian/Results")
DEFAULT_BASES = ("Chain Scaling Tests", "Chain Count Test", "Multiparam Tests")
OLD_ESS = 400.0
PER_SPLIT = 50.0        # Vehtari's per-split-chain minimum; x2 splits x chains
RHAT_MAX = 1.01
CONSECUTIVE = 2
POST_BLOCKS = 1


def _rhat_ess(arr):
    import arviz as az
    d = az.convert_to_dataset({"x": arr[:, :, None]})
    return (float(az.rhat(d, method="rank").x.values), float(az.ess(d, method="bulk").x.values))


def _fired(checks, ess_min, block):
    """Where the production rule lands, separating the two reasons a run can fall short.

    `first_ok` is the first check that satisfies r-hat and ESS on its own. `fired` adds the
    sampler's confirmation structure on top: the same test has to hold on two consecutive
    checks, and one further block is drawn after that. A run can meet the ESS bar
    comfortably and still have no `fired` draw simply because it ended before those
    confirmation blocks -- a different situation from one whose ESS never got there, and
    the two must not be reported the same way.
    """
    streak, first_ok = 0, None
    for c in checks:
        ok = c["rhat"] <= RHAT_MAX and c["ess"] >= ess_min
        if ok and first_ok is None:
            first_ok = c["draws"]
        streak = streak + 1 if ok else 0
        if streak >= CONSECUTIVE:
            return first_ok, c["draws"] + POST_BLOCKS * block
    return first_ok, None


def analyse(run_dir):
    import zarr
    ck = run_dir / "checkpoint"
    meta = json.loads((ck / "checkpoint_meta.json").read_text())
    cfg = json.loads((run_dir / "solver_params.json").read_text())["posterior_sampling"]
    z = zarr.open(str(ck / "draws.zarr"), mode="r")
    n_chains = int(meta["n_chains"])
    block = int(cfg.get("rhat_check_every") or 100)

    # A config with no ess_threshold key ran under the hardcoded 400 and stopped on it;
    # ess_threshold: null means early stopping was off and the run went to its budget.
    stopped_on_old_rule = "ess_threshold" not in cfg and cfg.get("rhat_threshold") is not None

    out = {"run": run_dir.name, "chains": n_chains, "sampling_done": meta.get("sampling_done"),
           "phase": meta.get("phase"), "block": block,
           "ess_required_old": OLD_ESS, "ess_required_new": 2 * PER_SPLIT * n_chains,
           "early_stopping_was_on": stopped_on_old_rule}

    draws = {}
    for n in (list(z["sampling"].array_keys()) if "sampling" in z else []):
        a = np.asarray(z[f"sampling/{n}"][:])
        a = a.reshape(a.shape[0], a.shape[1]) if a.ndim == 3 and a.shape[2] == 1 else a
        if a.ndim == 2 and a.shape[1] >= block:
            draws[n] = a
    if not draws:
        # A run still in warmup has no sampling draws yet; that is a phase, not a failure.
        return {**out, "verdict": f"no sampling draws yet (phase {meta.get('phase')})"}

    n_draws = min(v.shape[1] for v in draws.values())
    checks = []
    for k in range(block, n_draws + 1, block):
        per = [_rhat_ess(v[:, :k]) for v in draws.values()]
        checks.append({"draws": k, "rhat": max(p[0] for p in per), "ess": min(p[1] for p in per)})
    out["checks"] = [{"draws": c["draws"], "rhat": round(c["rhat"], 4), "ess": round(c["ess"], 1)}
                     for c in checks]
    out["ess_at_end"] = round(checks[-1]["ess"], 1)
    out["rhat_at_end"] = round(checks[-1]["rhat"], 4)

    for label, ess_min in (("old", OLD_ESS), ("new", out["ess_required_new"])):
        first_ok, f = _fired(checks, ess_min, block)
        reached = f is not None and f <= n_draws
        out[f"converged_{label}"] = bool(reached)
        out[f"draw_{label}"] = f if reached else None
        out[f"first_ok_{label}"] = first_ok

    need = out["ess_required_new"]
    # Where the line goes on a plot. For a run that already fired, that is the measured
    # draw. For one that ended mid-rule, the earliest it could have fired is two checks
    # after the first passing one (the streak) plus the post-convergence block -- an
    # extrapolation that assumes the rule keeps holding, so it is flagged as projected and
    # must be drawn differently from a measured point.
    if out["converged_new"]:
        out["plot_draw_new"], out["projected"] = out["draw_new"], False
    elif out["first_ok_new"] is not None:
        out["plot_draw_new"] = out["first_ok_new"] + (CONSECUTIVE - 1 + POST_BLOCKS) * block
        out["projected"] = True
        out["extra_draws_needed"] = max(0, out["plot_draw_new"] - n_draws)
    else:
        out["plot_draw_new"], out["projected"] = None, False

    if out["converged_new"]:
        out["verdict"] = ("unchanged" if out["draw_new"] == out["draw_old"]
                          else f"moves {out['draw_old']} -> {out['draw_new']}")
    elif out["first_ok_new"] is not None:
        # ESS and r-hat both cleared; only the confirmation blocks are missing, because the
        # run stopped on the old rule before drawing them. Report the shortfall in draws.
        out["verdict"] = (f"{n_draws} draws on disk, rule needs a confirmation block past them "
                          f"(ESS {out['ess_at_end']:g} >= {need:g} from draw {out['first_ok_new']})")
    elif out["converged_old"]:
        # The new rule never had a single passing check. Name the condition that blocks it:
        # at eight chains ESS is usually comfortable and r-hat is what binds, so reporting
        # this as an ESS shortfall would point at the wrong thing.
        blocking = []
        if out["ess_at_end"] < need:
            blocking.append(f"ESS {out['ess_at_end']:g} < {need:g}")
        if out["rhat_at_end"] > RHAT_MAX:
            blocking.append(f"r-hat {out['rhat_at_end']:g} > {RHAT_MAX}")
        out["verdict"] = ("no check passes the new rule: "
                          + (", ".join(blocking) or "passes only outside the streak")
                          + " at the last draw")
    else:
        out["verdict"] = "never converged under either rule"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for b in (a.base or DEFAULT_BASES):
        d = ROOT / b
        if not d.is_dir():
            continue
        for r in sorted(x for x in d.iterdir() if (x / "checkpoint" / "checkpoint_meta.json").exists()):
            try:
                res[f"{b}/{r.name}"] = analyse(r)
            except Exception as e:
                res[f"{b}/{r.name}"] = {"run": r.name, "error": f"{type(e).__name__}: {e}"}
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
        print(f"wrote {a.out}  ({len(res)} runs)")
    else:
        print(txt)


if __name__ == "__main__":
    main()
