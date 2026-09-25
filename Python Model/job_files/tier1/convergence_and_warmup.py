"""When would each chain-count run have stopped, and was tune=300 enough warmup?

Reads the chain-count checkpoints directly (no finalize needed) and answers two questions
the runs themselves did not answer, because they sampled with early stopping switched off
so that every run sampled the same way.

1. CONVERGENCE, applied after the fact. Walk the sampling draws in blocks of
   `rhat_check_every` (100) and apply the production rule: rank-normalised r-hat <= 1.01
   AND bulk ESS >= 100 x chains, on two consecutive checks, then one further block. The ESS
   half has to scale with chain count -- Vehtari et al. 2021 state it per split chain (>= 50
   each), which is 100 x chains once every chain is split in two -- because this comparison
   spans 4 to 64 chains, and a flat bar would hold the 64-chain run to a sixteenth of the
   standard the 4-chain run met, making "converged sooner" meaningless. Report the draw at
   which the rule would have fired, the sampling compute spent to reach it (A100-equivalent,
   from the progress log), and the bulk ESS there. A run that never satisfies the rule
   reports its best r-hat and how far it got, rather than a number that suggests it
   converged.

2. WARMUP ADEQUACY. Four independent signs that tune=300 was or was not enough:
     step size    per-chain adapted step size over the last warmup window, and how far the
                  final value sits from the window median -- a step size still moving at the
                  end of warmup means the dual-averaging had not settled
     agreement    spread of the final adapted step size across chains (chains that disagree
                  are still in different places)
     drift        parameter mean in the first sampling block vs the rest, in units of the
                  posterior sd: a short warmup shows up as the chains still moving when
                  sampling starts
     early r-hat  r-hat over all sampling draws vs r-hat with the first block dropped; a
                  large improvement means the first block was still transient

Usage: python convergence_and_warmup.py [run-dir-name ...]   (default: all)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path("/projects/anth4580/Bayesian/Results/Chain Count Test")
BLOCK = 100          # matches rhat_check_every in the configs
RHAT_MAX = 1.01
PER_SPLIT_ESS = 50   # ess_per_split_chain; x2 splits x chains gives the bar
CONSECUTIVE = 2      # convergence_consecutive_checks
POST_BLOCKS = 1      # post_convergence_checks


def compute_seconds(points):
    """Cumulative COMPUTE seconds at each logged step, with resume gaps removed.

    `t` in progress_log.jsonl is a wall-clock unix timestamp, not a compute clock, so a run
    that was checkpointed and requeued carries its entire queue wait inside t[-1] - t[0] --
    on the 8-chain run that is an 11.1 h gap between draws 255 and 260, which is longer than
    the sampling itself. Summing only the within-segment deltas gives the time the GPU was
    actually working. A gap is anything far outside the normal checkpoint spacing; the
    median delta sets the scale, so this adapts to runs that checkpoint at different rates.

    Returns (steps, cumulative_seconds) as parallel lists.
    """
    if len(points) < 2:
        return [p[0] for p in points], [0.0 for _ in points]
    deltas = [b[1] - a[1] for a, b in zip(points, points[1:])]
    positive = sorted(d for d in deltas if d > 0)
    median = positive[len(positive) // 2] if positive else 0.0
    limit = max(600.0, 10.0 * median)
    total, cum = 0.0, [0.0]
    for d in deltas:
        if 0 < d <= limit:          # a gap (or a clock going backwards) contributes nothing
            total += d
        cum.append(total)
    return [p[0] for p in points], cum


def _rhat_ess(arr):
    import arviz as az
    d = az.convert_to_dataset({"x": arr[:, :, None]})
    return (float(az.rhat(d, method="rank").x.values), float(az.ess(d, method="bulk").x.values))


def analyse(run_dir):
    import zarr
    ck = run_dir / "checkpoint"
    meta = json.loads((ck / "checkpoint_meta.json").read_text())
    rows = [json.loads(l) for l in (ck / "progress_log.jsonl").read_text().splitlines() if l.strip()]
    z = zarr.open(str(ck / "draws.zarr"), mode="r")
    f = meta.get("gpu_speed_vs_a100") or 1.0
    samp_rows = [(r["sampling_done"], r["t"]) for r in rows if r["phase"] == "sampling"]
    samp_steps, samp_cum = compute_seconds(samp_rows)

    def sampling_hours_at(draw):
        """A100-equivalent compute hours spent sampling by the time `draw` was reached."""
        s = next((c for d, c in zip(samp_steps, samp_cum) if d >= draw), None)
        return float(s * f / 3600) if s is not None else None

    ess_min = 2.0 * PER_SPLIT_ESS * int(meta["n_chains"])
    out = {"run": run_dir.name, "chains": meta["n_chains"], "tune": meta["n_tune"],
           "sampling_done": meta["sampling_done"], "gpu": meta.get("gpu"),
           "ess_threshold": ess_min, "warmup_hours": None, "params": {}}
    w_rows = [(r["warmup_done"], r["t"]) for r in rows if r["phase"] == "warmup"]
    if len(w_rows) >= 2:
        out["warmup_hours"] = float(compute_seconds(w_rows)[1][-1] * f / 3600)
    out["sampling_hours_total"] = float(samp_cum[-1] * f / 3600) if samp_cum else None

    names = list(z["sampling"].array_keys()) if "sampling" in z else []
    draws = {}
    for n in names:
        a = np.asarray(z[f"sampling/{n}"][:])
        draws[n] = a.reshape(a.shape[0], a.shape[1]) if a.ndim == 3 and a.shape[2] == 1 else a

    # --- 1. when the production rule would have fired ---
    n_draws = min((v.shape[1] for v in draws.values()), default=0)
    checks, streak, fired_at = [], 0, None
    for k in range(BLOCK, n_draws + 1, BLOCK):
        per = [_rhat_ess(v[:, :k]) for v in draws.values()]
        worst, worst_ess = max(p[0] for p in per), min(p[1] for p in per)
        checks.append({"draws": k, "rhat_max": round(worst, 4), "ess_bulk_min": round(worst_ess, 1)})
        streak = streak + 1 if (worst <= RHAT_MAX and worst_ess >= ess_min) else 0
        if streak >= CONSECUTIVE and fired_at is None:
            fired_at = k + POST_BLOCKS * BLOCK          # one further block, as in production
    out["rhat_checks"] = checks
    converged = fired_at is not None and fired_at <= n_draws
    out["converged"] = bool(converged)
    out["converged_at_draw"] = fired_at if converged else None
    out["best_rhat"] = min((c["rhat_max"] for c in checks), default=None)
    out["best_ess"] = max((c["ess_bulk_min"] for c in checks), default=None)
    # Which half of the rule is holding a non-converged run back -- at high chain counts
    # ESS is usually the binding one, and saying so is the point of the comparison.
    if not converged and checks:
        last = checks[-1]
        out["blocked_by"] = ("r-hat" if last["rhat_max"] > RHAT_MAX else "") + \
                            ("+" if last["rhat_max"] > RHAT_MAX and last["ess_bulk_min"] < ess_min else "") + \
                            ("ESS" if last["ess_bulk_min"] < ess_min else "")
        out["blocked_by"] = out["blocked_by"] or "confirmation blocks only"
    if converged:
        out["sampling_hours_to_converge"] = sampling_hours_at(fired_at)
        out["total_hours_to_converge"] = (out["warmup_hours"] or 0) + (out["sampling_hours_to_converge"] or 0)
        out["chain_draws_to_converge"] = fired_at * meta["n_chains"]
        for n, v in draws.items():
            r, e = _rhat_ess(v[:, :fired_at])
            out["params"][n] = {"rhat_at_convergence": r, "ess_bulk_at_convergence": e}
    for n, v in draws.items():
        if n_draws:
            r, e = _rhat_ess(v[:, :n_draws])
            out["params"].setdefault(n, {}).update({"rhat_all": r, "ess_bulk_all": e,
                                                    "mean": float(v.mean()), "sd": float(v.std())})

    # --- 2. was tune=300 enough? ---
    warm = {}
    if "warmup_stats" in z and "step_size" in z["warmup_stats"]:
        ss = np.asarray(z["warmup_stats"]["step_size"][:])          # (chains, warmup draws)
        last = ss[:, -1]
        tail = ss[:, -50:]
        warm["final_step_size_by_chain"] = [float(x) for x in last]
        warm["step_size_spread_across_chains"] = float(last.max() / last.min()) if last.min() > 0 else None
        warm["final_vs_last50_median"] = float(np.median(np.abs(last / np.median(tail, axis=1) - 1)))
        warm["step_size_window_medians"] = [float(np.median(ss[:, i:i + 50])) for i in range(0, ss.shape[1], 50)]
    if "warmup_stats" in z and "acceptance_rate" in z["warmup_stats"]:
        acc = np.asarray(z["warmup_stats"]["acceptance_rate"][:])
        warm["acceptance_last50"] = float(np.mean(acc[:, -50:]))
    if draws and n_draws >= 2 * BLOCK:
        drift = {}
        for n, v in draws.items():
            first, rest = v[:, :BLOCK], v[:, BLOCK:]
            sd = rest.std()
            drift[n] = {"first_block_mean_minus_rest_in_sd":
                        float((first.mean() - rest.mean()) / sd) if sd else None,
                        "rhat_all": _rhat_ess(v)[0],
                        "rhat_dropping_first_block": _rhat_ess(rest)[0]}
        warm["drift"] = drift
    out["warmup_check"] = warm
    return out


def main():
    wanted = sys.argv[1:]
    runs = [BASE / w for w in wanted] if wanted else sorted(
        (d for d in BASE.iterdir() if (d / "checkpoint" / "checkpoint_meta.json").exists()),
        key=lambda d: int(re.search(r"_(\d+)chains", d.name).group(1)) if re.search(r"_(\d+)chains", d.name) else 0)
    out = {}
    for r in runs:
        try:
            out[r.name] = analyse(r)
        except Exception as e:
            out[r.name] = {"run": r.name, "error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
