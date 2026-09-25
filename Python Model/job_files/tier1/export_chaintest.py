"""Summarise the chain-count runs into JSON for chain_count_analysis.ipynb.

Reads each run's checkpoint directly (draws.zarr + progress_log.jsonl + checkpoint_meta),
so it works while a run is still going and needs no finalize step. Everything time-based is
in A100-equivalent seconds, the units the sampler banks compute in.

Per run it reports:
  timing      warmup and sampling A100-equivalent COMPUTE seconds (queue waits between
              resume segments removed -- see compute_seconds), seconds per warmup step and
              per sampling draw (first chunk of each phase dropped: it carries JIT
              compilation)
  posterior   per-parameter bulk/tail ESS and rank-normalised r-hat over the sampling draws
  trajectory  ESS and elapsed sampling compute at increasing draw prefixes, so ESS per hour
              can be plotted as it accumulates rather than only at the end
  sampler     median adapted step size, acceptance, leapfrog steps per draw, divergence rate
  groups      r-hat and ESS computed within disjoint groups of 4, 8, 16, 32 chains -- the
              spread across groups at one size is how uncertain that diagnostic is with
              that many chains

Usage: python export_chaintest.py [run-dir-name ...]   (default: every dir in Results/Chain Count Test)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path("/projects/anth4580/Bayesian/Results/Chain Count Test")
PREFIXES = 12          # points on the ESS-vs-compute trajectory
GROUP_SIZES = (4, 8, 16, 32)


def compute_seconds(points):
    """Cumulative COMPUTE seconds per logged step, with resume gaps removed.

    `t` in progress_log.jsonl is a wall-clock timestamp, so a checkpointed-and-requeued run
    carries its whole queue wait inside t[-1] - t[0]: the 8-chain run waited 11.1 h between
    draws 255 and 260, longer than its sampling. Everything here is meant to be compute, and
    an ESS-per-hour figure that counts queue time would rank whichever run happened to wait
    longest as the least efficient. Only within-segment deltas count; the median delta sets
    the scale for what counts as a gap.
    """
    if len(points) < 2:
        return [p[0] for p in points], [0.0 for _ in points]
    deltas = [b[1] - a[1] for a, b in zip(points, points[1:])]
    positive = sorted(d for d in deltas if d > 0)
    median = positive[len(positive) // 2] if positive else 0.0
    limit = max(600.0, 10.0 * median)
    total, cum = 0.0, [0.0]
    for d in deltas:
        if 0 < d <= limit:
            total += d
        cum.append(total)
    return [p[0] for p in points], cum


def _ess_rhat(arr):
    """arr: (chains, draws). Returns bulk ESS, tail ESS, rank-normalised r-hat."""
    import arviz as az
    d = az.convert_to_dataset({"x": arr[:, :, None]})
    return (float(az.ess(d, method="bulk").x.values),
            float(az.ess(d, method="tail").x.values),
            float(az.rhat(d, method="rank").x.values))


def summarise(run_dir):
    import zarr
    ck = run_dir / "checkpoint"
    meta = json.loads((ck / "checkpoint_meta.json").read_text())
    cfg = json.loads((run_dir / "solver_params.json").read_text())["posterior_sampling"]
    rows = [json.loads(l) for l in (ck / "progress_log.jsonl").read_text().splitlines() if l.strip()]
    z = zarr.open(str(ck / "draws.zarr"), mode="r")

    out = {"run": run_dir.name, "chains": meta["n_chains"], "tune": meta["n_tune"],
           "warmup_done": meta["warmup_done"], "sampling_done": meta["sampling_done"],
           "phase": meta["phase"], "gpu": meta.get("gpu"),
           "gpu_speed_vs_a100": meta.get("gpu_speed_vs_a100", 1.0),
           "a100_equiv_seconds_total": meta.get("a100_equiv_seconds"),
           "compute_seconds_total": meta.get("compute_seconds")}

    # Phase timings from the progress log: cumulative seconds, so difference consecutive
    # rows within a phase. The first chunk of each phase includes compilation.
    f = out["gpu_speed_vs_a100"] or 1.0
    for phase, key in (("warmup", "warmup_done"), ("sampling", "sampling_done")):
        pts = [(r[key], r["t"]) for r in rows if r["phase"] == phase]
        if len(pts) >= 3:
            steps_l, cum_l = compute_seconds(pts)
            steps = np.array(steps_l, float)
            secs = np.array(cum_l, float)          # compute seconds, queue gaps removed
            per = np.diff(secs) / np.maximum(np.diff(steps), 1)
            out[f"{phase}_s_per_step_a100"] = float(np.median(per[1:]) * f)
            out[f"{phase}_span_s_a100"] = float(secs[-1] * f)
        else:
            out[f"{phase}_s_per_step_a100"] = None
            out[f"{phase}_span_s_a100"] = None

    names = list(z["sampling"].array_keys()) if "sampling" in z else []
    out["params"] = names
    out["posterior"], out["trajectory"] = {}, {}
    for n in names:
        a = np.asarray(z[f"sampling/{n}"][:])
        a = a.reshape(a.shape[0], a.shape[1]) if a.ndim == 3 and a.shape[2] == 1 else a
        if a.ndim != 2 or a.shape[1] < 20:
            continue
        b, t, r = _ess_rhat(a)
        out["posterior"][n] = {"ess_bulk": b, "ess_tail": t, "rhat": r,
                               "mean": float(a.mean()), "sd": float(a.std()),
                               "q025": float(np.quantile(a, 0.025)), "q975": float(np.quantile(a, 0.975))}
        # ESS as draws accumulate, paired with the sampling compute spent to get there
        samp = [(r_["sampling_done"], r_["t"]) for r_ in rows if r_["phase"] == "sampling"]
        samp_steps, samp_cum = compute_seconds(samp)
        traj = []
        for frac in np.linspace(1 / PREFIXES, 1.0, PREFIXES):
            k = max(20, int(a.shape[1] * frac))
            if k > a.shape[1]:
                continue
            eb, _, rh = _ess_rhat(a[:, :k])
            secs = next((c for dn, c in zip(samp_steps, samp_cum) if dn >= k), None)
            traj.append({"draws": k, "ess_bulk": eb, "rhat": rh,
                         "sampling_s_a100": float(secs * f) if secs is not None else None})
        out["trajectory"][n] = traj

    st = {}
    if "sampling_stats" in z:
        g = z["sampling_stats"]
        for k in ("step_size", "acceptance_rate", "n_steps", "diverging", "tree_depth"):
            if k in g:
                v = np.asarray(g[k][:])
                st[k] = {"median": float(np.median(v)), "mean": float(np.mean(v)), "max": float(np.max(v))}
    out["sampler_stats"] = st

    # Diagnostic power: disjoint groups of chains, same draws.
    out["groups"] = {}
    for n in names:
        a = np.asarray(z[f"sampling/{n}"][:])
        a = a.reshape(a.shape[0], a.shape[1]) if a.ndim == 3 and a.shape[2] == 1 else a
        if a.ndim != 2 or a.shape[1] < 50:
            continue
        per_size = {}
        for g in GROUP_SIZES:
            if a.shape[0] < 2 * g:       # need at least two disjoint groups to see spread
                continue
            vals = [_ess_rhat(a[i * g:(i + 1) * g]) for i in range(a.shape[0] // g)]
            per_size[str(g)] = {"rhat": [v[2] for v in vals], "ess_bulk": [v[0] for v in vals]}
        out["groups"][n] = per_size
    return out


def main():
    wanted = sys.argv[1:]
    runs = [BASE / w for w in wanted] if wanted else sorted(
        (d for d in BASE.iterdir() if (d / "checkpoint" / "checkpoint_meta.json").exists()),
        key=lambda d: int(re.search(r"_(\d+)chains", d.name).group(1)) if re.search(r"_(\d+)chains", d.name) else 0)
    out = {}
    for r in runs:
        try:
            out[r.name] = summarise(r)
        except Exception as e:                      # a run that has not checkpointed yet
            out[r.name] = {"run": r.name, "error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
