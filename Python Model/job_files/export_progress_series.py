"""Dump each run's progress log as {step -> cumulative COMPUTE seconds}, as JSON.

Exists because a progress log's raw "t" is wall-clock. Subtracting two of them
spans any queue wait and any preemption gap in between, so a locally-computed
rate silently includes time the job spent not running -- which is exactly the
error that made the first floor-vs-no-floor scaling fit wrong. warmup_status
already solves this against sacct (with --duplicates, so requeued intervals
aren't lost); this reuses that machinery and emits the result so a notebook on
a laptop can plot it without needing sacct itself.

    python3 export_progress_series.py > series.json
    python3 export_progress_series.py tightest tightest_nofloor > series.json

Shape:
    {set: {system: {"warmup":   {"5": 113.2, ...},
                    "sampling": {"100": 747.5, ...},
                    "n_tune": 1000, "converged_at": 700 | null,
                    "health": {...},
                    "dead_warmup": [10, 15, ...], "dead_sampling": [...],
                    "device_warmup": {"5": "NVIDIA A100-PCIE-40GB", ...},
                    "device_sampling": {...}}}}

"device_*" records which GPU wrote each checkpoint. Runs migrate between card
types across job segments -- one C6 run sat entirely on an H100 MIG slice while
its counterpart ran on A100s -- so a consumer that wants comparable timings has
to normalise per interval, not per run.

"dead_warmup"/"dead_sampling" list the checkpoint indices whose own block had
EVERY chain at zero acceptance. They are per-checkpoint, not per-run: a rate
computed across such a block measures how fast the sampler fails, not how fast
it samples, so those blocks have to leave the timing series entirely rather than
just be flagged on a plot.

"health" comes from draws.zarr/warmup_stats and exists to separate "slow" from
"not sampling". A chain whose acceptance rate is 0 with 100% divergences is not
producing draws at all -- it takes one leapfrog step, the solve fails, the
proposal is rejected, and the position never moves. Timing alone cannot tell
that apart from an expensive-but-working run, and treating the two the same
turns a broken configuration into a data point on a cost curve.
Sampling seconds are measured from the moment warmup finished, matching the
convention warmup_status prints, so they read directly as sampling cost.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warmup_status as W


def _health(system, suffix):
    """Sampler health from warmup_stats, or None if the zarr isn't readable."""
    p = "%s/Chain %s - %s/checkpoint/draws.zarr" % (W.BASE, system, suffix)
    if not os.path.exists(p):
        return None
    try:
        import zarr
        import numpy as np
        g = zarr.open(p, mode="r")
        ar = np.asarray(g["warmup_stats/acceptance_rate"][:])
        dv = np.asarray(g["warmup_stats/diverging"][:])
        ns = np.asarray(g["warmup_stats/n_steps"][:])
    except Exception:
        return None
    if ar.size == 0:
        return None
    # stored (chain, iter); transpose so axis 0 is iterations
    if ar.ndim == 2 and ar.shape[0] < ar.shape[1]:
        ar, dv, ns = ar.T, dv.T, ns.T
    n_chains = ar.shape[1] if ar.ndim == 2 else 1
    dead = int(sum(1 for c in range(n_chains) if ar[:, c].mean() < 1e-6))
    return {
        "acceptance": round(float(ar.mean()), 4),
        "divergence": round(float(dv.mean()), 4),
        "median_n_steps": int(np.median(ns)),
        "n_chains": n_chains,
        "dead_chains": dead,
        "iters": int(ar.shape[0]),
    }


def _per_checkpoint_stats(system, suffix, phase, every):
    """Per-checkpoint block summaries from warmup_stats/sampling_stats.

    n_steps summed over the block is the number of ODE solves the block paid
    for. Wall time divided by that is seconds per gradient evaluation -- the
    actual unit of work, and the only way to tell "took more leapfrog steps"
    apart from "each leapfrog step was expensive". A sampler iteration can hold
    anywhere from 1 to 100+ solves, so steps-per-second hides the difference.
    """
    p = "%s/Chain %s - %s/checkpoint/draws.zarr" % (W.BASE, system, suffix)
    if not os.path.exists(p) or every <= 0:
        return {}, {}, {}
    try:
        import zarr
        import numpy as np
        g = zarr.open(p, mode="r")
        ns = np.asarray(g["%s_stats/n_steps" % phase][:])
        ss = np.asarray(g["%s_stats/step_size" % phase][:])
        ar = np.asarray(g["%s_stats/acceptance_rate" % phase][:])
    except Exception:
        return {}, {}, {}
    if ns.size == 0:
        return {}, {}, {}
    if ns.ndim == 2 and ns.shape[0] < ns.shape[1]:
        ns, ss, ar = ns.T, ss.T, ar.T          # -> (iteration, chain)
    nblk = ns.shape[0] // every
    if nblk == 0:
        return {}, {}, {}
    cut = nblk * every
    solves = ns[:cut].reshape(nblk, every, -1).sum(axis=(1, 2))
    stepsz = ss[:cut].reshape(nblk, every, -1).mean(axis=(1, 2))
    accept = ar[:cut].reshape(nblk, every, -1).mean(axis=(1, 2))
    idx = [str((j + 1) * every) for j in range(nblk)]
    return ({k: int(v) for k, v in zip(idx, solves)},
            {k: float(v) for k, v in zip(idx, stepsz)},
            {k: round(float(v), 5) for k, v in zip(idx, accept)})


def _dead_checkpoints(system, suffix, phase, every):
    """Checkpoint indices whose block had all chains at zero acceptance.

    A checkpoint written at index k covers iterations (k-every, k], so block
    j = k/every - 1 of the per-iteration stats. Returned in checkpoint-index
    units so a caller can drop them straight out of a {index: seconds} series.
    """
    p = "%s/Chain %s - %s/checkpoint/draws.zarr" % (W.BASE, system, suffix)
    if not os.path.exists(p) or every <= 0:
        return []
    try:
        import zarr
        import numpy as np
        ar = np.asarray(zarr.open(p, mode="r")["%s_stats/acceptance_rate" % phase][:])
    except Exception:
        return []
    if ar.size == 0:
        return []
    if ar.ndim == 2 and ar.shape[0] < ar.shape[1]:
        ar = ar.T                      # -> (iteration, chain)
    n_it, n_ch = ar.shape if ar.ndim == 2 else (ar.shape[0], 1)
    n_blocks = n_it // every
    if n_blocks == 0:
        return []
    blocks = ar[:n_blocks * every].reshape(n_blocks, every, n_ch).mean(axis=1)
    return [int((j + 1) * every) for j in range(n_blocks)
            if bool((blocks[j] < 1e-6).all())]


def _n_tune(system, suffix):
    p = "%s/Chain %s - %s/solver_params.json" % (W.BASE, system, suffix)
    try:
        with open(p) as fh:
            return int(json.load(fh)["posterior_sampling"]["tune"])
    except Exception:
        return 1000


def _checkpoint_every(system, suffix):
    p = "%s/Chain %s - %s/solver_params.json" % (W.BASE, system, suffix)
    try:
        with open(p) as fh:
            return int(json.load(fh)["posterior_sampling"]["checkpoint_every_steps"])
    except Exception:
        return 5


def series_for_set(setkey):
    _, jobfile = W.SETS[setkey]
    pairs = W._job_pairs(jobfile)
    by_system = {}
    for system, jobid in pairs:
        by_system.setdefault(system, []).append(jobid)

    intervals_by_job = W.sacct_intervals([j for _, j in pairs])
    out = {}
    for system, jobids in by_system.items():
        # Per-system: a finished system may already be on the new post-rename
        # name while its still-running siblings in this set are not.
        suffix = W.resolve_suffix(setkey, system)
        rows = W._progress_rows(system, suffix)
        if not rows:
            continue
        ivs = W.productive_intervals(
            W.system_intervals(jobids, intervals_by_job), rows)
        if not ivs:
            continue

        n_tune = _n_tune(system, suffix)
        every = _checkpoint_every(system, suffix)
        dev_w, dev_s = {}, {}
        sv_w, sz_w, ac_w = _per_checkpoint_stats(system, suffix, "warmup", every)
        sv_s, sz_s, ac_s = _per_checkpoint_stats(system, suffix, "sampling", every)
        warm, samp, warm_done_at, converged_at = {}, {}, None, None
        for r in rows:
            if r.get("phase") == "done":
                # repeats the final counts; including it would make the last
                # delta zero and the last rate meaningless
                converged_at = r.get("sampling_done") or converged_at
                continue
            c = W.cumulative_at(r["t"], ivs)
            if c is None:
                continue
            w, s = r.get("warmup_done", 0), r.get("sampling_done", 0)
            dev = r.get("device", "unknown")
            if w and w not in warm:
                warm[w] = c
                dev_w[w] = dev
            if w >= n_tune and warm_done_at is None:
                warm_done_at = c
            if s and s not in samp:
                samp[s] = c
                dev_s[s] = dev

        if warm_done_at is not None:
            samp = {k: v - warm_done_at for k, v in samp.items()}
        else:
            samp = {}

        out[system] = {
            "warmup": {str(k): round(v, 2) for k, v in sorted(warm.items())},
            "sampling": {str(k): round(v, 2) for k, v in sorted(samp.items())},
            "n_tune": n_tune,
            "converged_at": converged_at,
            "health": _health(system, suffix),
            "dead_warmup": _dead_checkpoints(system, suffix, "warmup", every),
            "dead_sampling": _dead_checkpoints(system, suffix, "sampling", every),
            "device_warmup": {str(k): v for k, v in sorted(dev_w.items())},
            "device_sampling": {str(k): v for k, v in sorted(dev_s.items())},
            "solves_warmup": sv_w, "stepsize_warmup": sz_w, "accept_warmup": ac_w,
            "solves_sampling": sv_s, "stepsize_sampling": sz_s, "accept_sampling": ac_s,
        }
    return out


def main():
    wanted = sys.argv[1:] or list(W.SETS)
    bad = [s for s in wanted if s not in W.SETS]
    if bad:
        sys.exit("unknown set(s): %s\nknown: %s" % (", ".join(bad), ", ".join(W.SETS)))
    json.dump({s: series_for_set(s) for s in wanted}, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
