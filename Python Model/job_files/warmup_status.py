"""Status of the currently-running a1 inference comparisons.

Only shows systems that actually have a submitted job in each set's jobids
file. One table per set; sets are ordered so each prior width's floor table is
immediately followed by its no-floor table.

Timing columns are TOTAL COMPUTE TIME, summed across every run interval a
system has had -- not wall-clock since first launch, and not just the latest
interval. Two things make that distinction matter here:

  * A system can span several job IDs (resubmission, or a manual switch like a
    GPU change). List each one on its own line in the set's jobids file; they
    are stitched together in start order.
  * A single job ID on the preemptable QOS can be preempted and requeued, so it
    has MULTIPLE run intervals. `sacct` reports only the most recent one unless
    you pass --duplicates, which is why an earlier interval's checkpoints used
    to read as missing (C14/C14+unsat) or negative (C10).

Queue/idle time between intervals is excluded, since it isn't compute.

Warmup milestones are cumulative from the start of the run. Sampling
milestones are measured from the moment warmup finished, so they read directly
as sampling cost and stay comparable across runs whose warmups differed.

A STATUS column separates "converged", "DIED" (all chains at zero acceptance --
alive in slurm but producing nothing), "running", "queued", and "stalled".

    python3 warmup_status.py              # all sets, table form
    python3 warmup_status.py tightest     # just one set
    python3 warmup_status.py --parsable   # machine-readable
"""
import json
import os
import re
import subprocess
import sys
import time

BASE = "/projects/anth4580/Bayesian/Results/Chain Scaling Tests"
JOBDIR = "/projects/anth4580/Bayesian/job_files"
WARMUP_MILESTONES = [5, 50, 100, 500, 1000]
SAMPLING_MILESTONES = [100, 500, 1000]

SETS = {
    "tightest":          ("a1 tightest",                f"{JOBDIR}/inference_a1_tightest_jobids.txt"),
    "tightest_nofloor":  ("a1 tightest nofloor-eqxnan",  f"{JOBDIR}/inference_a1_tightest_nofloor_eqxnan_jobids.txt"),
    "narrowest_matched": ("a1 narrowest matched",        f"{JOBDIR}/inference_a1_narrowest_matched_jobids.txt"),
    "narrowest_nofloor": ("a1 narrowest nofloor-eqxnan", f"{JOBDIR}/inference_a1_narrowest_nofloor_eqxnan_jobids.txt"),
    # Multi-parameter pilot (C6/C10/C14, no-floor only, tune=600, r-hat-only
    # convergence) -- see build_multiparam_config.py. The "suffix" here is
    # never actually used for path-building (resolve_suffix short-circuits on
    # the *_no_floor pattern), only jobids_file matters for these two.
    "a1c2_no_floor": ("a1c2_no_floor", f"{JOBDIR}/inference_a1c2_no_floor_jobids.txt"),
    "a1c3_no_floor": ("a1c3_no_floor", f"{JOBDIR}/inference_a1c3_no_floor_jobids.txt"),
    # 3-parameter runs (same C6/C10/C14, no-floor, tune=600, r-hat-only). a2 is
    # an ordinary multiplicative group; d1 is additive inside exp() and so
    # carries a Normal prior around its 0.0 nominal rather than a median-1.0
    # LogNormal -- see build_multiparam_config.py's rate-window conversion.
    "a1c3a2_no_floor": ("a1c3a2_no_floor", f"{JOBDIR}/inference_a1c3a2_no_floor_jobids.txt"),
    "a1c3d1_no_floor": ("a1c3d1_no_floor", f"{JOBDIR}/inference_a1c3d1_no_floor_jobids.txt"),
}
SET_LABEL = {
    "tightest": "a1 [0.1,10] sigma=1.18 -- FLOOR",
    "tightest_nofloor": "a1 [0.1,10] -- NO-FLOOR + EQX_ON_ERROR=nan",
    "narrowest_matched": "a1 [0.05,20], tightest's tune/draws/chains -- FLOOR",
    "narrowest_nofloor": "a1 [0.05,20] -- NO-FLOOR + EQX_ON_ERROR=nan",
    "a1c2_no_floor": "a1+c2 pilot -- NO-FLOOR, tune=600, r-hat-only convergence",
    "a1c3_no_floor": "a1+c3 pilot -- NO-FLOOR, tune=600, r-hat-only convergence",
    "a1c3a2_no_floor": "a1+c3+a2 -- NO-FLOOR, tune=600, r-hat-only convergence",
    "a1c3d1_no_floor": "a1+c3+d1 (d1 Normal, additive-in-exp) -- NO-FLOOR, tune=600, r-hat-only",
}

# 2026-09-09 renaming: result folders are moving from the qualitative
# tightest/narrowest naming (SETS above) to prior-value-based names, but only
# for a system once it's finished (converged or DIED) -- a run still writing
# checkpoints keeps its old directory name until it's done, so it isn't
# renamed out from under a live process. That means, for a while, systems
# within the SAME set can be split across both naming generations. Every
# per-system path lookup below resolves through here rather than assuming one
# suffix for the whole set: try the new name on disk first, fall back to the
# set's original suffix if the new one doesn't exist yet.
NEW_SUFFIX = {
    "tightest": "a1_0.1-10_floor",
    "tightest_nofloor": "a1_0.1-10_no_floor",
    "narrowest_matched": "a1_0.05-20_floor",
    "narrowest_nofloor": "a1_0.05-20_no_floor",
}


def resolve_suffix(setname, sysname):
    """The CURRENT on-disk suffix for this (set, system), new-name-first."""
    new = NEW_SUFFIX.get(setname)
    if new and os.path.isdir(f"{BASE}/Chain {sysname} - {new}"):
        return new
    return SETS[setname][0]


def _chain_sort_key(name):
    """Ascending chain length, saturated before +unsat at the same length."""
    m = re.match(r"C(\d+)", name)
    n = int(m.group(1)) if m else 999
    return (n, "+unsat" in name, name)


def _job_pairs(jobids_file):
    """[(system, jobid), ...]. A system may appear on several lines; all of its
    job IDs are stitched together in start order."""
    if not os.path.exists(jobids_file):
        return []
    pairs = []
    for line in open(jobids_file):
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def _parse_dt(s):
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def _parse_elapsed(s):
    """'HH:MM:SS' or 'D-HH:MM:SS' -> seconds."""
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        h, m, sec = (int(x) for x in s.split(":"))
        return days * 86400 + h * 3600 + m * 60 + sec
    except (ValueError, AttributeError):
        return None


def _by_cluster(jobids):
    """{cluster: {bare id: id as written}}. Jobs submitted through gpu_submit.sh can
    run on Alpine; their jobids lines read "alpine:<id>", bare ids are Blanca."""
    groups = {}
    for jid in set(jobids):
        cluster, _, bare = jid.rpartition(":")
        groups.setdefault(cluster or "blanca", {})[bare] = jid
    return groups


def sacct_intervals(jobids):
    """jobid -> [(start_epoch, end_epoch_or_None, elapsed_seconds), ...].

    --duplicates is load-bearing: without it a preempted-and-requeued job
    reports only its most recent interval, silently losing whichever
    checkpoints its earlier interval wrote.
    """
    out = {}
    if not jobids:
        return out
    for cluster, ids_map in _by_cluster(jobids).items():
        ids = ",".join(sorted(ids_map))
        try:
            # NOT capture_output=: login-node python3 is 3.6, where that kwarg
            # does not exist. Loading the cluster's own slurm module makes it
            # local so sacct needs no cross-cluster accounting-DB lookup (unload
            # first: a plain load over another cluster's module keeps its config).
            proc = subprocess.run(
                ["bash", "-lc",
                 f"module unload slurm >/dev/null 2>&1; module load slurm/{cluster} >/dev/null 2>&1; "
                 f"sacct -D -j {ids} --format=JobID,Start,End,Elapsed -X --noheader -P"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
            raw = proc.stdout.decode("utf-8", "replace")
        except Exception as e:
            print(f"WARNING sacct lookup ({cluster}) failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for line in raw.splitlines():
            f = line.split("|")
            if len(f) != 4:
                continue
            bare, start, end, elapsed = (x.strip() for x in f)
            st = _parse_dt(start)
            if st is None or bare not in ids_map:
                continue
            out.setdefault(ids_map[bare], []).append((st, _parse_dt(end), _parse_elapsed(elapsed) or 0))
    for jid in out:
        out[jid].sort(key=lambda iv: iv[0])
    return out


def squeue_states(jobids):
    """jobid -> (state, node_or_reason) for jobs still in the queue.

    Only queued jobs appear; anything absent has already left the queue, which
    is how a finished run is told apart from one that is merely not running.
    squeue's own %R is dual-purpose -- the node name for a RUNNING job, the
    hold reason (e.g. "(Priority)", "(Resources)") for a PENDING one -- so one
    field covers both "where is it" and "why is it stuck", same as reading
    squeue by hand.
    """
    if not jobids:
        return {}
    out = {}
    for cluster, ids_map in _by_cluster(jobids).items():
        try:
            proc = subprocess.run(
                ["bash", "-lc",
                 f"module unload slurm >/dev/null 2>&1; module load slurm/{cluster} >/dev/null 2>&1; "
                 "squeue -u $USER -h -o '%i|%T|%R'"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            for line in proc.stdout.decode("utf-8", "replace").splitlines():
                f = line.split("|")
                if len(f) == 3 and f[0].strip() in ids_map:
                    out[ids_map[f[0].strip()]] = (f[1].strip(), f[2].strip())
        except Exception as e:
            print(f"WARNING squeue lookup ({cluster}) failed: {type(e).__name__}: {e}", file=sys.stderr)
    return out


def node_or_reason(jids, qstates):
    """The %R field (node if running, hold reason if pending) for whichever
    of this system's job IDs is currently in the queue, or '-' if none are."""
    for j in reversed(jids):
        hit = qstates.get(j)
        if hit:
            return hit[1]
    return "-"


_CONVERGENCE_LINE = re.compile(
    r"r_hat/ESS check at (\d+) sampling draws: r_hat=([\d.]+) ess_bulk=([\d.]+)"
    r"(?:.*?EXCLUDED (\d+)/(\d+) stranded)?"
)


def latest_convergence_check(jids):
    """(draw, r_hat, ess, n_stranded, n_chains) from the most recent
    "r_hat/ESS check" line logged by any of this system's job IDs, or None.

    Reads the .out log directly rather than recomputing r-hat/ESS from the
    zarr arrays -- the sampler already computed and printed this once per
    check (every rhat_check_every draws), so re-deriving it here would cost
    an arviz call for information that's already sitting in a text file.
    Globs by job ID rather than a fixed job-name prefix so it works for both
    the floor ("floorreal") and no-floor ("nfeqx") sbatch scripts.
    """
    import glob
    best = None
    for j in jids:
        matches = glob.glob(f"{JOBDIR}/chain_scaling_tests/*.{j.rpartition(':')[2]}.out")
        for path in matches:
            try:
                with open(path, "r", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            found = _CONVERGENCE_LINE.findall(text)
            if found:
                best = found[-1]  # last check in this log; jids are in start order
    if best is None:
        return None
    draw, rhat, ess, n_stranded, n_chains = best
    return (int(draw), float(rhat), float(ess),
            int(n_stranded) if n_stranded else 0,
            int(n_chains) if n_chains else 0)


_BAYESIAN_PY = "/projects/anth4580/software/anaconda/envs/Bayesian/bin/python"


def sampler_health_batch(paths):
    """{path: (acceptance, dead_chains, n_chains)} from draws.zarr. Missing
    paths are absent from the result -- a dead run reads as (None, 0, 0).

    ONE subprocess for every path, not one per system: profiled at ~1.2s/spawn
    (interpreter startup, not the read), and the old per-system version was
    34 spawns / 41s of a 50s total run. Pinned to the Bayesian env's python
    since bare `python3` on the login node has no zarr -- that used to fail
    silently and misreport a 0%-acceptance run as "stalled" instead of "DIED".
    """
    paths = sorted(p for p in set(paths) if os.path.exists(p))
    if not paths:
        return {}
    script = (
        "import sys, json\n"
        "import zarr, numpy as np\n"
        "out = {}\n"
        "for p in sys.argv[1:]:\n"
        "    try:\n"
        "        g = zarr.open(p, mode='r')\n"
        "        ar = np.asarray(g['warmup_stats/acceptance_rate'][:])\n"
        "        if ar.size == 0:\n"
        "            out[p] = None; continue\n"
        "        if ar.ndim == 2 and ar.shape[0] < ar.shape[1]:\n"
        "            ar = ar.T\n"
        "        n_chains = ar.shape[1] if ar.ndim == 2 else 1\n"
        "        dead = int(sum(1 for c in range(n_chains) if ar[:, c].mean() < 1e-6))\n"
        "        out[p] = [float(ar.mean()), dead, n_chains]\n"
        "    except Exception:\n"
        "        out[p] = None\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run([_BAYESIAN_PY, "-c", script] + paths,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60)
        raw = json.loads(proc.stdout.decode("utf-8", "replace").strip() or "{}")
    except Exception:
        return {}
    out = {}
    for p, v in raw.items():
        if v is not None:
            out[p] = (v[0], int(v[1]), int(v[2]))
    return out


def run_status(prog, warm, samp, n_tune, jids, qstates, accept, dead, n_chains):
    """One word for where this run stands.

    Order matters: DIED outranks RUNNING because a job can be very much alive
    in slurm while its sampler has stopped producing draws entirely -- that is
    precisely the case worth surfacing.
    """
    done = bool(prog) and prog[-1].get("phase") == "done"
    queued = [q for q in (qstates.get(j) for j in jids) if q]
    if done:
        return "converged" if (samp or 0) > 0 else "finished"
    if n_chains and dead >= n_chains:
        return "DIED"
    if any(state == "RUNNING" for state, _node_or_reason in queued):
        return "running"
    if queued:
        return "queued"
    return "stalled"


def system_intervals(jobids, intervals_by_job):
    """All run intervals for one system, across all its job IDs, in start order."""
    ivs = []
    for jid in jobids:
        ivs.extend(intervals_by_job.get(jid, []))
    ivs.sort(key=lambda iv: iv[0])
    return ivs


def productive_intervals(intervals, rows):
    """Intervals from the first one that actually produced a checkpoint onward.

    A leading interval that was cancelled before writing anything (e.g. killed
    during the 100k-draw prior sampling that runs before warmup) burned real
    cluster time but made no progress. Counting it would inflate every
    milestone by that dead time and make a floor-vs-no-floor comparison
    meaningless -- the floor runs have such aborted first attempts, the
    no-floor ones mostly don't.
    """
    if not rows or not intervals:
        return intervals
    t_first = rows[0]["t"]
    for i, (start, end, _) in enumerate(intervals):
        if start <= t_first and (end is None or t_first <= end + 60):
            return intervals[i:]
    return intervals


def cumulative_at(t, intervals):
    """Total compute seconds accrued by wall-clock time t, or None if t falls
    outside every known interval (i.e. some segment isn't being tracked)."""
    prior = 0.0
    for start, end, elapsed in intervals:
        if start <= t and (end is None or t <= end + 60):
            return prior + (t - start)
        prior += elapsed
    return None


def total_compute(intervals):
    return sum(iv[2] for iv in intervals)


def _progress_rows(sysname, suffix):
    p = f"{BASE}/Chain {sysname} - {suffix}/checkpoint/progress_log.jsonl"
    if not os.path.exists(p):
        return []
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return []


def milestone_times(rows, intervals, n_tune):
    """(warmup_milestone -> secs, sampling_milestone -> secs).

    Warmup times are cumulative compute from the start. Sampling times are
    measured from when warmup finished.
    """
    warm_out = {m: None for m in WARMUP_MILESTONES}
    samp_out = {m: None for m in SAMPLING_MILESTONES}
    if not rows or not intervals:
        return warm_out, samp_out

    warm_at, samp_at, warmup_done_t = {}, {}, None
    for r in rows:
        c = cumulative_at(r["t"], intervals)
        if c is None:
            continue
        w, s = r.get("warmup_done", 0), r.get("sampling_done", 0)
        if w not in warm_at:
            warm_at[w] = c
        if w >= n_tune and warmup_done_t is None:
            warmup_done_t = c
        if s > 0 and s not in samp_at:
            samp_at[s] = c

    for m in WARMUP_MILESTONES:
        for d in sorted(warm_at):
            if d >= m:
                warm_out[m] = warm_at[d]
                break
    if warmup_done_t is not None:
        for m in SAMPLING_MILESTONES:
            for d in sorted(samp_at):
                if d >= m:
                    samp_out[m] = samp_at[d] - warmup_done_t
                    break
    return warm_out, samp_out


def phase_rate(rows, phase_key, intervals):
    """Seconds per step for the given phase, from its last two log entries.

    Keeps only the first sighting of each count: a finished run's last entry is
    a phase="done" record repeating the final count, so taking the raw last two
    rows gives a zero delta and no rate at all.
    """
    seen, pts = set(), []
    for r in rows:
        v = r.get(phase_key, 0)
        if v > 0 and v not in seen:
            c = cumulative_at(r["t"], intervals)
            if c is not None:
                seen.add(v)
                pts.append((v, c))
    if len(pts) < 2:
        return None
    (v0, c0), (v1, c1) = pts[-2], pts[-1]
    if v1 <= v0:
        return None
    return (c1 - c0) / (v1 - v0)


def _by_system(pairs):
    """{system: [jobid, ...]} in file order, each job ID once. The same ID can be
    recorded twice (resubmit_if_needed.sh logs a successor at submit time, and
    the job logs itself again when it starts), and system_intervals() sums every
    interval per listed ID -- so without this a duplicate doubles compute time."""
    by_system = {}
    for sysname, jid in pairs:
        jids = by_system.setdefault(sysname, [])
        if jid not in jids:
            jids.append(jid)
    return by_system


def collect(setname, qstates, intervals_by_job, health_cache):
    """qstates/intervals_by_job/health_cache: fetched once in main() for all
    sets being shown, not re-fetched per set."""
    _, jobids_file = SETS[setname]
    pairs = _job_pairs(jobids_file)
    by_system = _by_system(pairs)

    rows_out = []
    for sysname, jids in by_system.items():
        # Per-system, not per-set: a finished system may already be on the new
        # name while its still-running siblings in this same set are not.
        suffix = resolve_suffix(setname, sysname)
        intervals = system_intervals(jids, intervals_by_job)
        meta = f"{BASE}/Chain {sysname} - {suffix}/checkpoint/checkpoint_meta.json"
        warm, samp, n_tune = 0, 0, 1000
        if os.path.exists(meta):
            try:
                d = json.load(open(meta))
                warm = d["warmup_done"]; samp = d["sampling_done"]; n_tune = d["n_tune"]
            except Exception as e:
                print(f"WARNING {sysname}: {type(e).__name__}: {e}", file=sys.stderr)
        prog = _progress_rows(sysname, suffix)
        intervals = productive_intervals(intervals, prog)
        w_ms, s_ms = milestone_times(prog, intervals, n_tune)
        w_rate = phase_rate(prog, "warmup_done", intervals)
        s_rate = phase_rate(prog, "sampling_done", intervals)
        eta = ((n_tune - warm) * w_rate / 3600.0) if (w_rate and warm < n_tune) else None
        zpath = f"{BASE}/Chain {sysname} - {suffix}/checkpoint/draws.zarr"
        accept, dead, n_chains = health_cache.get(zpath, (None, 0, 0))
        status = run_status(prog, warm, samp, n_tune, jids, qstates,
                            accept, dead, n_chains)
        rows_out.append(dict(system=sysname, elapsed=total_compute(intervals),
                             warm=warm, samp=samp, n_tune=n_tune,
                             w_ms=w_ms, s_ms=s_ms, w_rate=w_rate, s_rate=s_rate, eta=eta,
                             status=status, accept=accept,
                             dead=dead, n_chains=n_chains,
                             jobid=jids[-1] if jids else "-",
                             node=node_or_reason(jids, qstates),
                             convergence=latest_convergence_check(jids)))
    rows_out.sort(key=lambda r: _chain_sort_key(r["system"]))
    return rows_out


def _hms(seconds):
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _rate_str(r):
    if r.get("status") == "DIED":
        return f"accept {r['accept']:.3f}" if r.get("accept") is not None else "no draws"
    """Warmup rate + ETA while warming; sampling rate once sampling."""
    if r["warm"] < r["n_tune"]:
        if r["w_rate"] is None:
            return "-"
        eta = f" {r['eta']:.1f}h" if r["eta"] is not None else ""
        return f"{r['w_rate']:.0f}s{eta}"
    if r["s_rate"] is not None:
        return f"{r['s_rate']:.1f}s/draw"
    return "-"


def _rhat_str(r):
    """'1.0004/7835' (r_hat/ess_bulk) from the most recent live convergence
    check, or '-' if none has run yet (e.g. still early in warmup)."""
    c = r.get("convergence")
    if not c:
        return "-"
    _draw, rhat, ess, _n_str, _n_ch = c
    return f"{rhat:.4f}/{ess:.0f}"


def _stranded_str(r):
    """'1/8' if the last convergence check excluded any chains, else '-'."""
    c = r.get("convergence")
    if not c:
        return "-"
    _draw, _rhat, _ess, n_stranded, n_chains = c
    return f"{n_stranded}/{n_chains}" if n_stranded else "-"


def main():
    which = [a for a in sys.argv[1:] if a in SETS] or list(SETS)

    # Fetch squeue/sacct/zarr-health once for all sets, not per set/system --
    # profiled at 50s/run, 41s of it 34 redundant per-system subprocess spawns.
    # Cuts the subprocess count to 3 regardless of how many sets are shown.
    all_pairs = {s: _job_pairs(SETS[s][1]) for s in which}
    all_jids = [jid for pairs in all_pairs.values() for _, jid in pairs]
    qstates = squeue_states(all_jids)
    intervals_by_job = sacct_intervals(all_jids)

    zpaths = set()
    for s in which:
        for sysname in _by_system(all_pairs[s]):
            suffix = resolve_suffix(s, sysname)
            zpaths.add(f"{BASE}/Chain {sysname} - {suffix}/checkpoint/draws.zarr")
    health_cache = sampler_health_batch(zpaths)

    all_rows = {s: collect(s, qstates, intervals_by_job, health_cache) for s in which}

    # A baseline (floor) set often has historical data for systems that aren't
    # part of any active comparison. Restrict it to systems that appear in a
    # "_nofloor" set -- show only what's actually being compared.
    compared = {r["system"] for s in which if s.endswith("_nofloor") for r in all_rows[s]}
    if compared:
        for s in which:
            if not s.endswith("_nofloor"):
                all_rows[s] = [r for r in all_rows[s] if r["system"] in compared]

    if "--parsable" in sys.argv:
        for s in which:
            for r in all_rows[s]:
                w = "|".join(_hms(r["w_ms"][k]) for k in WARMUP_MILESTONES)
                sm = "|".join(_hms(r["s_ms"][k]) for k in SAMPLING_MILESTONES)
                print(f"{s}|{r['system']}|{r.get('status','?')}|"
                      f"{r['warm']}|{r['samp']}|{r['n_tune']}|"
                      f"{_hms(r['elapsed'])}|{w}|{sm}|{_rate_str(r)}")
        return

    # Narrow enough for a normal terminal without wrapping: this table is for
    # "how is my job doing right now" (an alternative to squeue, plus the
    # convergence state squeue can't show). Per-milestone timing history
    # (W5/W50/.../S1000) is still available via --parsable and the export
    # scripts, which is where that data actually gets used for real analysis
    # -- printing it here every time was most of this table's width for the
    # least-used information in it.
    W_SYS, W_ST, W_JOB, W_NODE, W_WS, W_SAMP, W_RHAT, W_STR, W_RATE, W_EL = \
        11, 11, 10, 14, 11, 6, 13, 7, 12, 11
    for s in which:
        rows = all_rows[s]
        if not rows:
            continue
        hdr = (f"{'SYSTEM':<{W_SYS}}{'STATUS':<{W_ST}}{'JOBID':<{W_JOB}}"
               f"{'NODE/REASON':<{W_NODE}}{'WARMUP':<{W_WS}}{'DRAWS':<{W_SAMP}}"
               f"{'R-HAT/ESS':>{W_RHAT}}{'STUCK':>{W_STR}}"
               f"{'RATE/ETA':>{W_RATE}}{'COMPUTE':>{W_EL}}")
        print()
        print(f"SET '{s}'  {SET_LABEL.get(s, s)}")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['system']:<{W_SYS}}{r.get('status','?'):<{W_ST}}"
                  f"{r.get('jobid','-'):<{W_JOB}}{r.get('node','-'):<{W_NODE}}"
                  f"{str(r['warm'])+'/'+str(r['n_tune']):<{W_WS}}{str(r['samp']):<{W_SAMP}}"
                  f"{_rhat_str(r):>{W_RHAT}}{_stranded_str(r):>{W_STR}}"
                  f"{_rate_str(r):>{W_RATE}}{_hms(r['elapsed']):>{W_EL}}")
        print("-" * len(hdr))
    print("\nSTATUS  converged = stopped on r-hat     DIED = every chain at zero acceptance")
    print("        running/queued = live in slurm   stalled = unfinished with no job queued")
    print("\nR-HAT/ESS, STUCK from the most recent live convergence check logged")
    print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
