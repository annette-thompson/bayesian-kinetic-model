"""Status of the Tier-1 runs: one line per run, joining Slurm on both clusters with each run's
checkpoint and job log, so it says what squeue cannot.

For each run:
  WHERE     the running copy's cluster, node, GPU and time left in its segment, or how many
            copies are queued and the earliest estimated start
  PROGRESS  phase and steps (warmup n/300, sampling n/draws), with the current rate and, in
            warmup, when warmup should end
  CHECK     the latest r-hat / bulk-ESS check from the job log, and the pass streak
  A100-H    A100-equivalent compute used, and the whole run's estimate from cost_estimate.py
            (measured rates, every condition solved to 720 s); CAP if that passes max_total_hours
  ETA       a running run: when it should finish, or when its segment ends and how many more it
            needs; a queued run: how long it needs once it starts, and, once enough queue waits
            have ended, Slurm's estimated start corrected by its measured error (cost_estimate.py)
  NOTE      preemptions, stranded chains, errors, a stalled checkpoint; for finished runs, the
            recovery report's z and 95% coverage per parameter

States: DONE (finalized), FINALIZING, RUNNING, QUEUED, RESUBMIT (stopped at the segment's
wall clock with nothing queued: rerun tier1/submit_stage2.sh), CAPPED (hit its compute cap;
final, not resubmitted), FAILED (the job ended with an error and nothing is queued), SLOW (no
checkpoint for much longer than usual, but its GPU is busy: long trees, e.g. on a ridge),
STALLED (the same with an idle GPU, or one that could not be checked), NOT STARTED.

Usage (on the cluster, from /projects/anth4580/Bayesian/job_files):
    python3 tier1/tier1_status.py              # runs that have started, finished or are queued
    python3 tier1/tier1_status.py --all        # also runs never submitted
    python3 tier1/tier1_status.py R1 R4 sbc01  # filter: run groups (R0-R8) or name substrings
    python3 tier1/tier1_status.py --check      # same table; exit 3 if anything needs action
                                               # (RESUBMIT, FAILED, STALLED), 4 if all are done

Every call appends each queued run's earliest estimated start (the "est." in WHERE) to
tier1/start_predictions.jsonl, which tier1/start_accuracy.py checks against the actual starts;
--no_log skips that.

Written for the login node's Python 3.6: standard library only.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

BASE = "/projects/anth4580/Bayesian"
RESULTS = os.path.join(BASE, "Results", "Tier1")
LOGS = os.path.join(BASE, "job_files", "tier1")
START_LOG = os.path.join(LOGS, "start_predictions.jsonl")
SEGMENT_H = 11.5           # tier1.sbatch's per-segment budget (MAXH)
STALL_FLOOR_MIN = 45       # a running run with no checkpoint for this long is flagged...
STALL_FACTOR = 8           # ...or for this many times its usual checkpoint interval, if longer

GROUPS = [                 # (id, pattern on the run name), first match wins
    ("R4", r"C8_sbc\d+"),
    ("R5", r"C8_noise\d+|prior\+\d+sd"),
    ("R8", r"ta0\.95|rtol1e-5"),
    ("R3", r"d1d2"),
    ("R7", r"a1c3 - (profile|rates)$"),
    ("R6", r"c3split|a1c3sc3l"),
    ("R2", r"C14\+unsat - a1c3a2$"),
    ("R1", r"C14\+unsat - a1c3$"),
    ("R0", r"^Tier1 C8 - a1c3$"),
]


def group_of(run):
    for gid, pat in GROUPS:
        if re.search(pat, run):
            return gid
    return "--"


def slug_of(run):
    """Job name submit_stage2.sh gives a run."""
    return "tier1_" + re.sub(" ", "_", run[len("Tier1 "):].replace(" - ", "_"))


def sh(cmd, env=None):
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, env=env, timeout=60)
        return out.stdout
    except Exception:
        return ""


def cluster_env(cluster):
    env = dict(os.environ)
    env["SLURM_CONF"] = "/curc/slurm/%s/etc/slurm.conf" % cluster
    return env


def parse_time(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def parse_elapsed(s):
    """Slurm [d-]hh:mm:ss or mm:ss -> hours."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 24 + parts[0] + parts[1] / 60.0 + parts[2] / 3600.0


def slurm_jobs():
    """{job name: [job dict, ...]} for this user's queued and running jobs on both clusters."""
    jobs = {}
    for cl in ("blanca", "alpine"):
        out = sh(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j|%T|%M|%N|%b|%r|%S"],
                 env=cluster_env(cl))
        for line in out.splitlines():
            f = line.split("|")
            if len(f) < 8:
                continue
            jobs.setdefault(f[1], []).append(dict(
                cluster=cl, id=f[0], state=f[2], elapsed=f[3], node=f[4], gres=f[5],
                reason=f[6], start=parse_time(f[7])))
    return jobs


def preemptions():
    """{job name: preemption count} over the last week, from sacct's duplicate records."""
    counts = {}
    for cl in ("blanca", "alpine"):
        out = sh(["sacct", "-u", os.environ.get("USER", ""), "-S", "now-7days", "--duplicates", "-X",
                  "-n", "-P", "-o", "JobName,State"], env=cluster_env(cl))
        for line in out.splitlines():
            f = line.split("|")
            if len(f) == 2 and f[1].startswith("PREEMPTED"):
                counts[f[0]] = counts.get(f[0], 0) + 1
    return counts


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def progress(run_dir):
    rows = []
    try:
        with open(os.path.join(run_dir, "checkpoint", "progress_log.jsonl")) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def rate(rows, phase, open_until=None, every=5):
    """(seconds per step/draw, median seconds between checkpoints) over the latest rows of a phase.
    For a running run (`open_until` = now), time since the latest checkpoint counts as one more
    chunk of `every` steps once it is longer than the usual interval."""
    key = "warmup_done" if phase == "warmup" else "sampling_done"
    r = [x for x in rows if x.get("phase") == phase][-11:]
    if len(r) < 2:
        return None, None
    dts = [b["t"] - a["t"] for a, b in zip(r, r[1:])]
    dsteps = r[-1][key] - r[0][key]
    span = r[-1]["t"] - r[0]["t"]
    dts.sort()
    gap = dts[len(dts) // 2]
    if open_until and dsteps > 0 and open_until - r[-1]["t"] > gap:
        span += open_until - r[-1]["t"]
        dsteps += every
    per = span / dsteps if dsteps > 0 else None
    return per, gap


def winner_log(slug):
    """The newest log of the copy that actually ran (not a copy that exited at the claim)."""
    for path in sorted(glob.glob(os.path.join(LOGS, "%s.*.out" % glob.escape(slug))),
                       key=os.path.getmtime, reverse=True):
        try:
            with open(path, errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if "==> Tier1 " in text or "=== Resumable BlackJAX Sampling ===" in text:
            return path, text
    return None, ""


def log_facts(text, run):
    facts = {}
    checks = re.findall(r"r_hat/ESS check at (\d+) sampling draws: r_hat=([\d.]+) ess_bulk=([\d.]+).*?\| "
                        r"(FAIL|pass) streak (\d+)/(\d+)", text)
    if checks:
        n, rh, ess, _, k, need = checks[-1]
        facts["check"] = "r-hat %s ESS %.0f @%s (%s/%s)" % (rh, float(ess), n, k, need)
    m = re.findall(r"gpu=([^|\n]+)", text)
    if m:
        facts["gpu"] = m[-1].strip().replace("NVIDIA ", "")
    if re.search(r"^Traceback|^\w*Error: ", text, re.M):
        err = re.findall(r"^(\w*Error: .*)$", text, re.M)
        facts["error"] = (err[-1] if err else "Traceback in log")[:70]
    rec = re.findall(r"^%s\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+(yes|no)"
                     % re.escape(run), text, re.M)
    if rec:
        facts["recovery"] = ", ".join("%s z %+.2f %s" % (p, float(z), "in 95%" if ok == "yes" else "OUT of 95%")
                                      for p, _, _, _, z, _, ok in rec)
    return facts


GPU_SHORT = [("A100-PCIE-40GB", "A100 40GB"), ("A100-SXM4-40GB", "A100 40GB"), ("A100 80GB PCIe", "A100 80GB"),
             ("A100-SXM4-80GB", "A100 80GB"), ("H100 NVL", "H100"), ("H200", "H200")]
REASONS = {
    "Priority": "waiting behind higher-priority jobs",
    "Resources": "waiting for a free GPU",
    "QOSGrpGRES": "the QOS's group GPU limit is in use",
    "QOSMaxGRESPerUser": "your per-user GPU limit is in use",
    "BeginTime": "requeued after preemption; restarts shortly",
}


def gpu_busy(cluster, job_id):
    """GPU utilisation (%) inside a running job, via srun --overlap; None if it cannot be read."""
    out = sh(["srun", "--jobid=%s" % job_id, "--overlap", "-n1", "nvidia-smi",
              "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], env=cluster_env(cluster))
    vals = [int(x) for x in re.findall(r"^\s*(\d+)\s*$", out, re.M)]
    return max(vals) if vals else None


def short_gpu(name):
    for long, short in GPU_SHORT:
        if long in name:
            return short
    return name


def hms(hours):
    if hours is None:
        return "?"
    m = int(round(hours * 60))
    return "%d:%02d" % (m // 60, m % 60)


def describe(run, jobs, preempt, now):
    run_dir = os.path.join(RESULTS, run)
    slug = slug_of(run)
    est_start, reasons, seg_left, speed_now = None, [], None, None
    meta = read_json(os.path.join(run_dir, "checkpoint", "checkpoint_meta.json")) or {}
    status = read_json(os.path.join(run_dir, "checkpoint", "status.json")) or {}
    cfg = read_json(os.path.join(run_dir, "solver_params.json")) or {}
    cap = (cfg.get("posterior_sampling") or {}).get("max_total_hours")
    finalized = os.path.exists(os.path.join(run_dir, "posterior_samples_pm.nc"))
    rows = progress(run_dir)
    mine = jobs.get(slug, [])
    running = [j for j in mine if j["state"] in ("RUNNING", "COMPLETING")]
    pending = [j for j in mine if j["state"] == "PENDING"]
    log_path, text = winner_log(slug)
    facts = log_facts(text, run)
    used = (meta.get("a100_equiv_seconds") or 0) / 3600.0
    notes = []
    if preempt.get(slug):
        notes.append("preempted %dx" % preempt[slug])
    if status.get("stranded_chains"):
        notes.append("stranded chains %s" % status["stranded_chains"])

    phase = meta.get("phase")
    if phase == "warmup":
        prog = "warmup %s/%s" % (meta.get("warmup_done", 0), meta.get("n_tune", "?"))
    elif phase == "sampling":
        prog = "sampling %s/%s" % (meta.get("sampling_done", 0), meta.get("n_draws", "?"))
    elif phase == "done":
        prog = "sampled %s draws" % meta.get("sampling_done", "?")
    else:
        prog = "-"
    rate_txt = ""
    every = (cfg.get("posterior_sampling") or {}).get("checkpoint_every_steps", 5)
    live = now.timestamp() if running else None
    per, gap = rate(rows, phase, live, every) if phase in ("warmup", "sampling") else (None, None)
    if per:
        if phase == "warmup":
            left = (meta.get("n_tune", 0) - meta.get("warmup_done", 0)) * per / 3600.0
            rate_txt = "%.0f s/step, warmup ends in ~%s" % (per, hms(left))
        else:
            rate_txt = "%.1f s/draw, %s per 100" % (per, hms(per * 100 / 3600.0))

    if finalized:
        state = "DONE"
        where = ""
        prog, rate_txt = "finalized", ""
        if facts.get("recovery"):
            notes.insert(0, facts["recovery"])
    elif running:
        j = running[0]
        elapsed = parse_elapsed(j["elapsed"])
        gpu = short_gpu(facts.get("gpu") or j["gres"].replace("gres/gpu:", "").replace("gres/", ""))
        seg_left = max(SEGMENT_H - elapsed, 0)
        speed_now = meta.get("gpu_speed_vs_a100") or 1.0
        where = "%s %s %s, %s left" % (j["cluster"], j["node"], gpu, hms(seg_left))
        state = "FINALIZING" if phase == "done" else "RUNNING"
        if rows:
            since = (now - dt.datetime.fromtimestamp(rows[-1]["t"])).total_seconds() / 60.0
            limit = max(STALL_FLOOR_MIN, STALL_FACTOR * (gap or 0) / 60.0)
            if state == "RUNNING" and since > limit:
                util = gpu_busy(j["cluster"], j["id"])
                if util is not None and util >= 50:
                    state = "SLOW"
                    notes.append("no checkpoint for %.0f min, GPU %d%% busy" % (since, util))
                else:
                    state = "STALLED"
                    notes.append("no checkpoint for %.0f min, GPU %s" % (
                        since, "idle (%d%%)" % util if util is not None else "not readable"))
        elif elapsed > 0.75:
            notes.append("no checkpoint yet after %s" % hms(elapsed))
    elif pending:
        state = "QUEUED"
        starts = [j["start"] for j in pending if j["start"]]
        where = "%d queued" % len(pending)
        if starts:
            est_start = min(starts)
            where += ", est. %s" % est_start.strftime("%a %H:%M")
        reasons = sorted({j["reason"] for j in pending})
        where += " (%s)" % ", ".join(reasons)
    else:
        reason = status.get("stopped_reason")
        where = ""
        if reason in ("total_time_budget", "sampling_time_budget"):
            state = "CAPPED"
            notes.append("hit its compute cap (%s)" % reason)
        elif facts.get("error"):
            state = "FAILED"
            notes.append(facts["error"])
        elif meta or rows:
            state = "RESUBMIT"
            notes.append("stopped (%s); rerun tier1/submit_stage2.sh" % (reason or "no job"))
        else:
            state = "NOT STARTED"
    if facts.get("error") and state in ("RUNNING", "SLOW", "STALLED"):
        notes.append(facts["error"])
    check = facts.get("check", "") if not finalized else ""
    a100 = ("%.1f/%s" % (used, "%g" % cap if cap else "-")) if (meta or finalized) else ""
    return dict(group=group_of(run), run=run[len("Tier1 "):], state=state, where=where, prog=prog,
                rate=rate_txt, check=check, a100=a100, eta="", note="; ".join(notes), log=log_path,
                used=used, est_start=est_start, reasons=reasons if pending else [], seg_left=seg_left,
                speed_now=speed_now, full_run=run)


def add_estimates(lines, now):
    """Fill A100-H with the whole-run estimate and ETA with the finish time; the table still
    prints if the estimator fails."""
    try:
        import cost_estimate
        running = {r["full_run"] for r in lines if r["state"] in ("RUNNING", "SLOW", "STALLED")}
        est = {e["run"]: e for e in cost_estimate.all_estimates(running=running, now=now)[0]}
        acc = cost_estimate.start_accuracy() if any(r["est_start"] for r in lines) else None
    except Exception as exc:
        for r in lines:
            r["eta"] = ""
        if lines:
            lines[0]["note"] = ("; " if lines[0]["note"] else "").join(
                [lines[0]["note"], "cost_estimate failed: %s" % str(exc)[:60]]).lstrip("; ")
        return
    for r in lines:
        e = est.get(r["full_run"])
        if not e or e.get("total") is None:
            continue
        if r["state"] in ("DONE", "CAPPED"):
            r["a100"] = "%.1f" % e["total"]
            continue
        r["a100"] = "%.1f of ~%.1f%s" % (r["used"], e["total"], " CAP" if e.get("capped") else "")
        left = max(e["compute"] - r["used"], 0)
        if r["state"] in ("RUNNING", "SLOW", "STALLED") and r["seg_left"] is not None:
            wall = left / r["speed_now"]
            if wall <= r["seg_left"]:
                done = now + dt.timedelta(hours=wall + e["finalize"] / r["speed_now"])
                r["eta"] = "done ~%s" % done.strftime("%a %H:%M")
            else:
                more = int(-(-(wall - r["seg_left"]) // SEGMENT_H))
                seg_end = now + dt.timedelta(hours=r["seg_left"])
                r["eta"] = "segment ends %s, then %d more" % (seg_end.strftime("%a %H:%M"), more)
        elif r["state"] in ("QUEUED", "RESUBMIT", "NOT STARTED"):
            r["eta"] = "needs ~%s once started" % hms(left + e["finalize"])
            start = cost_estimate.corrected_start(r["est_start"], now, acc)
            if start:
                # Slurm's estimate shifted by how far off its estimates this far ahead have been.
                r["eta"] = "starts ~%s, needs ~%s" % (start.strftime("%a %H:%M"), hms(left + e["finalize"]))


def log_starts(lines, now):
    """Append each queued run's earliest estimated start (None when Slurm gives none)."""
    queued = {r["full_run"]: [r["est_start"].strftime("%Y-%m-%dT%H:%M:%S") if r["est_start"] else None,
                              r["reasons"]] for r in lines if r["state"] == "QUEUED"}
    if not queued:
        return
    try:
        with open(START_LOG, "a") as fh:
            fh.write(json.dumps({"t": now.strftime("%Y-%m-%dT%H:%M:%S"), "est": queued}) + "\n")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="run groups (R0-R8) or run-name substrings")
    ap.add_argument("--all", action="store_true", help="include runs never submitted")
    ap.add_argument("--check", action="store_true", help="exit 3 if anything needs action, 4 if all done")
    ap.add_argument("--logs", action="store_true", help="also print each run's log path")
    ap.add_argument("--no_log", action="store_true", help="do not log the estimated starts")
    a = ap.parse_args()

    now = dt.datetime.now()
    jobs = slurm_jobs()
    preempt = preemptions()
    runs = sorted(d for d in os.listdir(RESULTS) if d.startswith("Tier1 ")
                  and os.path.exists(os.path.join(RESULTS, d, "solver_params.json")))
    lines = []
    for run in runs:
        if a.filters and not any(f == group_of(run) or f in run for f in a.filters):
            continue
        r = describe(run, jobs, preempt, now)
        if r["state"] == "NOT STARTED" and not a.all:
            continue
        lines.append(r)
    order = {g: i for i, g in enumerate(["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "--"])}
    lines.sort(key=lambda r: (order.get(r["group"], 99), r["run"]))
    add_estimates(lines, now)
    if not a.no_log:
        log_starts(lines, now)

    cols = [("group", "GRP"), ("run", "RUN"), ("state", "STATE"), ("where", "WHERE"), ("prog", "PROGRESS"),
            ("rate", "RATE"), ("check", "LAST CHECK"), ("a100", "A100-H"), ("eta", "ETA"), ("note", "NOTE")]
    widths = {k: max([len(h)] + [len(r[k]) for r in lines]) for k, h in cols}
    print("Tier-1 status, %s" % now.strftime("%a %Y-%m-%d %H:%M"))
    print("  ".join(h.ljust(widths[k]) for k, h in cols).rstrip())
    for r in lines:
        print("  ".join(r[k].ljust(widths[k]) for k, _ in cols).rstrip())
        if a.logs and r["log"]:
            print("      log: %s" % r["log"])

    counts = {}
    for r in lines:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print("\n" + ", ".join("%d %s" % (n, s) for s, n in sorted(counts.items(), key=lambda kv: -kv[1])))
    seen = sorted({reason for r in lines for reason in re.findall(r"\b(%s)\b" % "|".join(REASONS), r["where"])})
    if seen:
        print("Queue reasons: " + "; ".join("%s = %s" % (k, REASONS[k]) for k in seen))
    act = [r for r in lines if r["state"] in ("RESUBMIT", "FAILED", "STALLED")]
    if act:
        print("Needs action: " + ", ".join("%s (%s)" % (r["run"], r["state"]) for r in act))
        if any(r["state"] == "RESUBMIT" for r in act):
            print("  resubmit: cd %s/job_files && tier1/submit_stage2.sh" % BASE)
    if a.check:
        if act:
            sys.exit(3)
        if lines and all(r["state"] in ("DONE", "CAPPED") for r in lines):
            sys.exit(4)


if __name__ == "__main__":
    main()
