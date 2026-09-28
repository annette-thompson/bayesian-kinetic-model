"""A100-hour estimates for every Tier-1 run, rebuilt from the rates measured so far.

The sampler solves every condition to the latest save time in the data (720 s), so a condition
observed only at 150 s costs as much as the time series. The estimates here come only from
runs measured under that, and they change as more runs report.

  python3 tier1/cost_estimate.py            # every run: basis, rates, used and estimated A100-h,
                                            # then totals per run group and per stage
  python3 tier1/cost_estimate.py R2 R6      # filter: run groups (R0-R8) or name substrings
  python3 tier1/cost_estimate.py --log      # also append the totals to cost_estimate_history.jsonl
                                            # and write Results/Tier1/figures/cost_estimate.json
  python3 tier1/cost_estimate.py --starts   # also list every run segment's start estimates

A run's estimate, in A100-equivalent hours (wall time x the GPU's speed factor, as the sampler
counts its compute cap), is
    startup per segment + warmup steps x s/step + draws x s/draw + finalize
and each rate comes from the most specific measurement there is:
  own     the run's own progress log
  class   the median over measured runs on the same system with the same free parameters,
          leaving out solver variants (ta0.95, rtol1e-5, dense) and data subsets (profile, rates)
  scaled  nothing in the class has been measured: the a1+c3 class on the same system, times the
          three-parameter pilot factors (3.0x per warmup step, 3.5x per draw, 2.24x the draws) for
          a three-parameter run
Before a run samples, its s/draw is its own warmup rate times the sampling-to-warmup ratio of the
runs that have done both. Draws to finish: the median over finalized runs with the same number of
free parameters, and at least 200 more than a sampling run has drawn (two passing checks and one
more block). Finalize is measured per draw on finalized runs and scaled to other systems by their
per-step cost. Compute stops at the run's cap (max_total_hours); a run the estimate puts past it
is marked CAP.

Start estimates: tier1_status.py logs, on every call, each queued run's earliest estimated start
(Slurm's, the "est." in its WHERE column) to start_predictions.jsonl. This matches every logged
estimate with the run's next actual start in sacct (any copy, either cluster) and reports the
error by how far ahead the estimate was: error = actual - estimated, so positive means the run
started later than Slurm said.

Written for the login node's Python 3.6: standard library only.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys

BASE = "/projects/anth4580/Bayesian"
RESULTS = os.path.join(BASE, "Results", "Tier1")
HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "cost_estimate_history.jsonl")
START_LOG = os.path.join(HERE, "start_predictions.jsonl")
LEAD_BUCKETS = [(0, 1, "<1 h"), (1, 3, "1-3 h"), (3, 6, "3-6 h"), (6, 12, "6-12 h"), (12, 24, "12-24 h"),
                (24, 1e9, ">24 h")]
OUT_JSON = os.path.join(RESULTS, "figures", "cost_estimate.json")

SEGMENT_H = 11.5      # tier1.sbatch's per-segment budget (MAXH)
STARTUP_H = 0.2       # per segment, job start to the sampler's first steps (R0: 0.19 h)
AFTER_H = 0.1         # recovery report and figures after finalize; placeholder until measured
MIN_EXTRA_DRAWS = 200
THREE_PARAM = {"warmup": 3.0, "sampling": 3.5, "draws": 2.24}   # pilot medians, appendix B
VARIANT = re.compile(r" - (ta0\.95|rtol1e-5|dense|profile|rates)$")

# Kept in step with resumable_sampler.GPU_SPEED_VS_A100; a card it does not list counts at 1.00,
# as the sampler counts it.
GPU_SPEED_VS_A100 = (("H100 NVL", 1.37), ("H200", 1.23), ("MIG 3g.40gb", 0.90), ("A100", 1.00),
                     ("V100-SXM2", 0.67), ("V100", 0.75))

STAGES = [("Stage 1", lambda g, r: g == "R0"),
          ("Stage 2", lambda g, r: g in ("R1", "R3", "R5", "R7", "R8") or (g == "R4" and _sbc(r) < 10)),
          ("Stage 3", lambda g, r: g in ("R2", "R6") or (g == "R4" and _sbc(r) >= 10))]

sys.path.insert(0, HERE)
from tier1_status import (cluster_env, group_of, is_finalized, parse_time, progress, read_json, sh,  # noqa: E402
                          slug_of, slurm_jobs)


def _sbc(run):
    m = re.search(r"_sbc(\d+)", run)
    return int(m.group(1)) if m else -1


def speed(gpu):
    for key, factor in GPU_SPEED_VS_A100:
        if gpu and key in gpu:
            return factor
    return 1.0


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def system_of(run):
    s = run[len("Tier1 "):].split(" - ")[0]
    s = re.sub(r"_(sbc|noise)\d+$", "", s)
    return s.replace("+c3split_c3l3", "")


def phase_rate(rows, phase, open_until=None, every=5):
    """(A100-equivalent seconds per step, steps measured) from consecutive checkpoints of one phase.
    A gap far longer than the usual interval (queue wait plus startup) is a segment boundary or a
    preemption, and is left out. For a running run (`open_until` = now) whose latest checkpoint is
    in this phase, the time since it counts as one more chunk of `every` steps once it is longer
    than the usual interval: a lower bound on a chunk that is taking longer than the rest."""
    key = "warmup_done" if phase == "warmup" else "sampling_done"
    r = [x for x in rows if x.get("phase") == phase]
    pairs = [(a, b) for a, b in zip(r, r[1:]) if b[key] > a[key] and b["t"] > a["t"]]
    if not pairs:
        return None, 0
    usual = median([b["t"] - a["t"] for a, b in pairs])
    keep = [(a, b) for a, b in pairs if b["t"] - a["t"] <= max(8 * usual, 1200)]
    steps = sum(b[key] - a[key] for a, b in keep)
    secs = sum((b["t"] - a["t"]) * speed(b.get("gpu")) for a, b in keep)
    if open_until and r and r[-1] is rows[-1] and open_until - r[-1]["t"] > usual:
        steps += every
        secs += (open_until - r[-1]["t"]) * speed(r[-1].get("gpu"))
    return (secs / steps if steps else None), steps


def measure(run, running=False, now=None, job_ends=None):
    d = os.path.join(RESULTS, run)
    cfg = read_json(os.path.join(d, "solver_params.json")) or {}
    ps = cfg.get("posterior_sampling") or {}
    meta = read_json(os.path.join(d, "checkpoint", "checkpoint_meta.json")) or {}
    status = read_json(os.path.join(d, "checkpoint", "status.json")) or {}
    rows = progress(d)
    m = dict(run=run, group=group_of(run), system=system_of(run),
             params=[p["param_name"] for p in cfg.get("free_kinetic_params", [])],
             tune=ps.get("tune", 300), cap=ps.get("max_total_hours"),
             used=(meta.get("a100_equiv_seconds") or 0) / 3600.0,
             warmup_done=meta.get("warmup_done", 0), sampling_done=meta.get("sampling_done", 0),
             phase=meta.get("phase"), n_draws=meta.get("n_draws"), max_draws=ps.get("draws"),
             segments=status.get("n_invocations", 1 if meta else 0), gpu=meta.get("gpu"),
             finalized=is_finalized(d, meta))
    # Reopened for more draws: its earlier posterior file stays until it finalizes again, and its
    # n_draws is a ceiling, not the final count a converged run sets.
    m["reopened"] = os.path.exists(os.path.join(d, "posterior_samples_pm.nc")) and not m["finalized"]
    m["key"] = (m["system"], "+".join(m["params"]))
    m["pool"] = not VARIANT.search(run)
    m["sbc"] = "_sbc" in run
    open_until = (now or dt.datetime.now()).timestamp() if running else None
    every = ps.get("checkpoint_every_steps", 5)
    m["warmup"], m["warmup_steps"] = phase_rate(rows, "warmup", open_until, every)
    m["sampling"], m["sampling_steps"] = phase_rate(rows, "sampling", open_until, every)
    m["finalize"] = m["after"] = None
    if m["finalized"] and rows:
        nc = os.path.getmtime(os.path.join(d, "posterior_samples_pm.nc"))
        m["finalize"] = max(nc - rows[-1]["t"], 0) / 3600.0 * speed(m["gpu"])
        # Recovery report and figures: from the posterior file to the end of the job that wrote
        # it (sacct), not to a figure's timestamp, which a later redraw moves.
        end = next((e for e in (job_ends or {}).get(slug_of(run), []) if e.timestamp() >= nc), None)
        if end is not None and end.timestamp() - nc < 3 * 3600:
            m["after"] = (end.timestamp() - nc) / 3600.0 * speed(m["gpu"])
    return m


class Rates:
    """Class-level rates from the measured runs, with the fallbacks in the module docstring."""

    def __init__(self, ms):
        self.ms = ms
        both = [m["sampling"] / m["warmup"] for m in ms
                if m["warmup"] and m["warmup_steps"] >= 250 and m["sampling"] and m["sampling_steps"] >= 100]
        self.samp_per_warm = median(both)
        self.n_ratio = len(both)
        done = [m for m in ms if m["finalized"]]
        self.draws = {}
        for m in done:
            self.draws.setdefault(len(m["params"]), []).append(m["sampling_done"])
        per_draw = [(m, m["finalize"] / m["sampling_done"]) for m in done if m["finalize"] and m["sampling_done"]]
        self.fin_per_draw = {}
        for m, f in per_draw:
            self.fin_per_draw.setdefault(m["system"], []).append(f)
        afters = [m["after"] for m in done if m["after"] is not None]
        self.after = median(afters) if afters else AFTER_H
        self.after_measured = bool(afters)

    def per_step(self, m):
        return m["warmup"] if m["warmup"] and m["warmup_steps"] >= 25 else None

    def per_draw(self, m):
        """A measured run's s/draw: sampled, or its warmup rate times the sampling-to-warmup ratio."""
        if m["sampling"] and m["sampling_steps"] >= 100:
            return m["sampling"]
        if self.per_step(m) and self.samp_per_warm:
            return m["warmup"] * self.samp_per_warm
        return None

    def pooled(self, key, phase, sbc):
        """Median over the class, from the SBC replicates or the fixed-truth runs as the run is one
        or the other (the SBC truths reach far from the prior's centre), else from both."""
        fn = self.per_step if phase == "warmup" else self.per_draw
        cands = [m for m in self.ms if m["pool"] and m["key"] == key]
        same = [m for m in cands if m["sbc"] == sbc]
        for group, label in ((same, "SBC" if sbc else "fixed-truth"), (cands, "all")):
            xs = [v for v in (fn(m) for m in group) if v]
            if xs:
                return median(xs), "class (%d %s)" % (len(xs), label)
        return None, None

    def rate(self, system, params, phase, sbc=False):
        """(s per step or draw, basis) for a run with nothing of its own."""
        key = (system, "+".join(params))
        v, b = self.pooled(key, phase, sbc)
        if v:
            return v, b
        if key[1] != "a1+c3":
            v, b = self.rate(system, ["a1", "c3"], phase, sbc)
            if v:
                if len(params) == 3:
                    return v * THREE_PARAM[phase], "%s a1+c3 %s x 3-param" % (system, b)
                return v, "%s a1+c3 %s" % (system, b)
        return None, "no measurement"

    def expected_draws(self, nparams):
        if self.draws.get(nparams):
            return median(self.draws[nparams]), "measured (%d)" % len(self.draws[nparams])
        if nparams == 3 and self.draws.get(2):
            return median(self.draws[2]) * THREE_PARAM["draws"], "2-param x 2.24"
        return 800, "R0"

    def finalize_per_draw(self, system):
        if self.fin_per_draw.get(system):
            return median(self.fin_per_draw[system])
        # Scale a measured system's per-draw finalize by the two systems' a1+c3 warmup cost.
        for other, fs in self.fin_per_draw.items():
            here, _ = self.rate(system, ["a1", "c3"], "warmup")
            there, _ = self.rate(other, ["a1", "c3"], "warmup")
            if here and there:
                return median(fs) * here / there
        return None


def estimate(m, rates):
    e = dict(m)
    if m["finalized"]:
        e.update(compute=m["used"], finalize=(m["finalize"] or 0) + (m["after"] if m["after"] is not None else 0),
                 basis="measured", warm_rate=m["warmup"], samp_rate=m["sampling"], draws=m["sampling_done"],
                 capped=False)
        e["total"] = e["need"] = e["compute"] + e["finalize"]
        return e
    bases = []
    if m["warmup"] and m["warmup_steps"] >= 10:
        warm, wb = m["warmup"], "own"
    else:
        warm, wb = rates.rate(m["system"], m["params"], "warmup", m["sbc"])
    if m["sampling"] and m["sampling_steps"] >= 20:
        samp, sb = m["sampling"], "own"
    elif m["warmup"] and m["warmup_steps"] >= 10 and rates.samp_per_warm:
        samp, sb = m["warmup"] * rates.samp_per_warm, "own warmup x ratio"
    else:
        samp, sb = rates.rate(m["system"], m["params"], "sampling", m["sbc"])
    draws, db = rates.expected_draws(len(m["params"]))
    if m["phase"] == "done":
        # Sampled everything; only finalize is left.
        draws, db = m["sampling_done"], "sampled"
    elif m["n_draws"] and m["max_draws"] and m["n_draws"] < m["max_draws"] and not m["reopened"]:
        # Converged: the sampler has set its final count (the confirmation blocks).
        draws, db = m["n_draws"], "converged"
    elif m["sampling_done"]:
        draws = max(draws, int(math.ceil((m["sampling_done"] + MIN_EXTRA_DRAWS) / 100.0)) * 100)
        if m["reopened"] and m["n_draws"]:
            draws = min(draws, m["n_draws"])
    for label, b in (("warmup", wb), ("draw", sb), ("draws", db)):
        bases.append("%s %s" % (label, b))
    if warm is None or samp is None:
        e.update(compute=None, finalize=None, total=None, need=None, basis="; ".join(bases), warm_rate=warm,
                 samp_rate=samp, draws=draws, capped=False)
        return e
    left = (max(m["tune"] - m["warmup_done"], 0) * warm + max(draws - m["sampling_done"], 0) * samp) / 3600.0
    compute = m["used"] + left
    n_seg = max(int(math.ceil(compute / SEGMENT_H)), 1)
    compute += max(n_seg - m["segments"], 0) * STARTUP_H
    capped = bool(m["cap"]) and compute > m["cap"]
    fpd = rates.finalize_per_draw(m["system"])
    finalize = (fpd * draws if fpd else 0) + rates.after
    e.update(compute=min(compute, m["cap"]) if capped else compute, uncapped=compute, finalize=finalize,
             basis="; ".join(bases), warm_rate=warm, samp_rate=samp, draws=draws, capped=capped)
    e["total"] = e["compute"] + finalize
    e["need"] = compute + finalize
    return e


def stage_of(e):
    for name, test in STAGES:
        if test(e["group"], e["run"]):
            return name
    return "--"


def job_ends(days=14):
    """{job name: [end, ...] sorted} for this user's completed jobs on both clusters."""
    ends = {}
    for cl in ("blanca", "alpine"):
        # Filtered here: sacct's -s with -S returns nothing on these clusters.
        out = sh(["sacct", "-u", os.environ.get("USER", ""), "-S", "now-%ddays" % days, "-X", "-n", "-P",
                  "-o", "JobName,End,State"], env=cluster_env(cl))
        for line in out.splitlines():
            f = line.split("|")
            t = parse_time(f[1]) if len(f) == 3 and f[2].startswith("COMPLETED") else None
            if t:
                ends.setdefault(f[0], []).append(t)
    for v in ends.values():
        v.sort()
    return ends


def running_runs():
    """Runs with a copy running on either cluster, from squeue."""
    jobs = slurm_jobs()
    return {r for r in os.listdir(RESULTS) if r.startswith("Tier1 ")
            and any(j["state"] in ("RUNNING", "COMPLETING") for j in jobs.get(slug_of(r), []))}


def all_estimates(running=None, now=None):
    runs = sorted(d for d in os.listdir(RESULTS) if d.startswith("Tier1 ")
                  and os.path.exists(os.path.join(RESULTS, d, "solver_params.json")))
    if running is None:
        running = running_runs()
    ends = job_ends()
    ms = [measure(r, r in running, now, ends) for r in runs]
    rates = Rates(ms)
    es = [estimate(m, rates) for m in ms]
    for e in es:
        e["stage"] = stage_of(e)
    return es, rates


def actual_starts(since):
    """{job name: [(start, cluster), ...] sorted}, every start on both clusters since `since`
    (a requeued job has one record per start)."""
    starts = {}
    for cl in ("blanca", "alpine"):
        out = sh(["sacct", "-u", os.environ.get("USER", ""), "-S", since.strftime("%Y-%m-%dT%H:%M:%S"), "-X", "-n",
                  "-P", "--duplicates", "-o", "JobName,Start"], env=cluster_env(cl))
        for line in out.splitlines():
            f = line.split("|")
            t = parse_time(f[1]) if len(f) == 2 else None
            if t:
                starts.setdefault(f[0], []).append((t, cl))
    for v in starts.values():
        v.sort()
    return starts


def lead_bucket(hours):
    for lo, hi, label in LEAD_BUCKETS:
        if lo <= hours < hi:
            return label
    return LEAD_BUCKETS[0][2]


def start_accuracy():
    """Every logged estimate matched with the run's next actual start.

    Returns {"segments": [...], "buckets": {label: {...}}}, or None with nothing logged. A segment
    is one wait in the queue: a run's estimates up to one actual start (or still waiting). Buckets
    go by the estimate's lead (estimated start - time of the estimate), which is what a reader of
    the status table sees; each segment counts once per bucket, with its median error there."""
    preds = []
    try:
        with open(START_LOG) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                t = parse_time(row["t"])
                for run, (est, reasons) in row["est"].items():
                    preds.append((t, run, parse_time(est) if est else None, reasons))
    except OSError:
        return None
    if not preds:
        return None
    starts = actual_starts(min(p[0] for p in preds) - dt.timedelta(days=1))
    segs = {}
    for t, run, est, reasons in sorted(preds, key=lambda p: p[0]):
        nxt = next((x for x in starts.get(slug_of(run), []) if x[0] >= t), None)
        key = (run, nxt[0] if nxt else None)
        seg = segs.setdefault(key, dict(run=run, actual=nxt[0] if nxt else None, cluster=nxt[1] if nxt else None,
                                        preds=[]))
        seg["preds"].append((t, est, reasons))
    out = []
    per_bucket = {}
    for seg in segs.values():
        with_est = [(t, est) for t, est, _ in seg["preds"] if est]
        seg["n"], seg["n_est"] = len(seg["preds"]), len(with_est)
        seg["first_est"] = with_est[0][1] if with_est else None
        seg["last_est"] = with_est[-1][1] if with_est else None
        seg["first_t"] = seg["preds"][0][0]
        if seg["actual"] and with_est:
            errs = {}
            for t, est in with_est:
                lead = (est - t).total_seconds() / 3600.0
                errs.setdefault(lead_bucket(lead), []).append((seg["actual"] - est).total_seconds() / 3600.0)
            for b, es in errs.items():
                per_bucket.setdefault(b, []).append(median(es))
            seg["first_err"] = (seg["actual"] - with_est[0][1]).total_seconds() / 3600.0
            seg["last_err"] = (seg["actual"] - with_est[-1][1]).total_seconds() / 3600.0
            seg["first_lead"] = (with_est[0][1] - with_est[0][0]).total_seconds() / 3600.0
            seg["last_lead"] = (with_est[-1][1] - with_est[-1][0]).total_seconds() / 3600.0
        out.append(seg)
    buckets = {}
    for lo, hi, label in LEAD_BUCKETS:
        es = per_bucket.get(label)
        if es:
            buckets[label] = dict(n=len(es), median_err=median(es), median_abs=median([abs(x) for x in es]),
                                  within_1h=sum(abs(x) <= 1 for x in es) / float(len(es)),
                                  late=sum(x > 0 for x in es) / float(len(es)))
    return dict(segments=out, buckets=buckets, n_polls=len({p[0] for p in preds}),
                since=min(p[0] for p in preds))


def print_start_accuracy(acc, detail):
    print("\nSlurm start estimates vs actual starts (error = actual - estimated; + means later than Slurm said)")
    if not acc:
        print("  nothing logged yet (tier1_status.py logs on every call)")
        return
    done = [g for g in acc["segments"] if g["actual"]]
    waiting = [g for g in acc["segments"] if not g["actual"]]
    print("  %d polls since %s; %d queue waits ended in a start, %d still waiting" % (
        acc["n_polls"], acc["since"].strftime("%a %H:%M"), len(done), len(waiting)))
    if acc["buckets"]:
        print("  %-8s  %5s  %10s  %10s  %9s  %6s" % ("LEAD", "WAITS", "MEDIAN ERR", "MEDIAN |E|", "WITHIN 1H", "LATE"))
        for lo, hi, label in LEAD_BUCKETS:
            b = acc["buckets"].get(label)
            if b:
                print("  %-8s  %5d  %+9.1fh  %9.1fh  %8.0f%%  %5.0f%%" % (
                    label, b["n"], b["median_err"], b["median_abs"], 100 * b["within_1h"], 100 * b["late"]))
    no_est = sum(g["n"] - g["n_est"] for g in acc["segments"])
    total = sum(g["n"] for g in acc["segments"])
    if no_est:
        print("  %d of %d logged polls had no estimate from Slurm" % (no_est, total))
    drift = [g for g in waiting if g["first_est"] and g["last_est"]]
    if drift:
        moved = sorted((g["last_est"] - g["first_est"]).total_seconds() / 3600.0 for g in drift)
        print("  still waiting (%d runs): since first logged, the estimates have moved %+.1f to %+.1f h, "
              "median |move| %.1f h" % (len(drift), moved[0], moved[-1], median([abs(x) for x in moved])))
    if detail:
        print("  %-38s %-7s %-11s %-11s %-11s %8s %8s %6s" % ("RUN", "CLUSTER", "FIRST SEEN", "FIRST EST", "STARTED",
                                                             "ERR 1ST", "ERR LAST", "POLLS"))
        for g in sorted(acc["segments"], key=lambda g: (g["actual"] is None, g["actual"] or g["first_t"], g["run"])):
            f = lambda d: d.strftime("%a %H:%M") if d else "-"
            print("  %-38s %-7s %-11s %-11s %-11s %8s %8s %6d" % (
                g["run"][len("Tier1 "):][:38], g["cluster"] or "", f(g["first_t"]), f(g["first_est"]),
                f(g["actual"]) if g["actual"] else "waiting",
                "%+.1fh" % g["first_err"] if g.get("first_err") is not None else "",
                "%+.1fh" % g["last_err"] if g.get("last_err") is not None else "", g["n"]))


def corrected_start(est, now, acc):
    """Slurm's estimate shifted by the median error of estimates made as far ahead, once three or
    more queue waits in that bucket have ended; else None."""
    if not acc or not est:
        return None
    b = acc["buckets"].get(lead_bucket((est - now).total_seconds() / 3600.0))
    if not b or b["n"] < 3:
        return None
    return est + dt.timedelta(hours=b["median_err"])


def fmt(x, spec="%.1f"):
    return "" if x is None else spec % x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="run groups (R0-R8) or run-name substrings")
    ap.add_argument("--log", action="store_true", help="append totals to the history and write the JSON")
    ap.add_argument("--starts", action="store_true", help="list every run segment's start estimates")
    a = ap.parse_args()
    now = dt.datetime.now()
    es, rates = all_estimates()
    shown = [e for e in es if not a.filters or any(f == e["group"] or f in e["run"] for f in a.filters)]
    order = {g: i for i, g in enumerate(["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "--"])}
    shown.sort(key=lambda e: (order.get(e["group"], 99), e["run"]))

    print("Tier-1 compute estimates (A100-equivalent hours), %s" % now.strftime("%a %Y-%m-%d %H:%M"))
    print("Every condition solved to 720 s. Sampling-to-warmup cost ratio %s (from %d runs); draws to finish %s; "
          "figures after finalize %s h (%s)." % (
              fmt(rates.samp_per_warm, "%.2f"), rates.n_ratio,
              ", ".join("%d-param %s" % (k, "/".join(str(x) for x in v)) for k, v in sorted(rates.draws.items())),
              fmt(rates.after, "%.2f"), "measured" if rates.after_measured else "placeholder"))
    cols = [("group", "GRP"), ("run", "RUN"), ("state", "STATE"), ("warm", "S/STEP"), ("samp", "S/DRAW"),
            ("draws", "DRAWS"), ("used", "USED"), ("total", "EST"), ("note", "NOTE"), ("basis", "BASIS")]
    lines = []
    for e in shown:
        state = "done" if e["finalized"] else ("started" if e["used"] else "")
        note = ("CAP: needs ~%.0f, stops at %g" % (e["need"], e["cap"])) if e.get("capped") else ""
        row = dict(group=e["group"], run=e["run"][len("Tier1 "):], state=state, warm=fmt(e["warm_rate"], "%.0f"),
                   samp=fmt(e["samp_rate"], "%.1f"), draws=fmt(e["draws"], "%.0f"),
                   used=fmt(e["used"]) if e["used"] else "", total=fmt(e["total"]), note=note, basis=e["basis"])
        # Unstarted runs of one group with the same estimate share a line.
        same = ("group", "state", "warm", "samp", "draws", "total", "note", "basis")
        if lines and not state and all(lines[-1][k] == row[k] for k in same):
            lines[-1]["n"] = lines[-1].get("n", 1) + 1
            lines[-1]["last"] = row["run"]
            continue
        lines.append(row)
    for r in lines:
        if r.get("n", 1) >= 3:
            first, last = r["run"], r["last"]
            if "_sbc" in first and "_sbc" in last:
                first, last = first.split(" - ")[0], last.split(" - ")[0]
            r["run"] = "%s ... %s (%d runs)" % (first, last, r["n"])
            r["total"] = "%s each" % r["total"]
        elif r.get("n") == 2:
            r["run"] = "%s; %s" % (r["run"], r["last"])
            r["total"] = "%s each" % r["total"]
    widths = {k: max([len(h)] + [len(r[k]) for r in lines]) for k, h in cols}
    print("  ".join(h.ljust(widths[k]) for k, h in cols).rstrip())
    for r in lines:
        print("  ".join(r[k].ljust(widths[k]) for k, _ in cols).rstrip())

    def totals(key):
        t = {}
        for e in shown:
            u, s_, n, need = t.get(e[key], (0.0, 0.0, 0, 0.0))
            t[e[key]] = (u + e["used"], s_ + (e["total"] or 0), n + 1, need + (e["need"] or 0))
        return t

    def summary(t):
        return "   ".join("%s %d runs: %.0f used, ~%.0f est%s" % (
            g, n, u, s_, " (~%.0f uncapped)" % need if need > s_ + 0.5 else "")
            for g, (u, s_, n, need) in sorted(t.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])))
    print("\nBy group:  " + summary(totals("group")))
    print("By stage:  " + summary(totals("stage")))
    print("USED = compute so far, as the sampler counts it. EST = the whole run: compute (up to its cap), finalize "
          "and figures.")
    acc = start_accuracy()
    print_start_accuracy(acc, a.starts)

    if a.log and not a.filters:
        def rounded(t):
            return {k: {"used": round(u, 2), "est": round(s_, 2), "uncapped": round(need, 2), "runs": n}
                    for k, (u, s_, n, need) in t.items()}
        stages, groups = rounded(totals("stage")), rounded(totals("group"))
        with open(HISTORY, "a") as fh:
            fh.write(json.dumps({"t": now.strftime("%Y-%m-%dT%H:%M:%S"), "stages": stages, "groups": groups,
                                 "runs": {e["run"]: [round(e["used"], 2), None if e["total"] is None else round(e["total"], 2)]
                                          for e in es}}) + "\n")
        keep = ("run", "group", "stage", "system", "params", "finalized", "used", "compute", "finalize", "total",
                "need", "warm_rate", "samp_rate", "draws", "capped", "cap", "basis")
        out = {"generated": now.strftime("%Y-%m-%dT%H:%M:%S"), "samp_per_warm": rates.samp_per_warm,
               "draws": rates.draws, "after_h": rates.after, "after_measured": rates.after_measured,
               "stages": stages, "groups": groups, "runs": [{k: e.get(k) for k in keep} for e in es],
               "start_accuracy": (acc or {}).get("buckets")}
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
