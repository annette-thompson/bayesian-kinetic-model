import json, time
from pathlib import Path
BASE = Path("/projects/anth4580/Bayesian/Results/Chain Scaling Tests")

def rows_of(run):
    f = BASE / run / "checkpoint" / "progress_log.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

def rate(rows, phase, lo, hi):
    """seconds per step for steps lo..hi of a phase, using only consecutive rows within one segment."""
    key = "warmup_done" if phase == "warmup" else "sampling_done"
    r = [x for x in rows if x["phase"] == phase and lo <= x[key] <= hi]
    tot_t = tot_s = 0.0
    for a, b in zip(r, r[1:]):
        dt, ds = b["t"] - a["t"], b[key] - a[key]
        if 0 < ds and dt < 3 * 3600:
            tot_t += dt; tot_s += ds
    return tot_t / tot_s if tot_s else None

def fmt(x):
    return f"{x:6.1f}" if x else "     -"

print(f"{'run':<34}{'now':>18}  s/step: warm 0-75  75-300  300-600  sampling  gpu")
for run in ["Chain C6 - a1c3_no_floor", "Chain C6 - a1c3d1_no_floor", "Chain C6 - a1c3a2_no_floor",
            "Archive/Chain C6 - a1c3d1_unscaled_no_floor",
            "Chain C10 - a1c3_no_floor", "Chain C10 - a1c3d1_no_floor", "Chain C10 - a1c3a2_no_floor",
            "Archive/Chain C10 - a1c3d1_unscaled_no_floor",
            "Chain C14 - a1c3_no_floor", "Chain C14 - a1c3d1_no_floor", "Chain C14 - a1c3a2_no_floor",
            "Archive/Chain C14 - a1c3d1_unscaled_no_floor"]:
    rows = rows_of(run)
    if not rows:
        print(f"{run:<34} no progress log"); continue
    last = rows[-1]
    now = f"w{last['warmup_done']} s{last['sampling_done']} {time.strftime('%H:%M', time.localtime(last['t']))}"
    print(f"{run[-34:]:<34}{now:>18}  {fmt(rate(rows,'warmup',0,75))}   {fmt(rate(rows,'warmup',75,300))}  {fmt(rate(rows,'warmup',300,600))}   {fmt(rate(rows,'sampling',0,10**6))}   {last.get('gpu') or last.get('device')}")
for s in ("C6", "C10", "C14"):
    c = json.load(open(BASE / f"Chain {s} - a1c3d1_no_floor" / "solver_params.json"))["posterior_sampling"]
    print(f"{s} a1c3d1 config: tune={c.get('tune')} draws={c.get('draws')} chains={c.get('chains')} max_total_hours={c.get('max_total_hours')}")
    m = BASE / f"Chain {s} - a1c3d1_no_floor" / "checkpoint" / "checkpoint_meta.json"
    print("   meta:", json.load(open(m)) if m.exists() else "none yet")
for run in ("Chain C14 - a1_0.05-20_no_floor", "Chain C10 - a1c3_no_floor", "Chain C18+unsat - a1 tightest nofloor-eqxnan"):
    d = BASE / run
    for fn in ("finalize_stage.nc", "prior_samples_pm.nc", "posterior_samples_pm.nc"):
        p = d / fn
        print(f"{run:<45} {fn:<24} {time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime)) if p.exists() else '-'}")
